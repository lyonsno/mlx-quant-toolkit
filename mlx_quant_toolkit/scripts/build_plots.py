#!/usr/bin/env python3
"""
Build a small deterministic set of plots from table artifacts.

Usage:
  python scripts/build_plots.py --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


_PLOT_INPUTS_MODULE = None

_UNSET = object()


def _local_helper_module_key(module_name: str) -> str:
    return f"{__name__}.__local__.{module_name}"


def _load_local_helper_module(module_name: str):
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    local_module_key = _local_helper_module_key(module_name)

    for key in (local_module_key, module_name):
        existing = sys.modules.get(key)
        if existing is None:
            continue
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None:
            continue
        try:
            if Path(existing_file).resolve() == module_path:
                return existing
        except Exception:
            continue

    spec = importlib.util.spec_from_file_location(local_module_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    had_prior_entry = local_module_key in sys.modules
    prior_entry = sys.modules.get(local_module_key)
    sys.modules[local_module_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        current = sys.modules.get(local_module_key)
        if current is module:
            if had_prior_entry:
                sys.modules[local_module_key] = prior_entry
            else:
                sys.modules.pop(local_module_key, None)
        raise

    return module


def _get_plot_inputs_module():
    global _PLOT_INPUTS_MODULE
    if _PLOT_INPUTS_MODULE is None:
        _PLOT_INPUTS_MODULE = _load_local_helper_module("plot_inputs")
    return _PLOT_INPUTS_MODULE


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install plotting dependencies before running build_plots."
        ) from exc
    return plt


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "analysis_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path} (run init_run.py first)")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {cfg_path}: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {cfg_path}: expected top-level JSON object")
    return cfg


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _resolve_requested_artifact_keys(cfg: dict[str, Any]) -> tuple[list[str], bool]:
    plots_section = cfg.get("plots", _UNSET)
    if plots_section is _UNSET or plots_section is None:
        return list(_DEFAULT_SOURCE_ARTIFACT_KEYS), False
    if not isinstance(plots_section, dict):
        raise ValueError("Invalid plots config: expected object or null")

    artifact_keys_raw = plots_section.get("artifact_keys", _UNSET)
    if artifact_keys_raw is _UNSET or artifact_keys_raw is None:
        return list(_DEFAULT_SOURCE_ARTIFACT_KEYS), False
    if not isinstance(artifact_keys_raw, list):
        raise ValueError("Invalid plots.artifact_keys config: expected a list of non-empty strings")

    parsed_keys: list[str] = []
    for idx, raw in enumerate(artifact_keys_raw):
        if not isinstance(raw, str):
            raise ValueError(
                f"Invalid plots.artifact_keys[{idx}] value: expected non-empty string, got {type(raw).__name__}"
            )
        key = raw.strip()
        if not key:
            raise ValueError(f"Invalid plots.artifact_keys[{idx}] value: expected non-empty string")
        parsed_keys.append(key)

    requested = _dedupe_preserve_order(parsed_keys)
    unsupported = [key for key in requested if key not in _PLOT_SPECS_BY_SOURCE]
    if unsupported:
        raise ValueError(f"Unsupported plot artifact keys: {', '.join(unsupported)}")

    return requested, True


def _error_text(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__


def _make_manifest_entry(
    *,
    source_artifact: str,
    status: str,
    path: str = "",
    error: str = "",
) -> dict[str, str]:
    return {
        "path": path,
        "source_artifact": source_artifact,
        "format": "png",
        "status": status,
        "error": error,
    }


def _write_plots_manifest(
    run_dir: Path,
    requested_artifact_keys: list[str],
    artifacts: dict[str, dict[str, str]],
) -> Path:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_artifact_keys": requested_artifact_keys,
        "artifacts": artifacts,
    }
    manifest_path = logs_dir / "plots_write_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _safe_read_json_dict(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text()
    except Exception:
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve_expert_layer_heatmap_config(cfg: dict[str, Any]) -> dict[str, str] | None:
    plots_section = cfg.get("plots", _UNSET)
    if plots_section is _UNSET or plots_section is None:
        # Default-on behavior: emit expert/layer heatmaps when matrix_stats supports it.
        return {"metric": "outlier_p999_over_median", "source_artifact": "matrix_stats"}
    if not isinstance(plots_section, dict):
        # _resolve_requested_artifact_keys will also raise this shape error.
        return {"metric": "outlier_p999_over_median", "source_artifact": "matrix_stats"}

    raw = plots_section.get("expert_layer_heatmaps", _UNSET)
    if raw is _UNSET or raw is None or raw is False:
        if raw is False:
            return None
        return {"metric": "outlier_p999_over_median", "source_artifact": "matrix_stats"}
    if raw is True:
        return {"metric": "outlier_p999_over_median", "source_artifact": "matrix_stats"}
    if not isinstance(raw, dict):
        raise ValueError("Invalid plots.expert_layer_heatmaps config: expected bool, object, or null")

    enabled = raw.get("enabled", True)
    if enabled is None or enabled is False:
        return None
    if not isinstance(enabled, bool):
        raise ValueError("Invalid plots.expert_layer_heatmaps.enabled config: expected bool")

    source_artifact_raw = raw.get("source_artifact", "matrix_stats")
    if not isinstance(source_artifact_raw, str) or not source_artifact_raw.strip():
        raise ValueError("Invalid plots.expert_layer_heatmaps.source_artifact config: expected non-empty string")
    source_artifact = source_artifact_raw.strip()
    if source_artifact not in {"matrix_stats", "quant_sim"}:
        raise ValueError(
            "Invalid plots.expert_layer_heatmaps.source_artifact config: expected 'matrix_stats' or 'quant_sim'"
        )

    metric_raw = raw.get("metric", "outlier_p999_over_median")
    if not isinstance(metric_raw, str) or not metric_raw.strip():
        raise ValueError("Invalid plots.expert_layer_heatmaps.metric config: expected non-empty string")
    return {"metric": metric_raw.strip(), "source_artifact": source_artifact}


def _resolve_quant_error_layer_lines_config(cfg: dict[str, Any]) -> dict[str, str] | None:
    plots_section = cfg.get("plots", _UNSET)
    if plots_section is _UNSET or plots_section is None:
        return None
    if not isinstance(plots_section, dict):
        return None

    raw = plots_section.get("quant_error_layer_lines", _UNSET)
    if raw is _UNSET or raw is None or raw is False:
        return None
    if raw is True:
        return {"metric": "w_rel_fro", "source_artifact": "quant_sim"}
    if not isinstance(raw, dict):
        raise ValueError("Invalid plots.quant_error_layer_lines config: expected bool, object, or null")

    enabled = raw.get("enabled", True)
    if enabled is None or enabled is False:
        return None
    if not isinstance(enabled, bool):
        raise ValueError("Invalid plots.quant_error_layer_lines.enabled config: expected bool")

    source_artifact_raw = raw.get("source_artifact", "quant_sim")
    if not isinstance(source_artifact_raw, str) or not source_artifact_raw.strip():
        raise ValueError("Invalid plots.quant_error_layer_lines.source_artifact config: expected non-empty string")
    source_artifact = source_artifact_raw.strip()
    if source_artifact != "quant_sim":
        raise ValueError("Invalid plots.quant_error_layer_lines.source_artifact config: expected 'quant_sim'")

    metric_raw = raw.get("metric", "w_rel_fro")
    if not isinstance(metric_raw, str) or not metric_raw.strip():
        raise ValueError("Invalid plots.quant_error_layer_lines.metric config: expected non-empty string")
    return {"metric": metric_raw.strip(), "source_artifact": source_artifact}


def _is_direct_plot_option_explicitly_enabled(cfg: dict[str, Any], option_key: str) -> bool:
    plots_section = cfg.get("plots", _UNSET)
    if not isinstance(plots_section, dict):
        return False

    raw = plots_section.get(option_key, _UNSET)
    if raw is _UNSET or raw is None or raw is False:
        return False
    if raw is True:
        return True
    if isinstance(raw, dict):
        enabled = raw.get("enabled", True)
        if enabled is None or enabled is False:
            return False
        if isinstance(enabled, bool):
            return enabled
        return True
    # Let the downstream config resolver surface the validation error instead of
    # treating an invalid explicit config as if the feature were absent.
    return True


def _has_explicit_direct_plot_requests(cfg: dict[str, Any]) -> bool:
    return _is_direct_plot_option_explicitly_enabled(cfg, "expert_layer_heatmaps") or _is_direct_plot_option_explicitly_enabled(
        cfg, "quant_error_layer_lines"
    )


def _validate_direct_plot_configs(cfg: dict[str, Any]) -> None:
    _resolve_expert_layer_heatmap_config(cfg)
    _resolve_quant_error_layer_lines_config(cfg)


def _read_collect_data_input(run_dir: Path, artifact_key: str) -> pd.DataFrame:
    manifest = _safe_read_json_dict(run_dir / "logs" / "write_manifest.json")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    candidate_paths: list[Path] = []
    entry = artifacts.get(artifact_key)
    if isinstance(entry, dict):
        raw_path = str(entry.get("path", "")).strip()
        if raw_path:
            rel = Path(raw_path.replace("\\", "/"))
            abs_path = rel if rel.is_absolute() else (run_dir / rel)
            candidate_paths.append(abs_path)

    candidate_paths.extend(
        [
            run_dir / "data" / f"{artifact_key}.parquet",
            run_dir / "data" / f"{artifact_key}.csv",
        ]
    )

    seen: set[str] = set()
    for path in candidate_paths:
        normalized = _normalize_existing_collect_data_path(path, run_dir)
        if normalized is None:
            continue
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        if normalized.suffix.lower() == ".parquet":
            return pd.read_parquet(normalized)
        if normalized.suffix.lower() == ".csv":
            return pd.read_csv(normalized)
    raise FileNotFoundError(f"{artifact_key} input not found (expected data/{artifact_key}.{{parquet|csv}})")


def _normalize_direct_plot_axis_columns(
    df: pd.DataFrame,
    *,
    axis_columns: Iterable[str],
) -> pd.DataFrame:
    plot_inputs = _get_plot_inputs_module()
    return plot_inputs.normalize_plot_axis_columns(df, axis_columns=axis_columns)


def _normalize_existing_collect_data_path(path: Path, run_dir: Path) -> Path | None:
    try:
        resolved = path.resolve()
        rel_path = resolved.relative_to(run_dir.resolve())
    except Exception:
        return None

    rel_posix = rel_path.as_posix()
    if rel_path.is_absolute() or rel_posix in {"", "."}:
        return None
    if not rel_posix.startswith("data/"):
        return None
    if resolved.suffix.lower() not in {".csv", ".parquet"}:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _sanitize_label_fragment(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def _stable_unique_label_fragment(
    value: Any,
    *,
    assigned: dict[str, str],
    used: set[str],
) -> str:
    raw = str(value).strip()
    if raw in assigned:
        return assigned[raw]

    base = _sanitize_label_fragment(raw)
    slug = base
    if slug in used:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        slug = f"{base}__{digest}"
        counter = 2
        while slug in used:
            slug = f"{base}__{digest}_{counter}"
            counter += 1

    assigned[raw] = slug
    used.add(slug)
    return slug


def _process_expert_layer_heatmaps(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    plt=None,
    missing_is_error: bool = False,
    no_job_is_skipped: bool = False,
) -> tuple[list[Path], dict[str, dict[str, str]], int]:
    config = _resolve_expert_layer_heatmap_config(cfg)
    if config is None:
        return [], {}, 0

    metric_col = config["metric"]
    source_artifact = config["source_artifact"]
    artifact_key = f"{source_artifact}__expert_layer_heatmap__{metric_col}"

    try:
        df = _read_collect_data_input(run_dir, source_artifact)
    except FileNotFoundError as exc:
        if not missing_is_error:
            return [], {}, 0
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    required = {"proj", "layer", "expert_id", metric_col}
    group_cols = ["proj"]
    if source_artifact == "quant_sim":
        required.add("scheme")
        group_cols = ["proj", "scheme"]

    try:
        _require_columns(df, required, source_artifact)
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    try:
        normalized = _normalize_direct_plot_axis_columns(df, axis_columns=("layer", "expert_id"))
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    subset_cols = group_cols + ["layer", "expert_id", metric_col]
    subset = _numeric_frame(normalized[subset_cols], ["layer", "expert_id", metric_col]).dropna(subset=subset_cols)
    if subset.empty:
        if not no_job_is_skipped:
            return [], {}, 0
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="skipped",
            )
        }, 0

    subset["layer"] = subset["layer"].astype(int)
    subset["expert_id"] = subset["expert_id"].astype(int)

    artifacts: dict[str, dict[str, str]] = {}
    written: list[Path] = []
    error_count = 0
    vmax_by_scheme: dict[str, float] = {}
    proj_slug_by_raw: dict[str, str] = {}
    used_proj_slugs: set[str] = set()
    scheme_slug_by_raw: dict[str, str] = {}
    used_scheme_slugs: set[str] = set()

    if source_artifact == "quant_sim":
        scheme_max = subset.groupby("scheme", dropna=False)[metric_col].max()
        for scheme, value in scheme_max.items():
            try:
                vmax_by_scheme[str(scheme)] = float(value)
            except Exception:
                continue

    grouping_rows = subset[group_cols].drop_duplicates().to_dict("records")
    grouping_rows.sort(key=lambda row: tuple(str(row[col]) for col in group_cols))
    for group_row in grouping_rows:
        mask = pd.Series(True, index=subset.index)
        for col in group_cols:
            mask = mask & (subset[col] == group_row[col])
        proj_rows = subset[mask]
        agg = proj_rows.groupby(["expert_id", "layer"], as_index=False, sort=True)[metric_col].median()
        if agg.empty:
            continue

        grid = (
            agg.pivot(index="expert_id", columns="layer", values=metric_col)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        if grid.empty:
            continue

        proj_slug = _stable_unique_label_fragment(
            group_row["proj"],
            assigned=proj_slug_by_raw,
            used=used_proj_slugs,
        )
        scheme = group_row.get("scheme")
        if scheme is None:
            rel_name = f"{proj_slug}__{metric_col}.png"
            artifact_key = f"{source_artifact}__expert_layer_heatmap__{proj_slug}__{metric_col}"
            title_suffix = str(group_row["proj"])
        else:
            scheme_slug = _stable_unique_label_fragment(
                scheme,
                assigned=scheme_slug_by_raw,
                used=used_scheme_slugs,
            )
            rel_name = f"{proj_slug}__{scheme_slug}__{metric_col}.png"
            artifact_key = f"{source_artifact}__expert_layer_heatmap__{proj_slug}__{scheme_slug}__{metric_col}"
            title_suffix = f"{group_row['proj']} / {scheme}"
        out_abs = run_dir / "plots" / "expert_layer_heatmaps" / rel_name
        try:
            if plt is None:
                plt = _load_pyplot()
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8, 4))
            imshow_kwargs = {
                "aspect": "auto",
                "interpolation": "nearest",
                "origin": "lower",
            }
            if source_artifact == "quant_sim" and scheme is not None:
                vmax = vmax_by_scheme.get(str(scheme))
                if vmax is not None:
                    imshow_kwargs["vmax"] = vmax
            plt.imshow(np.asarray(grid), **imshow_kwargs)
            x_vals = list(grid.columns)
            y_vals = list(grid.index)
            plt.xticks(range(len(x_vals)), [str(v) for v in x_vals], rotation=45, ha="right")
            plt.yticks(range(len(y_vals)), [str(v) for v in y_vals])
            plt.xlabel("layer")
            plt.ylabel("expert_id")
            plt.title(f"Expert-layer heatmap: {title_suffix}")
            plt.colorbar(label=metric_col)
            plt.tight_layout()
            plt.savefig(out_abs, dpi=160)
            plt.close()
            rel_path = out_abs.relative_to(run_dir).as_posix()
            artifacts[artifact_key] = _make_manifest_entry(
                source_artifact=source_artifact,
                status="written",
                path=rel_path,
            )
            written.append(out_abs)
        except Exception as exc:
            artifacts[artifact_key] = _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
            error_count += 1

    if not artifacts and no_job_is_skipped and error_count == 0:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="skipped",
            )
        }, 0

    return written, artifacts, error_count


def _has_expert_layer_heatmap_jobs(*, run_dir: Path, cfg: dict[str, Any]) -> bool:
    config = _resolve_expert_layer_heatmap_config(cfg)
    if config is None:
        return False

    metric_col = config["metric"]
    source_artifact = config["source_artifact"]
    try:
        df = _read_collect_data_input(run_dir, source_artifact)
    except FileNotFoundError:
        return False
    except Exception:
        # Treat unreadable direct inputs as actionable so build_plots emits
        # audited error entries instead of returning early with "no jobs".
        return True

    required = {"proj", "layer", "expert_id", metric_col}
    group_cols = ["proj"]
    if source_artifact == "quant_sim":
        required.add("scheme")
        group_cols = ["proj", "scheme"]
    try:
        _require_columns(df, required, source_artifact)
    except Exception:
        return True

    try:
        normalized = _normalize_direct_plot_axis_columns(df, axis_columns=("layer", "expert_id"))
    except Exception:
        return True

    subset_cols = group_cols + ["layer", "expert_id", metric_col]
    subset = _numeric_frame(normalized[subset_cols], ["layer", "expert_id", metric_col]).dropna(subset=subset_cols)
    return not subset.empty


def _process_plot_for_spec(
    *,
    run_dir: Path,
    spec: dict[str, Any],
    source_key: str,
    frame: pd.DataFrame,
    plt,
) -> tuple[Path | None, dict[str, str]]:
    out_abs = run_dir / spec["output_relpath"]
    plot_fn: Callable[..., bool] = spec["plot_fn"]
    try:
        wrote = plot_fn(frame, out_abs, plt, source_key)
    except Exception as exc:
        return None, _make_manifest_entry(
            source_artifact=source_key,
            status="error",
            error=_error_text(exc),
        )

    if not wrote:
        return None, _make_manifest_entry(source_artifact=source_key, status="skipped")

    rel_path = out_abs.relative_to(run_dir).as_posix()
    return out_abs, _make_manifest_entry(
        source_artifact=source_key,
        status="written",
        path=rel_path,
    )


def _numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _require_columns(df: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{artifact_name} is missing required columns for plotting: {missing_text}")


def _optional_columns_present(df: pd.DataFrame, spec: dict[str, Any]) -> bool:
    optional_cols = spec.get("optional_if_columns_present")
    if not optional_cols:
        return True
    return set(optional_cols).issubset(set(df.columns))


def _is_optional_spec(spec: dict[str, Any]) -> bool:
    return bool(spec.get("optional_if_columns_present"))


def _plot_a_weight_global(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "A_weight_global_summary",
) -> bool:
    metric_col = "mean_abs__median"
    _require_columns(df, {"proj", metric_col}, artifact_name)
    subset = _numeric_frame(df[["proj", metric_col]], [metric_col]).dropna(subset=["proj", metric_col])
    if subset.empty:
        return False
    agg = subset.groupby("proj", as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(agg["proj"].astype(str), agg[metric_col])
    plt.xlabel("proj")
    plt.ylabel(metric_col)
    plt.title("A weight global summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_a_weight_global_tail_proxy(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "A_weight_global_summary",
) -> bool:
    metric_col = "outlier_p999_over_median__median"
    _require_columns(df, {"proj", metric_col}, artifact_name)
    subset = _numeric_frame(df[["proj", metric_col]], [metric_col]).dropna(subset=["proj", metric_col])
    if subset.empty:
        return False
    agg = subset.groupby("proj", as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(agg["proj"].astype(str), agg[metric_col])
    plt.xlabel("proj")
    plt.ylabel(metric_col)
    plt.title("A weight global tail proxy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_a_weight_layer(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "A_weight_layer_summary",
) -> bool:
    metric_col = "mean_abs__median"
    required = {"layer", "proj", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["layer", "proj", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", metric_col]
    )
    if subset.empty:
        return False
    subset["layer"] = subset["layer"].astype(int)
    agg = subset.groupby(["proj", "layer"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    proj_values = list(agg["proj"].drop_duplicates())
    proj_values.sort(key=lambda value: str(value))
    for proj in proj_values:
        proj_rows = agg[agg["proj"] == proj].sort_values("layer", kind="mergesort")
        plt.plot(
            proj_rows["layer"],
            proj_rows[metric_col],
            marker="o",
            label=str(proj),
        )
    plt.xlabel("layer")
    plt.ylabel(metric_col)
    plt.title("A weight layer summary")
    if agg["proj"].nunique() > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_a_weight_layer_tail_proxy(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "A_weight_layer_summary",
) -> bool:
    metric_col = "outlier_p999_over_median__median"
    required = {"layer", "proj", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["layer", "proj", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", metric_col]
    )
    if subset.empty:
        return False
    subset["layer"] = subset["layer"].astype(int)
    agg = subset.groupby(["proj", "layer"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    proj_values = list(agg["proj"].drop_duplicates())
    proj_values.sort(key=lambda value: str(value))
    for proj in proj_values:
        proj_rows = agg[agg["proj"] == proj].sort_values("layer", kind="mergesort")
        plt.plot(
            proj_rows["layer"],
            proj_rows[metric_col],
            marker="o",
            label=str(proj),
        )
    plt.xlabel("layer")
    plt.ylabel(metric_col)
    plt.title("A weight layer tail proxy")
    if agg["proj"].nunique() > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_b_quant_global(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "B_quant_global_summary",
) -> bool:
    metric_col = "w_rel_fro__median"
    required = {"proj", "scheme", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["proj", "scheme", metric_col]], [metric_col]).dropna(
        subset=["proj", "scheme", metric_col]
    )
    if subset.empty:
        return False
    agg = subset.groupby(["proj", "scheme"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False
    x_labels = (agg["proj"].astype(str) + " / " + agg["scheme"].astype(str)).tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(x_labels, agg[metric_col])
    plt.xlabel("proj / scheme")
    plt.ylabel(metric_col)
    plt.title("B quant global summary by proj and scheme")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_b_quant_layer(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "B_quant_layer_summary",
) -> bool:
    metric_col = "w_rel_fro__median"
    required = {"layer", "proj", "scheme", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["layer", "proj", "scheme", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", "scheme", metric_col]
    )
    if subset.empty:
        return False
    subset["layer"] = subset["layer"].astype(int)
    agg = subset.groupby(["proj", "scheme", "layer"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    pairs = agg[["proj", "scheme"]].drop_duplicates().to_dict("records")
    pairs.sort(key=lambda row: (str(row["proj"]), str(row["scheme"])))
    for pair in pairs:
        proj = pair["proj"]
        scheme = pair["scheme"]
        rows = agg[(agg["proj"] == proj) & (agg["scheme"] == scheme)].sort_values("layer", kind="mergesort")
        plt.plot(rows["layer"], rows[metric_col], marker="o", label=f"{proj} / {scheme}")
    plt.xlabel("layer")
    plt.ylabel(metric_col)
    plt.title("B quant layer summary by proj and scheme")
    if len(pairs) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_b_quant_layer_max(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "B_quant_layer_summary",
) -> bool:
    metric_col = "w_rel_max__median"
    required = {"layer", "proj", "scheme", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["layer", "proj", "scheme", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", "scheme", metric_col]
    )
    if subset.empty:
        return False
    subset["layer"] = subset["layer"].astype(int)
    agg = subset.groupby(["proj", "scheme", "layer"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    pairs = agg[["proj", "scheme"]].drop_duplicates().to_dict("records")
    pairs.sort(key=lambda row: (str(row["proj"]), str(row["scheme"])))
    for pair in pairs:
        proj = pair["proj"]
        scheme = pair["scheme"]
        rows = agg[(agg["proj"] == proj) & (agg["scheme"] == scheme)].sort_values("layer", kind="mergesort")
        plt.plot(rows["layer"], rows[metric_col], marker="o", label=f"{proj} / {scheme}")
    plt.xlabel("layer")
    plt.ylabel(metric_col)
    plt.title("B quant layer max summary by proj and scheme")
    if len(pairs) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_quant_sim_layer(
    df: pd.DataFrame,
    out_path: Path,
    plt,
    artifact_name: str = "quant_sim",
    metric_col: str = "w_rel_fro",
) -> bool:
    required = {"layer", "proj", "scheme", metric_col}
    _require_columns(df, required, artifact_name)
    subset = _numeric_frame(df[["layer", "proj", "scheme", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", "scheme", metric_col]
    )
    if subset.empty:
        return False
    subset["layer"] = subset["layer"].astype(int)
    agg = subset.groupby(["proj", "scheme", "layer"], as_index=False, sort=True)[metric_col].median()
    if agg.empty:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    pairs = agg[["proj", "scheme"]].drop_duplicates().to_dict("records")
    pairs.sort(key=lambda row: (str(row["proj"]), str(row["scheme"])))
    for pair in pairs:
        proj = pair["proj"]
        scheme = pair["scheme"]
        rows = agg[(agg["proj"] == proj) & (agg["scheme"] == scheme)].sort_values("layer", kind="mergesort")
        plt.plot(rows["layer"], rows[metric_col], marker="o", label=f"{proj} / {scheme}")
    plt.xlabel("layer")
    plt.ylabel(metric_col)
    plt.title("Quant sim by proj and scheme")
    if len(pairs) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _process_quant_error_layer_lines(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    plt=None,
    missing_is_error: bool = False,
) -> tuple[list[Path], dict[str, dict[str, str]], int]:
    config = _resolve_quant_error_layer_lines_config(cfg)
    if config is None:
        return [], {}, 0

    metric_col = config["metric"]
    source_artifact = config["source_artifact"]
    artifact_key = f"{source_artifact}__{metric_col}_by_proj_and_scheme"
    try:
        df = _read_collect_data_input(run_dir, source_artifact)
    except FileNotFoundError as exc:
        if not missing_is_error:
            return [], {}, 0
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    required = {"layer", "proj", "scheme", metric_col}
    try:
        _require_columns(df, required, source_artifact)
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    try:
        normalized = _normalize_direct_plot_axis_columns(df, axis_columns=("layer",))
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    subset = _numeric_frame(normalized[["layer", "proj", "scheme", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", "scheme", metric_col]
    )
    if subset.empty:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="skipped",
            )
        }, 0

    out_abs = run_dir / "plots" / "layer" / f"{source_artifact}__{metric_col}_by_proj_and_scheme.png"
    try:
        if plt is None:
            plt = _load_pyplot()
        wrote = _plot_quant_sim_layer(normalized, out_abs, plt, source_artifact, metric_col)
    except Exception as exc:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="error",
                error=_error_text(exc),
            )
        }, 1

    if not wrote:
        return [], {
            artifact_key: _make_manifest_entry(
                source_artifact=source_artifact,
                status="skipped",
            )
        }, 0

    rel_path = out_abs.relative_to(run_dir).as_posix()
    return [out_abs], {
        artifact_key: _make_manifest_entry(
            source_artifact=source_artifact,
            status="written",
            path=rel_path,
        )
    }, 0


def _has_quant_error_layer_line_jobs(*, run_dir: Path, cfg: dict[str, Any]) -> bool:
    config = _resolve_quant_error_layer_lines_config(cfg)
    if config is None:
        return False

    metric_col = config["metric"]
    source_artifact = config["source_artifact"]
    try:
        df = _read_collect_data_input(run_dir, source_artifact)
    except FileNotFoundError:
        return False
    except Exception:
        # Treat unreadable direct inputs as actionable so build_plots emits
        # audited error entries instead of returning early with "no jobs".
        return True

    required = {"layer", "proj", "scheme", metric_col}
    try:
        _require_columns(df, required, source_artifact)
    except Exception:
        return True

    try:
        normalized = _normalize_direct_plot_axis_columns(df, axis_columns=("layer",))
    except Exception:
        return True

    subset = _numeric_frame(normalized[["layer", "proj", "scheme", metric_col]], ["layer", metric_col]).dropna(
        subset=["layer", "proj", "scheme", metric_col]
    )
    return not subset.empty


def _process_direct_data_plots(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    explicit_only: bool,
) -> tuple[list[Path], dict[str, dict[str, str]], int]:
    written: list[Path] = []
    artifacts: dict[str, dict[str, str]] = {}
    error_count = 0

    heatmaps_explicit = _is_direct_plot_option_explicitly_enabled(cfg, "expert_layer_heatmaps")
    quant_lines_explicit = _is_direct_plot_option_explicitly_enabled(cfg, "quant_error_layer_lines")

    if not explicit_only or heatmaps_explicit:
        hm_written, hm_artifacts, hm_error_count = _process_expert_layer_heatmaps(
            run_dir=run_dir,
            cfg=cfg,
            missing_is_error=heatmaps_explicit,
            no_job_is_skipped=heatmaps_explicit,
        )
        written.extend(hm_written)
        artifacts.update(hm_artifacts)
        error_count += hm_error_count

    if not explicit_only or quant_lines_explicit:
        qline_written, qline_artifacts, qline_error_count = _process_quant_error_layer_lines(
            run_dir=run_dir,
            cfg=cfg,
            missing_is_error=quant_lines_explicit,
        )
        written.extend(qline_written)
        artifacts.update(qline_artifacts)
        error_count += qline_error_count

    return written, artifacts, error_count


_PLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_artifact": "A_weight_global_summary",
        "plot_artifact": "A_weight_global_summary__mean_abs__median",
        "output_relpath": "plots/global/A_weight_global_summary__mean_abs__median.png",
        "plot_fn": _plot_a_weight_global,
    },
    {
        "source_artifact": "A_weight_global_summary",
        "plot_artifact": "A_weight_global_summary__outlier_p999_over_median__median",
        "output_relpath": "plots/global/A_weight_global_summary__outlier_p999_over_median__median.png",
        "plot_fn": _plot_a_weight_global_tail_proxy,
        "optional_if_columns_present": ["outlier_p999_over_median__median"],
    },
    {
        "source_artifact": "A_weight_layer_summary",
        "plot_artifact": "A_weight_layer_summary__mean_abs__median",
        "output_relpath": "plots/layer/A_weight_layer_summary__mean_abs__median.png",
        "plot_fn": _plot_a_weight_layer,
    },
    {
        "source_artifact": "A_weight_layer_summary",
        "plot_artifact": "A_weight_layer_summary__outlier_p999_over_median__median",
        "output_relpath": "plots/layer/A_weight_layer_summary__outlier_p999_over_median__median.png",
        "plot_fn": _plot_a_weight_layer_tail_proxy,
        "optional_if_columns_present": ["outlier_p999_over_median__median"],
    },
    {
        "source_artifact": "B_quant_global_summary",
        "plot_artifact": "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
        "output_relpath": "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
        "plot_fn": _plot_b_quant_global,
    },
    {
        "source_artifact": "B_quant_layer_summary",
        "plot_artifact": "B_quant_layer_summary__w_rel_fro__median_by_proj_and_scheme",
        "output_relpath": "plots/layer/B_quant_layer_summary__w_rel_fro__median_by_proj_and_scheme.png",
        "plot_fn": _plot_b_quant_layer,
    },
    {
        "source_artifact": "B_quant_layer_summary",
        "plot_artifact": "B_quant_layer_summary__w_rel_max__median_by_proj_and_scheme",
        "output_relpath": "plots/layer/B_quant_layer_summary__w_rel_max__median_by_proj_and_scheme.png",
        "plot_fn": _plot_b_quant_layer_max,
        "optional_if_columns_present": ["w_rel_max__median"],
    },
)
_PLOT_SPECS_BY_SOURCE: dict[str, list[dict[str, Any]]] = {}
for _spec in _PLOT_SPECS:
    _PLOT_SPECS_BY_SOURCE.setdefault(_spec["source_artifact"], []).append(_spec)
_DEFAULT_SOURCE_ARTIFACT_KEYS = [
    "A_weight_global_summary",
    "A_weight_layer_summary",
    "B_quant_global_summary",
]


def build_plots(run_dir: str | Path) -> list[Path]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    cfg = _load_config(run_dir_path)
    requested_keys, explicit_selection = _resolve_requested_artifact_keys(cfg)
    _validate_direct_plot_configs(cfg)
    has_explicit_direct_plot_requests = _has_explicit_direct_plot_requests(cfg)

    # Explicit empty selection is an intentional "select nothing" request:
    # skip discovery and emit an invocation-scoped empty audit manifest unless
    # direct-data plots were explicitly requested alongside the empty table set.
    if explicit_selection and not requested_keys and not has_explicit_direct_plot_requests:
        _write_plots_manifest(run_dir_path, requested_keys, {})
        return []

    plot_inputs = _get_plot_inputs_module()

    # First attempt a single loader call for the full selected key list so loader-boundary
    # behavior (selection, dedupe) remains directly testable.
    bulk_loaded: dict[str, pd.DataFrame] | None = None
    bulk_error: Exception | None = None
    if requested_keys:
        try:
            bulk_loaded = plot_inputs.load_plot_tables(run_dir_path, artifact_keys=requested_keys)
        except Exception as exc:
            bulk_error = exc
    else:
        bulk_loaded = {}

    if not explicit_selection:
        if bulk_error is not None:
            raise bulk_error
        assert bulk_loaded is not None
        artifacts: dict[str, dict[str, str]] = {}
        written: list[Path] = []
        error_count = 0
        plt = None
        for source_key in requested_keys:
            frame = bulk_loaded.get(source_key)
            if frame is None:
                continue
            for spec in _PLOT_SPECS_BY_SOURCE[source_key]:
                if not _optional_columns_present(frame, spec):
                    continue
                if plt is None:
                    plt = _load_pyplot()
                out_abs, entry = _process_plot_for_spec(
                    run_dir=run_dir_path,
                    spec=spec,
                    source_key=source_key,
                    frame=frame,
                    plt=plt,
                )
                artifacts[spec["plot_artifact"]] = entry
                if out_abs is not None:
                    written.append(out_abs)
                elif entry["status"] == "error":
                    error_count += 1

        direct_written, direct_artifacts, direct_error_count = _process_direct_data_plots(
            run_dir=run_dir_path,
            cfg=cfg,
            explicit_only=False,
        )
        written.extend(direct_written)
        artifacts.update(direct_artifacts)
        error_count += direct_error_count

        if not artifacts:
            return []

        _write_plots_manifest(run_dir_path, requested_keys, artifacts)
        if error_count > 0:
            detail_texts = [
                entry.get("error", "")
                for entry in artifacts.values()
                if entry.get("status") == "error" and str(entry.get("error", "")).strip()
            ]
            detail_suffix = f": {'; '.join(detail_texts)}" if detail_texts else ""
            raise RuntimeError(
                f"{error_count} plot artifact(s) failed{detail_suffix}; see logs/plots_write_manifest.json"
            )
        return written

    # Explicit selection mode is best-effort and always audited once validation passes.
    written: list[Path] = []
    artifacts: dict[str, dict[str, str]] = {}
    error_count = 0
    errored_sources: list[str] = []
    plt = None
    pyplot_error: Exception | None = None

    if bulk_error is None:
        assert bulk_loaded is not None
        per_key_frames: dict[str, pd.DataFrame | None] = {key: bulk_loaded.get(key) for key in requested_keys}
    else:
        per_key_frames = {}
        for key in requested_keys:
            try:
                one = plot_inputs.load_plot_tables(run_dir_path, artifact_keys=[key])
            except Exception as exc:
                for spec in _PLOT_SPECS_BY_SOURCE[key]:
                    if _is_optional_spec(spec):
                        continue
                    artifacts[spec["plot_artifact"]] = _make_manifest_entry(
                        source_artifact=key,
                        status="error",
                        error=_error_text(exc),
                    )
                    error_count += 1
                errored_sources.append(key)
                per_key_frames[key] = None
                continue
            per_key_frames[key] = one.get(key)

    for source_key in requested_keys:
        frame = per_key_frames.get(source_key)
        if frame is None:
            for spec in _PLOT_SPECS_BY_SOURCE[source_key]:
                if _is_optional_spec(spec):
                    continue
                artifacts[spec["plot_artifact"]] = _make_manifest_entry(
                    source_artifact=source_key,
                    status="error",
                    error=f"Selected table artifact not found for plotting: {source_key}",
                )
                error_count += 1
            errored_sources.append(source_key)
            continue

        for spec in _PLOT_SPECS_BY_SOURCE[source_key]:
            plot_artifact = spec["plot_artifact"]
            if plot_artifact in artifacts:
                continue
            if not _optional_columns_present(frame, spec):
                continue

            if plt is None:
                if pyplot_error is None:
                    try:
                        plt = _load_pyplot()
                    except Exception as exc:
                        pyplot_error = exc
                if pyplot_error is not None:
                    artifacts[plot_artifact] = _make_manifest_entry(
                        source_artifact=source_key,
                        status="error",
                        error=_error_text(pyplot_error),
                    )
                    error_count += 1
                    errored_sources.append(source_key)
                    continue

            out_abs, entry = _process_plot_for_spec(
                run_dir=run_dir_path,
                spec=spec,
                source_key=source_key,
                frame=frame,
                plt=plt,
            )
            artifacts[plot_artifact] = entry
            if out_abs is not None:
                written.append(out_abs)
            elif entry["status"] == "error":
                error_count += 1
                errored_sources.append(source_key)

    direct_written, direct_artifacts, direct_error_count = _process_direct_data_plots(
        run_dir=run_dir_path,
        cfg=cfg,
        explicit_only=True,
    )
    written.extend(direct_written)
    artifacts.update(direct_artifacts)
    error_count += direct_error_count
    if direct_error_count > 0:
        errored_sources.extend(
            entry.get("source_artifact", "")
            for entry in direct_artifacts.values()
            if entry.get("status") == "error"
        )

    _write_plots_manifest(run_dir_path, requested_keys, artifacts)
    if error_count > 0:
        names = ", ".join(errored_sources)
        detail_texts = [
            entry.get("error", "")
            for entry in artifacts.values()
            if entry.get("status") == "error" and str(entry.get("error", "")).strip()
        ]
        detail_suffix = f": {'; '.join(detail_texts)}" if detail_texts else ""
        raise RuntimeError(
            f"{error_count} plot artifact(s) failed ({names}){detail_suffix}; "
            "see logs/plots_write_manifest.json"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build plots from table artifacts.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing tables/ and logs/")
    args = parser.parse_args()

    try:
        written = build_plots(args.run_dir)
    except Exception as exc:
        print(f"[build_plots] error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if written:
        print(f"[build_plots] wrote {len(written)} plot(s)")
    else:
        print("[build_plots] no compatible tables found for plotting")


if __name__ == "__main__":
    main()
