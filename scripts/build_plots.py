#!/usr/bin/env python3
"""
Build a small deterministic set of plots from table artifacts.

Usage:
  python scripts/build_plots.py --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

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


def _process_plot_for_key(
    *,
    run_dir: Path,
    source_key: str,
    frame: pd.DataFrame,
    plt,
) -> tuple[Path | None, dict[str, str]]:
    spec = _PLOT_SPECS_BY_SOURCE[source_key]
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


_PLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_artifact": "A_weight_global_summary",
        "plot_artifact": "A_weight_global_summary__mean_abs__median",
        "output_relpath": "plots/global/A_weight_global_summary__mean_abs__median.png",
        "plot_fn": _plot_a_weight_global,
    },
    {
        "source_artifact": "A_weight_layer_summary",
        "plot_artifact": "A_weight_layer_summary__mean_abs__median",
        "output_relpath": "plots/layer/A_weight_layer_summary__mean_abs__median.png",
        "plot_fn": _plot_a_weight_layer,
    },
    {
        "source_artifact": "B_quant_global_summary",
        "plot_artifact": "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
        "output_relpath": "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
        "plot_fn": _plot_b_quant_global,
    },
)
_PLOT_SPECS_BY_SOURCE = {spec["source_artifact"]: spec for spec in _PLOT_SPECS}
_DEFAULT_SOURCE_ARTIFACT_KEYS = [spec["source_artifact"] for spec in _PLOT_SPECS]


def build_plots(run_dir: str | Path) -> list[Path]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    cfg = _load_config(run_dir_path)
    requested_keys, explicit_selection = _resolve_requested_artifact_keys(cfg)

    # Explicit empty selection is an intentional "select nothing" request:
    # skip discovery and emit an invocation-scoped empty audit manifest.
    if explicit_selection and not requested_keys:
        _write_plots_manifest(run_dir_path, requested_keys, {})
        return []

    plot_inputs = _get_plot_inputs_module()

    # First attempt a single loader call for the full selected key list so loader-boundary
    # behavior (selection, dedupe) remains directly testable.
    bulk_loaded: dict[str, pd.DataFrame] | None = None
    bulk_error: Exception | None = None
    try:
        bulk_loaded = plot_inputs.load_plot_tables(run_dir_path, artifact_keys=requested_keys)
    except Exception as exc:
        bulk_error = exc

    if not explicit_selection:
        if bulk_error is not None:
            raise bulk_error
        assert bulk_loaded is not None
        if not bulk_loaded:
            return []

        plt = _load_pyplot()
        artifacts: dict[str, dict[str, str]] = {}
        written: list[Path] = []
        error_count = 0
        for source_key in requested_keys:
            frame = bulk_loaded.get(source_key)
            if frame is None:
                continue
            spec = _PLOT_SPECS_BY_SOURCE[source_key]
            out_abs, entry = _process_plot_for_key(
                run_dir=run_dir_path,
                source_key=source_key,
                frame=frame,
                plt=plt,
            )
            artifacts[spec["plot_artifact"]] = entry
            if out_abs is not None:
                written.append(out_abs)
            elif entry["status"] == "error":
                error_count += 1

        if not artifacts:
            return []

        _write_plots_manifest(run_dir_path, requested_keys, artifacts)
        if error_count > 0:
            raise RuntimeError(f"{error_count} plot artifact(s) failed; see logs/plots_write_manifest.json")
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
                spec = _PLOT_SPECS_BY_SOURCE[key]
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
        spec = _PLOT_SPECS_BY_SOURCE[source_key]
        plot_artifact = spec["plot_artifact"]

        if plot_artifact in artifacts:
            continue

        frame = per_key_frames.get(source_key)
        if frame is None:
            artifacts[plot_artifact] = _make_manifest_entry(
                source_artifact=source_key,
                status="error",
                error=f"Selected table artifact not found for plotting: {source_key}",
            )
            error_count += 1
            errored_sources.append(source_key)
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

        out_abs, entry = _process_plot_for_key(
            run_dir=run_dir_path,
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
