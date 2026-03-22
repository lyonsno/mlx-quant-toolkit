#!/usr/bin/env python3
"""
Build summary tables from data/matrix_stats and data/quant_sim.

Usage:
  python build_tables.py --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


MATRIX_STATS_EMPTY_COLUMNS = [
    "layer",
    "proj",
    "mean",
    "std",
    "mean_abs",
    "rms",
    "max_abs",
    "p50_abs",
    "p99_abs",
    "p999_abs",
    "outlier_max_over_mean",
    "outlier_p99_over_median",
    "outlier_p999_over_median",
]

MATRIX_STATS_AXIS_COLUMNS = [
    "layer",
    "proj",
]

MATRIX_STATS_METRIC_COLUMNS = [
    "mean",
    "std",
    "mean_abs",
    "rms",
    "max_abs",
    "p50_abs",
    "p99_abs",
    "p999_abs",
    "outlier_max_over_mean",
    "outlier_p99_over_median",
    "outlier_p999_over_median",
]

QUANT_SIM_EMPTY_COLUMNS = [
    "derived_tensor",
    "layer",
    "block4",
    "proj",
    "expert_id",
    "rows",
    "cols",
    "scheme",
    "w_rel_fro",
    "w_rel_max",
    "scale_mean",
    "scale_max",
    "bias_mean",
    "bias_max",
]

QUANT_SIM_AXIS_COLUMNS = [
    "derived_tensor",
    "layer",
    "proj",
    "expert_id",
    "rows",
    "cols",
    "scheme",
]

QUANT_METRIC_COLUMNS = [
    "w_rel_fro",
    "w_rel_max",
    "scale_mean",
    "scale_max",
    "bias_mean",
    "bias_max",
]

_TABLE_INPUT_SUFFIXES = {".csv", ".parquet"}


def _safe_read_json_dict(path: Path) -> Dict[str, Any]:
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


def _read_df(path: Path, empty_columns: List[str] | None = None) -> pd.DataFrame:
    def _read_csv(csv_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            if empty_columns is None:
                return pd.DataFrame()
            return pd.DataFrame(columns=list(empty_columns))

    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return _read_csv(path)
    # try fallback extensions
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return _read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(path)


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _path_for_manifest(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return str(path)


def _resolve_collect_artifact_input_path(
    run_dir: Path,
    manifest_artifacts: Dict[str, Any],
    artifact_key: str,
) -> Path | None:
    entry = manifest_artifacts.get(artifact_key)
    if not isinstance(entry, dict):
        return None

    raw_path = entry.get("path")
    if raw_path is None:
        return None
    path_text = str(raw_path).strip()
    if not path_text:
        return None

    # Accept manifests created on other platforms by normalizing separators.
    # Example: "data\\matrix_stats.csv" (Windows) -> "data/matrix_stats.csv".
    path_text = path_text.replace("\\", "/")

    candidate = Path(path_text)
    candidate_abs = candidate if candidate.is_absolute() else (run_dir / candidate)

    # Reject manifest paths outside run_dir for defensive path hygiene.
    try:
        rel = candidate_abs.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return None

    # build_tables inputs should come from data/ artifacts emitted by collect_data.
    if not rel.startswith("data/"):
        return None
    if candidate_abs.suffix.lower() not in _TABLE_INPUT_SUFFIXES:
        return None
    if not candidate_abs.is_file():
        return None

    return candidate_abs


def _normalize_write_meta(meta: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    raw_path = meta.get("path")
    if isinstance(raw_path, Path):
        rel_path = _path_for_manifest(raw_path, run_dir)
    else:
        rel_path = _path_for_manifest(Path(str(raw_path)), run_dir)
    return {
        "path": rel_path,
        "format": str(meta.get("format", "")),
        "fallback": bool(meta.get("fallback", False)),
        "error": str(meta.get("error", "")),
        "rows": int(meta.get("rows", 0)),
    }


def _write_df(df: pd.DataFrame, path: Path, fmt: str, compression: str | None) -> Dict[str, Any]:
    # CONTRACT SURFACE: tables/*.parquet|csv Parquet→CSV fallback + logs/tables_write_manifest.json
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs.
    # Tests: rg 'tables_write_manifest|build_tables' tests/
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(len(df))
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False, compression=compression)
            return {
                "path": path,
                "format": "parquet",
                "fallback": False,
                "error": "",
                "rows": rows,
            }
        except Exception as exc:
            print(f"[warn] parquet write failed ({exc}); falling back to CSV for {path}")
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return {
                "path": csv_path,
                "format": "csv",
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
                "rows": rows,
            }
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return {
        "path": csv_path,
        "format": "csv",
        "fallback": False,
        "error": "",
        "rows": rows,
    }


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "analysis_config.json"
    return json.loads(cfg_path.read_text())


def _quantile_func(q: float, label: str):
    def _fn(s: pd.Series):
        return s.quantile(q)
    _fn.__name__ = label
    return _fn


def _ensure_columns(df: pd.DataFrame, required_columns: List[str], *, fill_value: Any = pd.NA) -> pd.DataFrame:
    """Return a frame that contains at least the required columns.

    For numeric metric columns that are missing, force float dtype when
    adding them to avoid object-dtype promotion during aggregations.
    """
    out = df.copy()
    for col in required_columns:
        if col not in out.columns:
            # Determine if this is likely a numeric metric column by checking
            # against known numeric column sets in this module.
            # This is a heuristic: if the column name appears in either of the
            # metric column lists, enforce float dtype to prevent TypeError
            # during aggregations like mean/std on all-NA values.
            if col in (MATRIX_STATS_METRIC_COLUMNS + QUANT_METRIC_COLUMNS):
                out[col] = pd.Series(fill_value, index=out.index, dtype=float)
            else:
                out[col] = fill_value
    return out


def _agg_with_funcs(df: pd.DataFrame, group_cols: List[str], value_cols: List[str], agg_funcs: List[Any]) -> pd.DataFrame:
    if not value_cols:
        # Preserve group-axis rows when metric columns are unavailable.
        # This keeps plotting keys stable without emitting vacuous metric columns.
        base = df.copy()
        for col in group_cols:
            if col not in base.columns:
                base[col] = pd.NA
        return (
            base[group_cols]
            .drop_duplicates()
            .sort_values(group_cols, kind="mergesort")
            .reset_index(drop=True)
        )

    grouped = df.groupby(group_cols, dropna=False)[value_cols].agg(agg_funcs).reset_index()
    # Flatten MultiIndex columns produced by multi-agg.
    grouped.columns = [
        c if not isinstance(c, tuple) else (c[0] if (len(c) > 1 and (c[1] == "" or c[1] is None)) else f"{c[0]}__{c[1]}")
        for c in grouped.columns
    ]
    return grouped


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate matrix_stats and quant_sim into summary tables."
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing data/ artifacts from collect_data",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_config(run_dir)
    fmt = cfg.get("output", {}).get("format", "parquet")
    compression = cfg.get("output", {}).get("compression", None)
    write_manifest_artifacts: Dict[str, Dict[str, Any]] = {}

    collect_manifest = _safe_read_json_dict(run_dir / "logs" / "write_manifest.json")
    collect_manifest_artifacts = collect_manifest.get("artifacts", {})
    if not isinstance(collect_manifest_artifacts, dict):
        collect_manifest_artifacts = {}

    matrix_stats_input = _resolve_collect_artifact_input_path(
        run_dir,
        collect_manifest_artifacts,
        "matrix_stats",
    )
    quant_sim_input = _resolve_collect_artifact_input_path(
        run_dir,
        collect_manifest_artifacts,
        "quant_sim",
    )

    if matrix_stats_input is None:
        matrix_stats_input = run_dir / "data" / "matrix_stats.parquet"
    if quant_sim_input is None:
        quant_sim_input = run_dir / "data" / "quant_sim.parquet"

    ms = _read_df(
        matrix_stats_input,
        empty_columns=MATRIX_STATS_EMPTY_COLUMNS,
    )
    qs = _read_df(
        quant_sim_input,
        empty_columns=QUANT_SIM_EMPTY_COLUMNS,
    )
    ms = _ensure_columns(ms, MATRIX_STATS_AXIS_COLUMNS)
    # A-table metric columns must stay numeric-friendly when synthesized; using
    # pd.NA in object dtype can raise during std/mean aggregations on non-empty groups.
    ms = _ensure_columns(ms, MATRIX_STATS_METRIC_COLUMNS, fill_value=float("nan"))
    qs = _ensure_columns(qs, QUANT_SIM_AXIS_COLUMNS)

    # ensure block4 exists
    if "block4" not in ms.columns:
        ms["block4"] = ms["layer"].floordiv(4)
    if "block4" not in qs.columns:
        qs["block4"] = qs["layer"].floordiv(4)

    # -------- A: weight stats summaries --------
    stat_cols = [
        "mean", "std", "mean_abs", "rms", "max_abs",
        "p50_abs", "p99_abs", "p999_abs",
        "outlier_max_over_mean", "outlier_p99_over_median", "outlier_p999_over_median",
    ]
    # include any gXX columns that exist
    stat_cols += [c for c in ms.columns if c.startswith("g") and ("_outlier" in c)]

    p90 = _quantile_func(0.90, "p90")
    p99 = _quantile_func(0.99, "p99")
    p01 = _quantile_func(0.01, "p01")

    # per layer/proj
    A_layer = _agg_with_funcs(ms, ["layer", "proj"], stat_cols, ["median", "mean", "std", p90, p99])
    # CONTRACT SURFACE: tables/A_weight_layer_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'A_weight_layer_summary' tests/
    write_manifest_artifacts["A_weight_layer_summary"] = _normalize_write_meta(
        _write_df(A_layer, run_dir / "tables" / "A_weight_layer_summary.parquet", fmt, compression),
        run_dir,
    )

    # per block4/proj
    A_block4 = _agg_with_funcs(ms, ["block4", "proj"], stat_cols, ["median", "mean", "std", p90, p99])
    # CONTRACT SURFACE: tables/A_weight_block4_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'A_weight_block4_summary' tests/
    write_manifest_artifacts["A_weight_block4_summary"] = _normalize_write_meta(
        _write_df(A_block4, run_dir / "tables" / "A_weight_block4_summary.parquet", fmt, compression),
        run_dir,
    )

    # global/proj
    A_global = _agg_with_funcs(ms, ["proj"], stat_cols, ["min", p01, "median", p99, "max"])
    # CONTRACT SURFACE: tables/A_weight_global_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'A_weight_global_summary' tests/
    write_manifest_artifacts["A_weight_global_summary"] = _normalize_write_meta(
        _write_df(A_global, run_dir / "tables" / "A_weight_global_summary.parquet", fmt, compression),
        run_dir,
    )

    # -------- B: quant sim summaries --------
    qcols = QUANT_METRIC_COLUMNS
    qcols = [c for c in qcols if c in qs.columns]

    B_layer = _agg_with_funcs(qs, ["layer", "proj", "scheme"], qcols, ["median", "mean", p90, p99])
    # CONTRACT SURFACE: tables/B_quant_layer_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'B_quant_layer_summary' tests/
    write_manifest_artifacts["B_quant_layer_summary"] = _normalize_write_meta(
        _write_df(B_layer, run_dir / "tables" / "B_quant_layer_summary.parquet", fmt, compression),
        run_dir,
    )

    B_block4 = _agg_with_funcs(qs, ["block4", "proj", "scheme"], qcols, ["median", "mean", p90, p99])
    # CONTRACT SURFACE: tables/B_quant_block4_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'B_quant_block4_summary' tests/
    write_manifest_artifacts["B_quant_block4_summary"] = _normalize_write_meta(
        _write_df(B_block4, run_dir / "tables" / "B_quant_block4_summary.parquet", fmt, compression),
        run_dir,
    )

    B_global = _agg_with_funcs(qs, ["proj", "scheme"], qcols, ["min", p01, "median", p99, "max"])
    # CONTRACT SURFACE: tables/B_quant_global_summary.{parquet|csv}
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'B_quant_global_summary' tests/
    write_manifest_artifacts["B_quant_global_summary"] = _normalize_write_meta(
        _write_df(B_global, run_dir / "tables" / "B_quant_global_summary.parquet", fmt, compression),
        run_dir,
    )

    # -------- deltas (scheme A - scheme B) --------
    delta_pairs = cfg.get("delta_pairs", [])
    if delta_pairs:
        base_index = ["derived_tensor", "layer", "block4", "proj", "expert_id", "rows", "cols"]
        for col in base_index:
            if col not in qs.columns:
                qs[col] = pd.NA
        delta_metric_values = [c for c in ["w_rel_fro", "w_rel_max"] if c in qs.columns]
        if delta_metric_values:
            pivot = qs.pivot_table(
                index=base_index,
                columns="scheme",
                values=delta_metric_values,
                aggfunc="first",
            )
            pivot.columns = [f"{metric}__{scheme}" for (metric, scheme) in pivot.columns]
            pivot = pivot.reset_index()
        else:
            pivot = qs[base_index].drop_duplicates().reset_index(drop=True)

        delta_rows = []
        for pair in delta_pairs:
            a = pair["a"]
            b = pair["b"]
            name = pair["name"]

            fro_a = f"w_rel_fro__{a}"
            fro_b = f"w_rel_fro__{b}"
            max_a = f"w_rel_max__{a}"
            max_b = f"w_rel_max__{b}"

            df = pivot[base_index].copy()
            df["delta_name"] = name

            df["delta_w_rel_fro"] = None
            df["delta_w_rel_max"] = None
            if fro_a in pivot.columns and fro_b in pivot.columns:
                df["delta_w_rel_fro"] = pivot[fro_a] - pivot[fro_b]
            if max_a in pivot.columns and max_b in pivot.columns:
                df["delta_w_rel_max"] = pivot[max_a] - pivot[max_b]

            delta_rows.append(df)

        deltas = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()
        # CONTRACT SURFACE: tables/B_quant_deltas.{parquet|csv}
        # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
        # Tests: rg 'B_quant_deltas' tests/
        write_manifest_artifacts["B_quant_deltas"] = _normalize_write_meta(
            _write_df(deltas, run_dir / "tables" / "B_quant_deltas.parquet", fmt, compression),
            run_dir,
        )

    tables_write_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_format": fmt,
        "requested_compression": compression,
        "artifacts": write_manifest_artifacts,
    }
    # CONTRACT SURFACE: logs/tables_write_manifest.json
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'tables_write_manifest' tests/
    _write_json(tables_write_manifest, run_dir / "logs" / "tables_write_manifest.json")

    print("[build_tables] wrote tables/ A_* and B_*")


if __name__ == "__main__":
    main()
