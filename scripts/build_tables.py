#!/usr/bin/env python3
"""
Build summary tables from data/matrix_stats and data/quant_sim.

Usage:
  python build_tables.py --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_df(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    # try fallback extensions
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(path)


def _write_df(df: pd.DataFrame, path: Path, fmt: str, compression: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False, compression=compression)
            return
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=False)
            return
    df.to_csv(path.with_suffix(".csv"), index=False)


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "analysis_config.json"
    return json.loads(cfg_path.read_text())


def _quantile_func(q: float, label: str):
    def _fn(s: pd.Series):
        return s.quantile(q)
    _fn.__name__ = label
    return _fn


def _agg_with_funcs(df: pd.DataFrame, group_cols: List[str], value_cols: List[str], agg_funcs: List[Any]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_cols].agg(agg_funcs).reset_index()
    # Flatten MultiIndex columns produced by multi-agg.
    grouped.columns = [
        c if not isinstance(c, tuple) else (c[0] if (len(c) > 1 and (c[1] == "" or c[1] is None)) else f"{c[0]}__{c[1]}")
        for c in grouped.columns
    ]
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_config(run_dir)
    fmt = cfg.get("output", {}).get("format", "parquet")
    compression = cfg.get("output", {}).get("compression", None)

    ms = _read_df(run_dir / "data" / "matrix_stats.parquet")
    qs = _read_df(run_dir / "data" / "quant_sim.parquet")

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
    _write_df(A_layer, run_dir / "tables" / "A_weight_layer_summary.parquet", fmt, compression)

    # per block4/proj
    A_block4 = _agg_with_funcs(ms, ["block4", "proj"], stat_cols, ["median", "mean", "std", p90, p99])
    _write_df(A_block4, run_dir / "tables" / "A_weight_block4_summary.parquet", fmt, compression)

    # global/proj
    A_global = _agg_with_funcs(ms, ["proj"], stat_cols, ["min", p01, "median", p99, "max"])
    _write_df(A_global, run_dir / "tables" / "A_weight_global_summary.parquet", fmt, compression)

    # -------- B: quant sim summaries --------
    qcols = ["w_rel_fro", "w_rel_max", "scale_mean", "scale_max", "bias_mean", "bias_max"]
    qcols = [c for c in qcols if c in qs.columns]

    B_layer = _agg_with_funcs(qs, ["layer", "proj", "scheme"], qcols, ["median", "mean", p90, p99])
    _write_df(B_layer, run_dir / "tables" / "B_quant_layer_summary.parquet", fmt, compression)

    B_block4 = _agg_with_funcs(qs, ["block4", "proj", "scheme"], qcols, ["median", "mean", p90, p99])
    _write_df(B_block4, run_dir / "tables" / "B_quant_block4_summary.parquet", fmt, compression)

    B_global = _agg_with_funcs(qs, ["proj", "scheme"], qcols, ["min", p01, "median", p99, "max"])
    _write_df(B_global, run_dir / "tables" / "B_quant_global_summary.parquet", fmt, compression)

    # -------- deltas (scheme A - scheme B) --------
    delta_pairs = cfg.get("delta_pairs", [])
    if delta_pairs:
        base_index = ["derived_tensor", "layer", "block4", "proj", "expert_id", "rows", "cols"]
        pivot = qs.pivot_table(
            index=base_index,
            columns="scheme",
            values=["w_rel_fro", "w_rel_max"],
            aggfunc="first"
        )
        pivot.columns = [f"{metric}__{scheme}" for (metric, scheme) in pivot.columns]
        pivot = pivot.reset_index()

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
        _write_df(deltas, run_dir / "tables" / "B_quant_deltas.parquet", fmt, compression)

    print("[build_tables] wrote tables/ A_* and B_*")


if __name__ == "__main__":
    main()
