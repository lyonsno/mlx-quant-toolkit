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
from typing import Iterable

import pandas as pd


_PLOT_INPUTS_MODULE = None


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
    return json.loads(cfg_path.read_text())


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


def build_plots(run_dir: str | Path) -> list[Path]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    _load_config(run_dir_path)
    plot_inputs = _get_plot_inputs_module()
    tables = plot_inputs.load_plot_tables(
        run_dir_path,
        artifact_keys=(
            "A_weight_global_summary",
            "A_weight_layer_summary",
            "B_quant_global_summary",
        ),
    )

    jobs = []
    if "A_weight_global_summary" in tables:
        jobs.append(
            (
                tables["A_weight_global_summary"],
                run_dir_path / "plots" / "global" / "A_weight_global_summary__mean_abs__median.png",
                _plot_a_weight_global,
                "A_weight_global_summary",
            )
        )
    if "A_weight_layer_summary" in tables:
        jobs.append(
            (
                tables["A_weight_layer_summary"],
                run_dir_path / "plots" / "layer" / "A_weight_layer_summary__mean_abs__median.png",
                _plot_a_weight_layer,
                "A_weight_layer_summary",
            )
        )
    if "B_quant_global_summary" in tables:
        jobs.append(
            (
                tables["B_quant_global_summary"],
                run_dir_path / "plots" / "global" / "B_quant_global_summary__w_rel_fro__median.png",
                _plot_b_quant_global,
                "B_quant_global_summary",
            )
        )

    if not jobs:
        return []

    plt = _load_pyplot()
    written: list[Path] = []
    for frame, out_path, plot_fn, artifact_name in jobs:
        if plot_fn(frame, out_path, plt, artifact_name):
            written.append(out_path)

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
