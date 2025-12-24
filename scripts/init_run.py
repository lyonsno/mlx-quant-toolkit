#!/usr/bin/env python3
"""
Initialize an analysis run directory and write a config template.

Usage:
  python init_run.py --root ./runs --model-id qwen3next-80b-a3b --run-name hf-bf16 --model-path /path/to/model
"""

import argparse
import datetime as dt
import json
from pathlib import Path

DEFAULT_CONFIG = {
  "model_path": "",

  "scan": {
    "extensions": [".safetensors", ".npz"],
    "experts_only": True,
    "include_shared_expert": True,
    "inventory_all_tensors": True,
    "max_files": None
  },

  "parsing": {
    "layer_regex": r"(?:^|\.)layers\.(\d+)(?:\.|$)",
    "expert_regex": r"(?:^|\.)experts\.(\d+)(?:\.|$)",

    "proj_aliases": {
      "down_proj": ["down_proj", "w2", ".down.", "ffn_down"],
      "gate_proj": ["gate_proj", "w1", ".gate.", "ffn_gate"],
      "up_proj":   ["up_proj",   "w3", ".up.",   "ffn_up"]
    },

    "shared_expert_keywords": ["shared", "expert"]
  },

  "extract_rules": [
    {
      "name": "generic_3d_expert_bank_separate",
      "match": r".*experts.*\.(down_proj|gate_proj|up_proj)\.weight$",
      "ndim": 3,
      "layout": { "layer_axis": None, "expert_axis": 0, "rows_axis": 1, "cols_axis": 2 },
      "proj_group": 1
    },
    {
      "name": "generic_2d_per_expert_separate",
      "match": r".*experts\.(\d+)\.(down_proj|gate_proj|up_proj)\.weight$",
      "ndim": 2,
      "layout": { "layer_axis": None, "expert_axis": None, "rows_axis": 0, "cols_axis": 1 },
      "expert_group": 1,
      "proj_group": 2
    },

    {
      "name": "EXAMPLE_fused_gate_down_split_rows",
      "enabled": False,
      "match": r".*experts.*\.(w13|gate_down_proj)\.weight$",
      "ndim": 3,
      "layout": { "layer_axis": None, "expert_axis": 0, "rows_axis": 1, "cols_axis": 2 },
      "packed_split": {
        "axis": "rows",
        "splits": [512, 512],
        "projs": ["gate_proj", "down_proj"]
      }
    }
  ],

  "mlx": {
    "enabled": True,
    "device": "cpu"  # "cpu" or "gpu"
  },

  "stats": {
    "eps": 1e-12,
    "sample_per_matrix": 8192,
    "sample_seed": 1337,
    "percentiles_abs": [50.0, 99.0, 99.9],
    "group_outlier_percentile": 95.0,
    "group_sizes_lastdim": [32, 64]
  },

  "quant_schemes": [
    { "name": "affine_q4_g32", "mode": "affine", "bits": 4, "group_size": 32, "enabled": True },
    { "name": "mxfp4_g32",     "mode": "mxfp4",  "bits": 4, "group_size": 32, "enabled": True },
    { "name": "affine_q8_g32", "mode": "affine", "bits": 8, "group_size": 32, "enabled": True }
  ],

  "delta_pairs": [
    { "name": "mxfp4_minus_q4", "a": "mxfp4_g32", "b": "affine_q4_g32" },
    { "name": "q8_minus_mxfp4", "a": "affine_q8_g32", "b": "mxfp4_g32" }
  ],

  "output": {
    "format": "parquet",
    "compression": "zstd"
  },

  "debug": {
    "dump_unmatched_tensors": True,
    "print_progress_every_files": 1
  }
}


def init_run(root: Path, model_id: str, run_name: str, model_path: str | None) -> Path:
    run_dir = root / model_id / run_name
    for sd in [
        "logs",
        "data",
        "tables",
        "cache",
        "cache/sampled_indices",
        "plots",
        "plots/summary",
        "plots/global",
        "plots/block4",
        "plots/layer",
    ]:
        (run_dir / sd).mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_id": model_id,
        "run_name": run_name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "version": 2,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    cfg_path = run_dir / "analysis_config.json"
    if not cfg_path.exists():
        cfg = DEFAULT_CONFIG.copy()
        if model_path:
            cfg["model_path"] = str(Path(model_path).expanduser().resolve())
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[init_run] wrote config template to {cfg_path}")
    else:
        print(f"[init_run] config already exists at {cfg_path} (left unchanged)")

    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-path", required=False, default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    run_dir = init_run(root, args.model_id, args.run_name, args.model_path)
    print(f"[init_run] run directory: {run_dir}")


if __name__ == "__main__":
    main()
