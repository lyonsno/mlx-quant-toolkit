# Custom MLX Quant Tools - Navigation Reference

## Project Overview

A Python pipeline for analyzing Mixture-of-Experts (MoE) weight matrices from `.safetensors` and `.npz` files. It computes per-expert statistics, optionally simulates MLX quantization error, and builds summary tables.

**Key characteristics:**
- Local-first, no network dependencies during execution
- Config-driven behavior via `analysis_config.json`
- Produces both data artifacts (statistics) and auditability logs
- Supports optional MLX integration (falls back gracefully if unavailable)

---

## Directory Structure

```
custom_mlx_quant_tools/
├── scripts/              # Main pipeline modules (importable Python)
│   ├── init_run.py       # Create run directory + config template
│   ├── collect_data.py   # Main scanning/extraction/stats pipeline
│   ├── collect_*.py      # Sub-modules (extract, io, pipeline, quant, stats, reporting)
│   ├── metadata.py       # Config.json parsing
│   ├── plot_inputs.py    # Table discovery/normalization for plotting
│   └── build_plots.py    # Plot generation from table artifacts
├── tests/                # Unit and acceptance tests (unittest-style)
├── runs/                 # Output root (created by init_run)
│   └── <model-id>/<run-name>/
│       ├── data/         # Core artifacts: tensor_inventory, matrix_stats, quant_sim
│       ├── tables/       # Aggregated summaries (A_* for weights, B_* for quant)
│       ├── logs/         # Auditability: run_health, warnings, proj_canonicalization_report
│       └── cache/        # Internal: sampled indices cache
├── docs/                 # Documentation
│   └── safetensors_index_handling.md  # Deep dive on index behavior
├── example_safetensors_folder_metadata_convention_variance/  # Test fixtures
├── tmp_model_npz/        # Temporary fixtures (gitignored)
└── main.py               # Simple entry point (not the main pipeline)

Key config files:
├── AGENTS.md             # Agent-specific rules (this file is for AI assistants)
├── README.md             # User-facing documentation
├── pyproject.toml        # Base deps: ml-dtypes, numpy, pandas, safetensors; optional extras: parquet (pyarrow), plot (matplotlib + pyarrow), mlx (mlx, mlx-lm)
└── Makefile              # Test commands: `make test`, `make verbose-test`
```

---

## Key Scripts and Their Roles

### Entry Points (CLI)

| Script | Purpose | Outputs |
|--------|---------|---------|
| `scripts/init_run.py` | Creates run directory with `manifest.json` and `analysis_config.json` template | `runs/<model-id>/<run-name>/analysis_config.json` |
| `scripts/collect_data.py` | Main pipeline: scan files → extract matrices → compute stats → optionally quant-sim | `data/` artifacts, `logs/` audit artifacts |
| `scripts/build_tables.py` | Aggregates `matrix_stats` + `quant_sim` into layer/block/global tables | `tables/A_*`, `tables/B_*`, optional `B_quant_deltas.*` |
| `scripts/build_plots.py` | Loads table artifacts and renders baseline plots | `plots/global/*.png`, `plots/layer/*.png` |

Run sequence: `init_run` → edit `analysis_config.json` → `collect_data` → `build_tables` → `build_plots` (optional).

**Core Concepts**: See `quick_concept_reference.md` for detailed explanations of canonicalization, extraction rules, packed splits, projection canonicalization, and safetensors index support.

---

## Quick Reference: Key Files to Read

For understanding specific areas:

| Area | File(s) |
|------|---------|
| Extraction logic | `scripts/collect_extract.py` |
| Canonical layout handling | `scripts/collect_extract.py` (entrypoint: `_canonicalize_layout`, see definition at line ~190; used by `_apply_rules` for standard + packed-split paths) |
| Index discovery/validation | `scripts/metadata.py` (functions: `find_safetensors_index_json`, `parse_safetensors_index`) + `scripts/collect_data.py` (index activation/strict-index validation flow) |
| Statistics computation | `scripts/collect_stats.py` |
| Reporting/warnings | `scripts/collect_reporting.py` |
| Table aggregation | `scripts/build_tables.py` |
| Plotting inputs | `scripts/plot_inputs.py` |
| Plot generation | `scripts/build_plots.py` |
| Config validation | `scripts/collect_pipeline.py` (loads and validates config) |
| Contract surfaces | Search `CONTRACT SURFACE:` in `scripts/` |

---

## Version & Environment

- Python ≥ 3.12.9
- Base dependencies: `ml-dtypes`, `numpy`, `pandas`, `safetensors`
- Optional extras: `mlx` (`mlx`, `mlx-lm`), `parquet` (`pyarrow`), `plot` (`matplotlib`, `pyarrow`)
- Uses `uv` for package management (Makefile assumes `.venv/bin/python`)
- Tests are `unittest`-based, no pytest
