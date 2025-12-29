Custom MLX Quant Tools

This repo is a small pipeline for analyzing Mixture-of-Experts (MoE) weight
matrices and comparing MLX quantization schemes. It operates on a model
checkpoint directory, extracts expert weight matrices based on configurable
rules, computes summary statistics, optionally runs MLX quantization/dequant
simulations, and then builds aggregated tables for review.

What the scripts do

- `scripts/init_run.py` creates a run directory under `runs/<model-id>/<run-name>`
  with a template `analysis_config.json` and subfolders for data, tables, logs,
  and plots. The config defines how weights are discovered and parsed.
- `scripts/collect_data.py` scans the model files (safetensors or npz), matches
  tensors to expert weight matrices using regex rules, canonicalizes layouts to
  `(E,R,C)` or `(L,E,R,C)`, computes per-expert weight stats, and (if enabled)
  runs MLX quantization simulations for each scheme. Results are written to
  `runs/.../data/`.
- `scripts/build_tables.py` loads the raw data and writes aggregated tables for
  layer/block/global summaries, plus optional per-scheme deltas, into
  `runs/.../tables/`.

Typical usage

1) Initialize a run directory and edit the config:

```bash
python scripts/init_run.py --root ./runs --model-id <model> --run-name <run> --model-path /path/to/model
```

2) Collect raw stats and quantization simulations:

```bash
python scripts/collect_data.py --run-dir ./runs/<model>/<run>
```

3) Build summary tables:

```bash
python scripts/build_tables.py --run-dir ./runs/<model>/<run>
```

Outputs

- `runs/.../data/` includes `tensor_inventory`, `matrix_stats`, `quant_sim`,
  and optional `unmatched_tensors` and `warnings`.
- `runs/.../tables/` includes aggregated summaries for weight stats and quant
  sims, plus optional delta tables between schemes.
