# Custom MLX Quant Tools

Local-first Python pipeline for analyzing Mixture-of-Experts (MoE) expert weight matrices
from `.safetensors` and `.npz`, computing per-expert stats, and optionally simulating
MLX quantization/dequantization error. The outputs are designed to be inspectable and
auditable: each run writes durable JSON “context” and “what was actually written” artifacts.

## Pending decisions (read this first)

Some behaviors are still pending a policy decision, but the pipeline already works end-to-end.
Where this matters in the README, we mark those statements with `*` so you can avoid treating
them as final contracts. This section summarizes the open decisions so readers get context
up front, and the `*` markers below point to the exact spots they affect.

Pending decisions (*):
- How `scan.strict_index` should behave when the index is missing or invalid.
- Whether `packed_split.projs` values must be canonicalized via `proj_aliases`, and how strict to be.
- How to define the contract for single-file `model_path` when an index exists in the parent dir.

## Quickstart

1) Initialize a run directory and edit the config template:

```bash
python scripts/init_run.py \
  --root ./runs \
  --model-id <model> \
  --run-name <run> \
  --model-path /path/to/model
```

2) Collect raw stats (+ optional quant sim):

```bash
python scripts/collect_data.py --run-dir ./runs/<model>/<run>
```

Optional: override the model path at invocation time (captured in `logs/run_context.json`):

```bash
python scripts/collect_data.py --run-dir ./runs/<model>/<run> --model-path /path/to/model
```

3) Build summary tables:

```bash
python scripts/build_tables.py --run-dir ./runs/<model>/<run>
```

## Installation / prerequisites

- Python `3.12.9` (see `.python-version`).
- Dependencies are described in `pyproject.toml`.
- MLX is primarily a macOS-focused stack; quant simulation requires a working `mlx` install.

Common setups:

- Using `uv` (recommended if you already use it): `uv sync`
- Using `venv` + pip:
  - `python -m venv .venv`
  - `./.venv/bin/pip install -e .`

Optional dependencies (runtime behavior):

- Note: `pyproject.toml` currently includes `mlx` and `pyarrow` as dependencies, but the scripts are written to
  degrade gracefully when they are unavailable at runtime (for example: running from source without installing
  the full dependency set, or install failures on unsupported platforms).
- `mlx` is only required for quantization simulation. If `mlx` is not importable,
  `collect_data.py` will warn and still write `matrix_stats` and `quant_sim` (with zero rows).
- Parquet writing requires a working Parquet backend (typically `pyarrow`). If Parquet writing fails for any
  reason (missing backend, invalid compression, etc.), the pipeline falls back to CSV and records that fallback
  in `logs/write_manifest.json`.

## What the scripts do

- `scripts/init_run.py`
  - Creates `runs/<model-id>/<run-name>/` and a template `analysis_config.json`.
- `scripts/collect_data.py`
  - Scans the model files and inventories tensors (`data/tensor_inventory.*`).
  - Extracts expert weight matrices by config-driven rules (with a heuristic fallback).
  - Canonicalizes layouts to `(E,R,C)` or `(L,E,R,C)` and computes per-expert stats (`data/matrix_stats.*`).
  - Optionally runs MLX quantize/dequant simulations (`data/quant_sim.*`).
  - Writes auditability logs (`logs/run_health.json`, `logs/run_context.json`, `logs/write_manifest.json`, etc.).
- `scripts/build_tables.py`
  - Aggregates `matrix_stats` and `quant_sim` into layer/block/global summary tables under `tables/`.

## Configuration (`analysis_config.json`)

`analysis_config.json` is the main contract for how scanning/extraction/statistics behave.
`init_run.py` writes a template with reasonable defaults.

Key top-level sections:

- `model_path`: path to model directory (or a single weight shard file).
- `scan`: which files to scan and how.
- `parsing`: how to interpret tensor names and how strict to be.
- `extract_rules`: regex-driven extraction rules that map tensors into `(E,R,C)` banks.
- `stats`: deterministic sampling and which metrics to compute.
- `mlx` + `quant_schemes`: MLX quant simulation settings.
- `delta_pairs`: optional scheme-vs-scheme deltas computed by `build_tables.py`.
- `output`: preferred output format + compression.
- `metadata`: optional model `config.json` parsing/logging.
- `debug`: extra artifacts and progress printing.

### Scan options

Important keys under `scan`:

- `extensions`: e.g. `[".safetensors", ".npz"]`
- `max_files`: limit how many shards are scanned (useful for quick checks)
- `experts_only`: if true, only analyze tensors that look like expert weights
- `include_shared_expert`: include shared expert tensors when present
- `inventory_all_tensors`: if true, inventory every tensor (even non-float weights)
- `use_safetensors_index_json`: if true, prefer scanning only the shards referenced by an index file
- `strict_index`*: when index mode is active, fail if the index references missing shards

### Parsing options

Important keys under `parsing`:

- `layer_regex`: regex used to parse the layer id (first capture group).
- `expert_regex`: regex used to parse the expert id (first capture group).
- `proj_aliases`: map of canonical proj names to alias strings used for proj inference.
- `shared_expert_keywords`: keywords that must all appear to mark a tensor as a shared expert.
- `proj_group_strict`: when a rule uses `proj_group`, require alias resolution via `proj_aliases` (otherwise skip the rule).
- `strict_packed_split`: if true, packed_split mismatches raise; if false they warn + fall back.

### Extraction rules and canonical shapes

Each extraction rule declares how to map a matched tensor into a canonical axis order:

- `match`: regex applied to tensor name
- `ndim`: expected input ndim
- `layout`: `{layer_axis, expert_axis, rows_axis, cols_axis}` to transpose into `(L,E,R,C)` / `(E,R,C)` / `(R,C)`
- Optional `proj_group` / `expert_group`: regex capture group indices used to extract proj/expert id
- Optional `packed_split`*: split a fused matrix along rows/cols into multiple projections

If no rule matches, a heuristic fallback tries:

- 3D tensors as `(E,R,C)`
- 2D tensors as `(R,C)` with expert id parsed from the tensor name (if possible)

Run-level counts for “rule vs fallback” are recorded in `logs/run_health.json`.

### Canonicalization and packed splits (mental model)

This pipeline has to deal with the fact that different checkpoints store the same logical
“expert weight matrices” in different shapes, axis orders, and naming conventions.
`collect_data.py` normalizes that into a small, explicit representation before computing stats.

Canonicalization (in this repo) means:

- **Canonical axis order.** A rule’s `layout` tells us which input axes correspond to `layer`,
  `expert`, `rows`, and `cols`. We then transpose into `(L,E,R,C)`, `(E,R,C)`, or `(R,C)` so
  downstream code can treat everything uniformly.
- **No numeric normalization.** Canonicalization is about axis semantics (transpose/relabel),
  not changing weight values (no scaling, centering, etc.). Packed splits add slicing, but the
  slices still contain the original values.
- **Preserve provenance.** We keep the original tensor name as `source_tensor` and name the
  analyzed artifact `derived_tensor` (for example: `source_tensor::gate_proj`) so every row in
  `data/matrix_stats.*` can be traced back to the original file/tensor.
- **Usually canonical proj names.** When `proj` is inferred from a tensor name or a regex
  `proj_group`, we resolve aliases via `parsing.proj_aliases` so tables don’t fragment on
  `w1` vs `gate_proj`.

Packed splits are for “fused” tensors that contain multiple projections concatenated together:

- Some models store multiple projs in one tensor (for example: gate+up+down packed along rows or
  cols). A rule’s `packed_split` specifies how to slice the *canonicalized* matrix:
  `{ "axis": "rows"|"cols", "splits": [...], "projs": [...] }`.
- Each slice becomes its own extracted matrix with its own `proj` and a `derived_tensor` like
  `source_tensor::split[rows]::gate_proj`.

Why this adds complexity / how to think about it:

- `layout` answers “what do the axes mean?”; `packed_split` answers “how do I break a fused axis
  into multiple logical matrices?”. The `packed_split.axis` is interpreted *after*
  canonicalization, so configure `layout` first, then choose whether you’re splitting canonical
  rows or canonical cols.
- A wrong split config can either error (best case) or silently mislabel slices (worst case),
  which will skew per-proj aggregates. Use `parsing.strict_packed_split=true` while developing
  rules; when set to `false` the run continues but records `packed_split failed ...` warnings in
  `logs/warnings.*`.
- `packed_split.projs` are currently treated as literal strings*, so prefer canonical proj names
  in that list to avoid fragmenting tables by `proj`.

Practical sanity checks when you add/modify rules:

- `data/matrix_stats.*`: confirm fused tensors produce multiple `derived_tensor` rows and that
  `rows`/`cols` per `proj` match what you expect for that architecture.
- `logs/warnings.*` (and optionally `data/unmatched_tensors.*`): confirm you’re not “getting
  results” via a fallback path that hid a rule/packed_split mismatch.

### Output format and fallback

The preferred format is controlled by `output.format`:

- `parquet` (preferred) writes `*.parquet` when possible
- `csv` writes `*.csv`

`output.compression` is passed to Parquet writers (ignored for CSV).

When Parquet writing fails, the pipeline falls back to CSV for that artifact and records
the error in `logs/write_manifest.json`.

### Metadata options

Important keys under `metadata`:

- `mode`: currently informational/reserved (default: `"validate"`); does not change behavior yet.
- `enabled`: if true, parse a nearby `config.json` and emit metadata logs.
- `config_path`: optional override path to `config.json` (relative paths resolve under `model_path`).

### Debug options

Important keys under `debug`:

- `dump_unmatched_tensors`: if true, write `data/unmatched_tensors.*` for expertish tensors that failed extraction.
- `print_progress_every_files`: progress log cadence (0 to disable).

## Run outputs

A run directory looks like:

```text
runs/<model-id>/<run-name>/
  manifest.json
  analysis_config.json
  data/
    tensor_inventory.{parquet|csv}
    matrix_stats.{parquet|csv}
    quant_sim.{parquet|csv}
    unmatched_tensors.{parquet|csv}          (optional; requires `debug.dump_unmatched_tensors`)
  tables/
    A_weight_layer_summary.{parquet|csv}
    A_weight_block4_summary.{parquet|csv}
    A_weight_global_summary.{parquet|csv}
    B_quant_layer_summary.{parquet|csv}
    B_quant_block4_summary.{parquet|csv}
    B_quant_global_summary.{parquet|csv}
    B_quant_deltas.{parquet|csv}             (optional; requires `delta_pairs`)
  logs/
    warnings.{parquet|csv}                   (only if warnings were emitted)
    index_report.json                        (only if index mode is active and parses)
    model_config.raw.json                    (metadata enabled + config found)
    model_shape_budget.json                  (metadata enabled + config found)
    run_context.json                         (always)
    run_health.json                          (always)
    write_manifest.json                      (always)
  cache/
    sampled_indices/                         (deterministic sampling cache)
  plots/                                     (created by init_run; reserved for plots)
    summary/
    global/
    block4/
    layer/
```

### Auditability artifacts (logs)

- `logs/run_context.json` records:
  - configured vs resolved `model_path`
  - any CLI overrides (e.g. `--model-path`)
  - the final scan plan (`scan_mode`, scanned files count/examples, etc.)
  - index status (`disabled` / `not_found` / `active` / `unavailable` / `error`)
  - index discovery fields (`searched`, `found`, `active`) and the resolved `index_path` (or null)
- `logs/write_manifest.json` records:
  - requested output settings (`format`, `compression`)
  - the actual written artifact paths, formats, row counts, and Parquet→CSV fallbacks
- `logs/run_health.json` records:
  - scan summary (files scanned, tensors observed)
  - extraction summary (rule vs fallback counts, unmatched counts)
  - output summary (row counts, format, and whether optional artifacts were written)
  - the effective `config_used` snapshot (including any CLI overrides)
  - tensor name format info + a small set of example tensor names
  - index summary counts when index mode is active

### Data artifacts (high-level schema)

- `data/tensor_inventory.*`: one row per observed tensor
  - key columns: `file`, `tensor_name`, `dtype`, `shape`, `ndim`, `nbytes`
  - when index mode is active: `in_index`, `index_shard`
- `data/matrix_stats.*`: one row per extracted expert matrix
  - key columns: `file`, `source_tensor`, `derived_tensor`, `layer`, `block4`, `proj`, `expert_id`,
    `is_routed_expert`, `is_shared_expert`, `rows`, `cols`, `dtype`
  - includes numeric metrics like `mean_abs`, `max_abs`, `p99_abs`, and groupwise outlier ratios (e.g. `g32_*`)
  - conventions: unknown `layer` is recorded as `-1`; shared experts use `expert_id = -1`
- `data/quant_sim.*`: one row per (expert, scheme) simulation result
  - key columns: `file`, `source_tensor`, `derived_tensor`, `layer`, `block4`, `proj`, `expert_id`,
    `is_shared_expert`, `scheme`, `mode`, `bits`, `group_size`, `w_rel_fro`, `w_rel_max`, `scale_*`, `bias_*`, `error`
  - if a scheme fails, rows are still emitted with `error` populated (so you can see coverage)

## Safetensors index support (`model.safetensors.index.json`)

If `scan.use_safetensors_index_json=true` and an index file exists (either
`model.safetensors.index.json` or `*.safetensors.index.json` in the model directory),
`collect_data.py` will prefer scanning only the shards referenced by the index.

Note: if `model_path` points to a single shard file, index discovery is performed in the
parent directory. If an index is found, scanning may expand to additional shards referenced
by that index*. The final decision is recorded in `logs/run_context.json` under `scan_plan`.

When index mode is active:

- `logs/index_report.json` lists missing/extra shards and missing/extra tensors, and records
  `extra_safetensors_files_on_disk` plus any `index_metadata` from the index file.
- `data/tensor_inventory.*` includes `in_index` and `index_shard` columns.
- `logs/run_context.json` and `logs/run_health.json` include index status and counts.

If `scan.strict_index=true`, missing indexed shards will cause a non-zero exit.

Pending decisions (*): strict_index behavior when the index is missing/invalid; packed_split proj canonicalization policy; file `model_path` + index expansion behavior.

## Troubleshooting

- Parquet unexpectedly became CSV: check `logs/write_manifest.json` for the fallback `error`.
- bfloat16 decode errors: install `ml-dtypes` (NumPy needs it to handle `"bfloat16"` from safetensors).
- Packed split failures: set `parsing.strict_packed_split=false` to warn + fall back (see `logs/warnings.*`).
- Index strictness: set `scan.strict_index=false` to warn + continue when shards are missing.
- MLX missing or failing: set `mlx.enabled=false` to skip quant sims; errors and skips are recorded in `logs/warnings.*`.

## Tests

Run unit tests:
- `uv run make test`
- `make test` (Makefile uses `./.venv/bin/python`)
- fallback: `python -m unittest discover -s tests`

run in verbose mode:
- `make verbose-test`
- fallback: `python -m unittest discover -s tests -v`

Run one test module:
- `uv run python -m unittest tests.test_optional_mlx`
- `uv run python -m unittest tests.test_split_along_axis`

Tiny-fixture pipeline:
- `python scripts/init_run.py --root ./runs --model-id <model> --run-name <run> --model-path /path/to/model`
- `python scripts/collect_data.py --run-dir ./runs/<model>/<run>`
- `python scripts/build_tables.py --run-dir ./runs/<model>/<run>`

Subprocess-based acceptance tests must use:
- `sys.executable`, `cwd=repo_root`, `capture_output=True`, and `PYTHONWARNINGS=default`


## Config / metadata examples

`example_safetensors_folder_metadata_convention_variance/.../` contains example
`config.json` and `model.safetensors.index.json` files from various checkpoints and
folder conventions. These are used as test assets and as planning/reference material. One primary purpose is to demonstrate the LACK OF ANY STABLE SCHEMA and give *sample* of the potential variance.
