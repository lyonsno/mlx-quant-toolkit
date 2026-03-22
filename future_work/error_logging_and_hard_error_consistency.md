# Error logging + hard-error consistency (updated status + remaining work)

This note tracks what hard-failure auditability exists today, what contracts have already been chosen,
and what still remains worth doing.

Scope: `scripts/collect_data.py`, `scripts/build_tables.py`, and the run artifacts under `runs/<model>/<run>/`.

---

## Quick takeaway

The repo is in much better shape than the original inventory:

- `collect_data.py` now writes `logs/run_failure.json` on hard failures.
- `build_tables.py` now writes `logs/tables_failure.json` on hard failures.
- `build_tables.py` also records table writes and parquet-to-CSV fallback behavior in `logs/tables_write_manifest.json`.
- `scan.strict_index` is no longer an open policy question:
  - `scan.strict_index=True` requires `scan.use_safetensors_index_json=True`
  - and requires an active, successfully parsed index
  - missing indexed shards still fail hard as before

The main remaining auditability improvement is optional polish: decide whether to write an early
`logs/run_health.json` status like `"running"` and then overwrite it on success/failure.

---

## Hard-failure inventory (current state)

### `scripts/collect_data.py`

Current hard-failure paths include:

- explicit `SystemExit`
  - missing `analysis_config.json`
  - missing `model_path`
  - `scan.strict_index=True` with `use_safetensors_index_json=False`
  - `scan.strict_index=True` without an active parsed index
  - missing indexed shards when strict index scan is active
- explicit raised exceptions
  - `PackedSplitError` when `parsing.strict_packed_split=True`
- unhandled exceptions
  - invalid config JSON
  - invalid regex compilation
  - corrupt/invalid weight files
  - decode/runtime failures in dependency paths

Current recording surfaces:

- `logs/run_failure.json` on hard failures
- `logs/run_context.json`, `logs/run_health.json`, `logs/write_manifest.json`, and `logs/warnings.*` on successful completion
- `data/quant_sim.*` per-row `error` values for quant-scheme failures that do not abort the run

Remaining gap:

- there is still no early “started/running” health artifact written before the run does meaningful work

### `scripts/build_tables.py`

Current hard-failure paths include:

- missing input data (`FileNotFoundError`)
- unexpected schemas / missing columns (for example pandas `KeyError`)
- duplicate base keys for `B_quant_deltas`
- other runtime exceptions during aggregation or writing

Current recording surfaces:

- `logs/tables_failure.json` on hard failures
- `logs/tables_write_manifest.json` on successful completion, including fallback metadata
- `tables/*.csv|parquet` outputs on successful completion

Current cleanup behavior:

- duplicate-delta rerun failures invalidate stale owned `build_tables` outputs without wiping unrelated sidecars
- cleanup is hardened for poisoned manifests, symlinks, malformed owned paths, and rerun containment

### `scripts/init_run.py`

Still mostly conventional CLI behavior:

- argparse / filesystem failures surface directly
- no structured failure artifact today

That is acceptable for now unless we decide we want the same auditability pattern across every entrypoint.

---

## Soft-error / warning inventory (current state)

### `collect_data.py`

Recorded on successful run completion via `logs/warnings.{parquet|csv}`:

- `[meta]` metadata/config parsing issues
- `[index]` index discovery/parse/coverage warnings in non-strict paths
- `[extract]` extraction fallbacks and non-strict packed-split mismatches
- `[quant_sim]` MLX unavailable or per-scheme quant failures

Additional surfaces:

- `logs/index_report.json` when index parsing succeeds and the run completes
- `logs/write_manifest.json` for collect-stage artifact writes and fallback metadata
- `logs/run_health.json` summary counts

### `build_tables.py`

Recorded on successful completion via:

- `logs/tables_write_manifest.json`
  - artifact paths
  - requested format/compression
  - parquet-to-CSV fallback metadata where relevant

Unlike the earlier state of the repo, parquet fallback in `build_tables.py` is no longer silent.

---

## `strict_index` contract (chosen and enforced)

The repo now uses the stricter contract:

- `scan.strict_index=True` requires `scan.use_safetensors_index_json=True`
- `scan.strict_index=True` requires an active parsed index
- if an active index is in use, missing indexed shards are a hard error
- for file `model_path`, index discovery may still be logged explicitly, but scan expansion does not occur through the index

This contract is now aligned across:

- code in `scripts/collect_data.py`
- integration tests
- README / safetensors index docs

So this is no longer a pending policy decision.

---

## Remaining worthwhile work

### 1) Optional early `run_health.json`

Decide whether to emit a minimal early health artifact like:

- `status: "running"`
- `started_at`
- `run_dir`

and then overwrite it later with success or error state.

Why it still matters:

- even with `run_failure.json`, a very early crash currently leaves less “shape of the run” context than a started health record would

### 2) Keep the docs aligned as hardening lands

This note originally described a much rougher pre-hardening state. If more auditability work lands later,
update this file and [current_work.md](./current_work.md) together so they stay consistent.

### 3) Decide whether `init_run.py` needs structured failure artifacts

This is low priority. The current repo behavior is acceptable, but if we want one consistent story for every CLI stage,
`init_run.py` is the remaining outlier.

