# Error logging + hard-error consistency (unified inventory + recommendations)

This note unifies two inventories of:
- what “error types” exist in the pipeline today,
- where each is recorded (stdout/stderr vs durable artifacts),
- and what policy decisions would make hard errors consistent and auditable.

Scope: `scripts/collect_data.py`, `scripts/build_tables.py`, and the run artifacts under `runs/<model>/<run>/`.

---

## Quick takeaway

Today, “hard errors” are inconsistent:
- some are `SystemExit` with a short message (no traceback),
- others are uncaught exceptions with a traceback,
- and almost none of them persist structured run artifacts because `run_context.json`, `run_health.json`,
  `write_manifest.json`, and `warnings.*` are written at the end of a successful run.

Separately, `scan.strict_index` semantics are currently “strict only when an index is active”, but tests/readme
have been trending toward “strict implies index must be active and valid”.

---

## Error type inventory (union)

### Hard errors (non-zero exit / crash)

#### `scripts/collect_data.py`

1) **Explicit `SystemExit` (short message, no traceback)**
- Missing `analysis_config.json`: `_load_config()` raises `SystemExit`.
- `model_path` does not exist: `main()` raises `SystemExit`.
- `scan.strict_index=True` + missing indexed shard(s) *when index is active (`index_ready`)*: `SystemExit`.

**Recording surfaces**
- stderr/stdout only.
- No `logs/run_context.json`, `logs/run_health.json`, `logs/write_manifest.json`, `logs/warnings.*` because the run
  aborts before the “write outputs + write logs” block.

2) **Explicit exception raised to crash (traceback)**
- `PackedSplitError` from packed split mismatch when `parsing.strict_packed_split=True`.

**Recording surfaces**
- traceback only.
- Early artifacts may already exist (e.g., metadata files) if they were written before the crash.

3) **Implicit/unhandled exceptions (traceback)**
- Invalid config JSON (`JSONDecodeError`) and missing required keys (`KeyError`).
- Invalid regex compilation (`re.error`).
- Corrupt/invalid weight files (`.safetensors`, `.npz`) and safetensors/zip exceptions.
- bfloat16 decode path raising `RuntimeError` when NumPy cannot decode `bfloat16`.

**Recording surfaces**
- traceback only.
- No structured run logs unless they were written earlier (rare today).

#### `scripts/build_tables.py`
- Missing input data: `_read_df(...)` raises `FileNotFoundError`.
- Unexpected schemas: pandas errors (e.g., `KeyError` on missing columns).

**Recording surfaces**
- traceback only.
- No structured “tables run context” artifacts.
- Parquet write fallback is silent (CSV is written, but the exception is not recorded anywhere).

#### `scripts/init_run.py`
- Mostly unhandled filesystem/argparse errors (permissions, invalid paths, etc.).

---

### Recorded warnings / soft errors (run continues)

These are typically recorded only if `collect_data.py` reaches the end-of-run write phase.

#### `logs/warnings.{parquet|csv}` (via `warn_log`)
Produced by `collect_data.py` under prefixes:
- `[meta]`: metadata/config.json missing/invalid, metadata module unavailable
- `[index]`: index module/helpers unavailable; index parse failure; index coverage mismatches
- `[extract]`: rule application failures; fallback extraction failures; packed_split mismatch when non-strict
- `[quant_sim]`: MLX missing; per-scheme quantize/dequant failures

#### Index reporting
- `logs/index_report.json` is written only when the index successfully parses (`index_active`) and the run completes.

#### Quant simulation per-row failures
- `data/quant_sim.*` contains an `error` column for scheme failures (rows still emitted for coverage).

#### Parquet → CSV fallback recording
- `collect_data.py` records fallback + error string in `logs/write_manifest.json` per artifact.
- `build_tables.py` falls back to CSV but does not record the failure (no manifest).

#### Run summary counts
- `logs/run_health.json` includes `outputs_written.warnings_rows` and index summary counts.

---

## `strict_index` semantics: current vs intended

### What code does today
- Index discovery + parsing produces `index_ready` only when:
  - index usage is enabled (`use_safetensors_index_json=True`),
  - index exists,
  - and index parsing succeeds.
- `scan.strict_index=True` only enforces shard presence when `index_ready` is true (missing shards => hard error).
- Missing index / invalid index parse currently becomes:
  - warning(s) in `warn_log`,
  - status `not_found`/`error`,
  - and fallback to directory walk scan mode.

### What tests are starting to require
Some integration tests expect:
- `scan.strict_index=True` implies “must have an active index”; missing or invalid index should hard fail.
- `scan.strict_index=True` with `use_safetensors_index_json=False` should hard fail.

This is a policy decision: both interpretations are reasonable; we just need one coherent contract.

---

## Recommendations (smallest coherent set)

### 1) Make hard failures auditable (structured failure artifact)
Add a top-level `try/except/finally` in `scripts/collect_data.py` that guarantees at least one durable artifact
even when the run fails early. Two reasonable patterns:

- **Option A: `logs/run_failure.json` (new file)**
  - Written on any exception / `SystemExit`.
  - Records: timestamps, resolved model_path, configured vs overridden model_path, index status at time of failure,
    exception type/message, and optionally a trimmed traceback.

- **Option B: extend `logs/run_health.json` to support `status="error"`**
  - Write an initial “running” or “started” record early, then overwrite on success with `status="success"` or
    overwrite on error with `status="error"`.
  - Keeps a single “health” file but changes its semantics (needs careful contract update).

Either way: aim to always write a minimal `logs/run_context.json` early too (even if incomplete).

### 2) Decide and enforce `strict_index` contract
Pick one:
- **Contract 1 (permissive strict):** strict only applies if index is active; otherwise warn + fallback.
- **Contract 2 (strict means strict):** if `scan.strict_index=True`, then require:
  - `use_safetensors_index_json=True`
  - index exists and parses successfully
  - (and then enforce missing shards as today)

Then align:
- code behavior,
- test expectations in `tests/test_safetensors_index_integration.py`,
- README config docs.

### 3) (Optional) Improve `build_tables.py` auditability
If we care about reproducibility for tables:
- add a `logs/tables_write_manifest.json` (or similar) for `build_tables.py`, recording format fallbacks and output paths,
  or at least print a warning when parquet write falls back to CSV.

---

## Suggested next actions (if prioritizing)

1) Choose `strict_index` contract (this unblocks test alignment).
2) Add failure artifact emission for `collect_data.py` (most value for “auditability when things go wrong”).
3) Decide whether tables need a manifest (only if tables generation is a first-class pipeline stage).

