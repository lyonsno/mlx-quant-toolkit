# AGENTS.md — custom_mlx_quant_tools

This repository is a small, local-first Python pipeline for analyzing Mixture-of-Experts (MoE) weight matrices
(from `.safetensors` and `.npz`), computing per-expert stats, optionally simulating MLX quantization error,
and building summary tables.

Your job as an agent: make small, correct, test-backed changes quickly, without breaking the CLI pipeline.

---

## Repo orientation (what matters)

### Entry points
- `scripts/init_run.py`
  - Creates a run directory, writes `manifest.json`, and writes a config template `analysis_config.json`.
- `scripts/collect_data.py`
  - Scans model files, extracts expert matrices using config rules, canonicalizes layouts, computes stats,
    optionally runs MLX quant/dequant simulation, writes outputs to `runs/.../data/` and warnings to `runs/.../logs/`.
- `scripts/build_tables.py`
  - Aggregates `matrix_stats` + `quant_sim` into layer/block/global tables, and optional delta tables.

### Outputs (behavioral contract)
- `runs/<model-id>/<run-name>/data/`:
  - `tensor_inventory.*`, `matrix_stats.*`, `quant_sim.*`
  - optionally `unmatched_tensors.*` and `warnings.*`
- `runs/<model-id>/<run-name>/tables/`:
  - `A_*` weight summaries, `B_*` quant summaries, optionally `B_quant_deltas.*`
- Output format is config-driven (`parquet` preferred, with CSV fallback).

---

## Ground rules (non-negotiable)

1. **Do not introduce a new test framework.**
   - Tests are `unittest`-style. Keep it consistent.

2. **Keep tests fast and deterministic.**
   - Small arrays, small temporary fixtures, no network, no giant model downloads.

3. **Respect optional dependencies.**
   - `mlx` is optional. The pipeline must still run and write outputs when MLX is unavailable.
   - Tests should avoid requiring real MLX; prefer stubs.

4. **Avoid “tests that merely look good.”**
   - Don’t write vacuous/self-passing tests (e.g., only checking a file exists).
   - Prefer asserting concrete invariants: shapes, columns, row counts, warnings/errors emitted, numeric identities on tiny examples.
   - Use this checklist whenever applicable.
   	•	Does this test assert row counts (not just file existence)?
	•	Does it assert key columns and at least one meaningful value?
	•	Does it avoid depending on print formatting?
	•	If randomness exists, did you force determinism (e.g., sample_k >= total)?

5. **Prefer minimal diffs.**
   - No drive-by refactors unless explicitly requested.

---

## How to run things (try these in order)

### Run unit tests (preferred)
- `uv run make test`  
- If `uv` is unavailable: `make test`
- for verbose replace `make test` with make `verbose-test`

### Run a specific test module
- `uv run python -m unittest tests.test_split_along_axis`
- `uv run python -m unittest tests.test_optional_mlx`

### Run the pipeline manually
- `python scripts/init_run.py --root ./runs --model-id <model> --run-name <run> --model-path /path/to/model`
- `python scripts/collect_data.py --run-dir ./runs/<model>/<run>`
- `python scripts/build_tables.py --run-dir ./runs/<model>/<run>`

When running subprocess-based acceptance tests, ensure they use:
- `sys.executable`
- `cwd=repo_root`
- `capture_output=True`
- `PYTHONWARNINGS=default` in the env (so warnings show up reliably)

---

## Test design conventions for this repo

### Unit tests
Use unit tests to lock down:
- pure helpers (e.g., array splitting / validation),
- error-message propagation and formatting,
- shape / axis handling,
- config-driven edge cases.

Patterns that are already “normal” here:
- importing a script module via `importlib.util.spec_from_file_location` (because code lives in `scripts/`)
- monkeypatching module globals (e.g., swapping `collect_data.mx` with a stub) and restoring in `finally`
- `np.testing.assert_array_equal` for small deterministic arrays

### Acceptance tests (integration-ish)
Use acceptance tests to lock down:
- the CLI script runs end-to-end on a tiny fake “model dir”
- outputs are written (and not empty / not malformed)
- warnings/errors are emitted as expected

Fixture strategy:
- Use a temporary directory.
- Write a minimal `.npz` with one tensor name that matches config regex rules.
- Provide a stub `mlx` package via `PYTHONPATH` to simulate:
  - MLX missing (ImportError)
  - quantize failing (RuntimeError)
  - etc.

---

## Packed split + strictness expectations

This repo has a “packed split” feature (split a fused matrix into multiple projs).

If `parsing.strict_packed_split` is true:
- packed-split mismatch should **fail** (raise a PackedSplitError / non-zero exit).

If false:
- packed-split mismatch should **warn + fall back** (pipeline should still produce stats outputs).

If you touch this behavior, you must update/extend tests.

---

## Workflow protocol (fits a linear loop)

When a user request includes **tests + fix**, default to a two-phase approach:

### Phase 1 — Tests only (default safe posture)
- Write the unit test(s) and/or acceptance test(s).
- Ensure the tests would meaningfully fail on the pre-fix behavior.
- Stop and report:
  - what tests you added,
  - what failure you expect (and why),
  - what command(s) to run.

### Phase 2 — Implementation
- Implement the smallest fix that makes the tests pass.
- Run the test command(s) again.
- Report:
  - what changed,
  - why it fixes the failing tests,
  - what commands you ran and the outcome.

If the user explicitly says “do it end-to-end in one go,” you can do both phases without stopping.

---

## Progress notes (lightweight but persistent)

For each ticket/issue, create or update:
- `agents/<ticket_slug>_progress.md`
- continually update the file as you work, not only at the end.
Keep it short but concrete:
- Goal
- Plan
- Changes made (files + high-level)
- Decisions / tradeoffs
- Any assumptions you made
- Commands run + result (or why you couldn’t run them)
- Do not modify existing file contents, only append to it or annotate it.
- Do not delete the file.

The goal is not bureaucracy; it’s to make review faster later.

---

## Style + correctness notes (numerical code)

- Prefer explicit axis/shape handling over cleverness.
- When emitting tables, keep column names stable unless a user asks otherwise.
- When catching exceptions for “continue but record error,” include useful context in the recorded error string.
- For test assertions, prefer asserting on the actual written artifacts (data/*.csv, tables/*.csv, logs/warnings.csv) over matching printed output. Matching stdout is allowed only for exit-code / crash-path tests.
- Avoid float nondeterminism in tests: compare exact integers or small arrays, or use tolerances intentionally.
- When testing weight stats, set sample_per_matrix >= rows*cols so percentiles are computed on the full matrix and are deterministic.