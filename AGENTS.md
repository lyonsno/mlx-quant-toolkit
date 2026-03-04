# AGENTS.md - custom_mlx_quant_tools

This repository is a small, local-first Python pipeline for analyzing Mixture-of-Experts (MoE) weight matrices
(from `.safetensors` and `.npz`), computing per-expert stats, optionally simulating MLX quantization error,
and building summary tables.

Your job as an agent: make small, correct, test-backed changes quickly, without breaking the CLI pipeline.

This file is repo-specific. Follow `~/.codex/AGENTS.md` for global defaults, and use this file for local contracts.

**Note:** Before beginning any work that involves exploring the repository, please read `docs/quick_navigation_reference.md` to understand the project structure, file locations, and navigation conventions.

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

## Workflow protocol (default linear loop)

When a user request includes **tests + fix**, use this two-phase flow unless the user explicitly asks for end-to-end in one pass.

### Phase 1 - Tests only
- Write unit and/or acceptance tests that meaningfully fail on pre-fix behavior.
- Stop and report:
  - test files + intent,
  - exact expected failure signal (exception/message/shape/output mismatch),
  - why the failure is non-vacuous,
  - failing output excerpt if tests were run, or why tests could not be run.

### Phase 2 - Implementation
- Implement the smallest fix that makes Phase 1 tests pass.
- Re-run tests and report:
  - changed files + what changed,
  - why the change fixes the failing tests,
  - commands run and pass/fail outcome,
  - any behavior changes / new or changed artifacts.

### Phase 3 - Iteration
- If new adjustments are requested, restart at Phase 1 and stop for test review before implementing.

---

## Repo-specific quality bar

1) **Do not introduce a new test framework.**
- Tests are `unittest`-style.

2) **Keep tests fast and deterministic.**
- Small arrays, small temporary fixtures, no network, no giant model downloads.

3) **Respect optional dependencies.**
- `mlx` is optional. The pipeline must still run and write outputs when MLX is unavailable.
- Tests should avoid requiring real MLX; prefer stubs.
- Dependency placement contract:
  - Use required `project.dependencies` only for packages needed by the base pipeline on all installs.
  - Use `project.optional-dependencies` for capability-specific features that intentionally degrade when unavailable.
  - Keep extras capability-scoped (e.g. `mlx`, `parquet`, future `plot`) and maintain an `all` union extra.
  - Any new optional extra should ship with at least one fail-meaningful test covering the missing-dependency path.

4) **Prefer minimal diffs / limited blast radius.**
- No drive-by refactors unless explicitly requested.
- No formatting-only diffs outside touched modules.

### Anti-vacuity checklist
Avoid vacuous/self-passing tests (for example, only checking a file exists).
Prefer concrete invariants: shapes, columns, row counts, warnings/errors, tiny deterministic numeric identities.

Required checks:
- assert row counts and key columns,
- avoid print-format-dependent assertions,
- force determinism (`sample_k >= total`, fixed seeds, or full sampling).

Ticket-specific guardrails:
- Scanning/selection/filtering tickets must include at least one "poison pill" / "fail-if-touched" fixture.
  - Prefer invalid content over permission/chmod tricks for portability.
- Tickets introducing a report (`index_report`, warnings, etc.) should include at least one test where each category is non-empty when feasible.
- Any new config key needs a backward-compat test: key absent behaves sensibly.

---

## Running locally

- Unit tests: `uv run make test` (fallback: `make test`; verbose: `make verbose-test`).
- Specific module: `uv run python -m unittest tests.test_split_along_axis`.
- Manual pipeline:
  - `python scripts/init_run.py --root ./runs --model-id <model> --run-name <run> --model-path /path/to/model`
  - `python scripts/collect_data.py --run-dir ./runs/<model>/<run>`
  - `python scripts/build_tables.py --run-dir ./runs/<model>/<run>`

Acceptance-test subprocesses should use:
- `sys.executable`,
- `cwd=repo_root`,
- `capture_output=True`,
- `PYTHONWARNINGS=default`.

---

## Test design conventions for this repo

Patterns that are already normal here:
- import script modules via `importlib.util.spec_from_file_location` (because code lives in `scripts/`),
- monkeypatch module globals (for example, swap `collect_data.mx` with a stub) and restore in `finally`,
- use `np.testing.assert_array_equal` for small deterministic arrays,
- use temp dirs + tiny `.npz` fixtures; provide stub `mlx` via `PYTHONPATH` to simulate missing/failing MLX.

---

## Packed split + strictness expectations

This repo has a "packed split" feature (split a fused matrix into multiple projs).

If `parsing.strict_packed_split` is true:
- packed-split mismatch should **fail** (raise a `PackedSplitError` / non-zero exit).

If false:
- packed-split mismatch should **warn + fall back** (pipeline should still produce stats outputs).

If you touch this behavior, you must update/extend tests.

---

## `logs/run_health.json` upkeep

- This pipeline records run health stats including files scanned, tensors observed, extracted-by-rule vs fallback counts,
  unmatched count, and (if index-active) missing/extra shard/tensor counts, plus run-time config/model metadata.
- If your change touches one of these metrics, ensure the accurate value still ends up in `run_health.json`.
- If your change adds a stat that should belong there, say so and test for its presence.
- If unsure whether a new stat belongs there, ask the user immediately.

---

## Style + correctness notes (numerical code)

- Prefer explicit axis/shape handling over cleverness.
- Keep emitted table column names stable unless the user asks otherwise.
- When catching exceptions for "continue but record error", include useful context in the recorded error string.
- Prefer asserting on written artifacts (`data/*.csv`, `tables/*.csv`, `logs/warnings.csv`) over printed output.
  - `stdout`/`stderr` assertions are mainly for crash-path / exit-code contracts.
- Avoid float nondeterminism in tests: exact integers/small arrays or intentional tolerances.
- For deterministic weight-stat percentiles, set `sample_per_matrix >= rows*cols`.
