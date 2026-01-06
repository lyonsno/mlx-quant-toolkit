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

## Workflow protocol (default linear loop)

When a user request includes **tests + fix**, default to a two-phase approach.

### Phase 1 — Tests only (default safe posture)
- Update your progress file **continually** as you work (append-only; see “Progress notes”).
- Write the unit test(s) and/or acceptance test(s).
- Ensure the tests would meaningfully fail on the pre-fix behavior.

Stop and report:
- what tests you added (files + intent),
- the exact exception / failure message you expect on pre-fix (or the missing symbol / wrong output shape),
- one sentence on why this failure demonstrates the test is non-vacuous,
- if you can run tests, run them and paste a short excerpt of the failing output (no walls of text, just the key line),
- if you cannot run tests, say what prevented it (missing dependency, environment, etc.), not just “not run.”

Done when:
- tests exist,
- they fail for the intended reason pre-fix (either observed by running, or described precisely),
- your progress note reflects what you did and any assumptions/ambiguities you hit.

### Phase 2 — Implementation
- Implement the smallest fix that makes the Phase 1 tests pass.
- Run the test command(s) again.

Report:
- what changed (files + short description),
- why it fixes the failing tests,
- what commands you ran and the outcome,
- any behavior changes and any new/changed output artifacts.

Done when:
- tests pass,
- CLI pipeline still runs on a tiny fixture,
- diff stays within ticket scope (or extra scope is justified explicitly).

If the user explicitly says “do it end-to-end in one go,” you can do both phases without stopping.

## Phase 3 - Iteration
- if new adjustments are requested, the process should start over from phase one, and stop for test review before preceding to implementation
---

## Ground rules (non-negotiable)

1) **Do not introduce a new test framework.**
- Tests are `unittest`-style. Keep it consistent.

2) **Keep tests fast and deterministic.**
- Small arrays, small temporary fixtures, no network, no giant model downloads.

3) **Respect optional dependencies.**
- `mlx` is optional. The pipeline must still run and write outputs when MLX is unavailable.
- Tests should avoid requiring real MLX; prefer stubs.

4) **Prefer minimal diffs / limited blast radius.**
- No drive-by refactors unless explicitly requested.
- If you touch files not required by the ticket, justify each extra file in the progress log (one sentence each).
- No formatting-only diffs outside the touched module(s).

---

## Anti-vacuity and anti-sandbagging checklist (use whenever applicable)

Avoid “tests that merely look good.” Don’t write vacuous/self-passing tests (e.g., only checking a file exists).
Prefer asserting concrete invariants: shapes, columns, row counts, warnings/errors emitted, numeric identities on tiny examples.

Ask yourself:
- Does this test assert row counts (not just file existence)?
- Does it assert key columns and at least one meaningful value?
- Does it avoid depending on print formatting?
- If randomness exists, did you force determinism (e.g., set sample_k >= total or fix the seed)?

Ticket-specific guardrails:
- Every ticket that changes scanning/selection/filtering must include at least one “poison pill” / “fail-if-touched” fixture.
  - Example: add an extra file that is invalid so the run only succeeds if the scanner truly ignores it.
  - Prefer invalid content over permission/chmod tricks for portability.
- Every ticket that introduces a report (index_report, warnings, etc.) must include at least one test where each category is non-empty (when feasible).
  - Example: don’t only assert `extra_tensors == []`; force a scenario where it’s `["…"]`.
- For any new config key: add a test for “key absent behaves sensibly” (backward compatibility with older configs).

---

## How to run things (try these in order)

### Run unit tests (preferred)
- `uv run make test`
- If `uv` is unavailable: `make test`
- For verbose output: `make verbose-test`
- If one-time network permissions are needed to enable running tests in your environment, request it.

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

## `logs/run_health.json` upkeep

- This pipeline records relevant run health stats (files scanned, tensors observed, extracted-by-rule vs fallback counts, unmatched count, and (if index-active) missing/extra shard/tensor counts, config file settings at run time, model name if available, time, date, etc.) during every run.
- If your change touches one of these metrics, make sure the accurate value still ends up in the `run_health.json` file.
- If your change adds a stat or metric that would make sense to include in this file, state that you plan to include it and test for its presence.
- If you are unsure if something you add belings in the file, **ASK THE USER AS SOON AS IT OCCURS TO YOU**

---

## Progress notes (lightweight but persistent)

For each ticket/issue, create (if necessary), and then update as you work:
- `docs/agent_progress_reports/<ticket_slug>_progress.md`

Append-only rules:
- Append new entries as you work (timestamps are optional but helpful).
- Do not delete or rewrite earlier content; corrections should be new lines.

Keep it short but concrete:
- Goal
- Plan
- Changes made (files + high-level)
- Decisions / tradeoffs
- Any assumptions you made
- Known open questions / ambiguities (when encountered, even if resolved)
- Risk of test loophole (if you notice any, e.g., “could be passed by filtering outputs”)
- Commands run + result (or why you couldn’t run them)
- Add any and all commit hashes that emerged during the execution of the ticket, as soon as the commit is performed.

The goal is not bureaucracy; it’s to make review faster later.

---

## Style + correctness notes (numerical code)
- Include numerous brief high level descriptive comments of why code is doing what it is. Do not include them if they are redundant, but attempt to comment at a high enough level that it cannot be redundant, even with self documenting code.
- Prefer explicit axis/shape handling over cleverness.
- When emitting tables, keep column names stable unless a user asks otherwise.
- When catching exceptions for “continue but record error,” include useful context in the recorded error string.
- For test assertions, prefer asserting on actual written artifacts (data/*.csv, tables/*.csv, logs/warnings.csv) over matching printed output.
  - Matching stdout/stderr is allowed only for:
    - crash-path / exit-code tests where stderr content is the contract (e.g., strict mode errors),
    - or when stdout is explicitly part of a stable CLI contract (rare here).
  - If you need to prove something happened, prefer checking logs/*.json, warnings.csv, or a report artifact.
- Avoid float nondeterminism in tests: compare exact integers or small arrays, or use tolerances intentionally.
- When testing weight stats, set `sample_per_matrix >= rows*cols` so percentiles are computed on the full matrix and are deterministic.