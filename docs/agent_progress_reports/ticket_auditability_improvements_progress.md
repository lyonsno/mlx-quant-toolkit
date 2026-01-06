# Ticket: ticket_auditability_improvements — Progress

## Goal
Add auditability artifacts so runs can be reconstructed:
- Persist resolved model_path + CLI overrides (run_context/scan_plan).
- Log index searched/found/activated/not found status every run.
- Record parquet vs CSV fallback and actual written artifacts (write_manifest.json).

## Plan (Phase 1: tests only)
1. Add a collect_data integration test that uses a CLI model_path override + active index and asserts run_context + write_manifest contents (including parquet fallback behavior).
2. Add a collect_data integration test that asserts index status is logged when the index is missing and when it is disabled.
3. Run the new tests and capture the expected pre-fix failure signal.

## Changes made
- Added `tests/test_auditability_artifacts_integration.py`
  - `test_collect_data_writes_run_context_and_write_manifest_with_cli_override_and_index_active`
    - Asserts `logs/run_context.json` captures resolved model_path + CLI override + scan_plan + index status.
    - Forces parquet fallback via invalid compression and asserts `logs/write_manifest.json` records CSV outputs + row counts.
  - `test_collect_data_run_context_logs_index_status_when_index_missing`
    - Asserts index status is `not_found` (searched true, found/active false) when the index file is absent.
  - `test_collect_data_run_context_logs_index_status_when_index_disabled`
    - Asserts index status is `disabled` (searched false) when index usage is disabled.

## Assumptions
- The durable artifact for model_path + CLI overrides will be `logs/run_context.json` and will include a `scan_plan` section.
- The write manifest will be `logs/write_manifest.json` and will list actual output paths plus per-artifact format and fallback status.

## Commands run
- `./.venv/bin/python -m unittest tests.test_auditability_artifacts_integration`
  - Expected pre-fix failure: `AssertionError: False is not true` for missing `logs/run_context.json` (and FileNotFoundError in the index-missing/disabled tests).

## Test adjustments (per review)
- Updated `tests/test_auditability_artifacts_integration.py`
  - Resolve write_manifest paths relative to `run_dir` when they are not absolute.
  - Assert fallback records include a non-empty `error` string.
  - Assert `logs/run_health.json` records `scan_summary.files_scanned == 1` to make the poison-pill check explicit.
  - Assert `run_context.scan_plan.scanned_files_count == 1` and `scanned_files_example` ends with `shard1.npz`.
  - Assert `write_manifest.requested_compression` records the requested compression.
  - Assert `run_context.scan_plan.scanned_files_example` is a relative path (not absolute).

## Updated assumptions
- When a parquet write falls back to CSV, the write manifest includes a non-empty `error` string for that artifact.

## Commands run (after adjustments)
- `./.venv/bin/python -m unittest tests.test_auditability_artifacts_integration`
  - Expected pre-fix failure: `AssertionError: False is not true` for missing `logs/run_context.json` (and FileNotFoundError in the index-missing/disabled tests).
  - Re-ran after scan_plan + compression assertions; same failure (run_context missing).
  - Re-ran after requiring relative `scanned_files_example`; failing with `AssertionError: True is not false` because scan_plan currently stores absolute paths.

---

## Phase 2 — Implementation (scan_plan relativity)

## Changes made
- Updated `scripts/collect_data.py`
  - Normalize `scan_plan.scanned_files_example` to be relative to `model_path` for portability.

## Commands run (post-fix)
- `./.venv/bin/python -m unittest tests.test_auditability_artifacts_integration` (PASS)

---

## Phase 3 — Index found semantics (tests + implementation)

## Changes made
- Added `tests/test_index_found_semantics.py`
  - Asserts `run_context.index.found` is `False` when the index finder returns a candidate path that does not exist.
  - Locks down `index_path` as `None` for missing index files even when a candidate path is returned.
- Updated `scripts/collect_data.py`
  - `index.found` now reflects “index file exists on disk”.
  - `run_context.index.index_path` is only recorded when the path exists.

## Commands run
- `./.venv/bin/python -m unittest tests.test_index_found_semantics` (PASS)

---

## Phase 2 — Implementation

## Changes made
- Updated `scripts/collect_data.py`
  - Emit `logs/run_context.json` with resolved vs configured model_path, CLI overrides, scan_plan (including scanned_files_count/example), and index status every run.
  - Emit `logs/write_manifest.json` capturing requested format/compression and per-artifact output paths, formats, row counts, and fallback errors.
  - Track index status (`disabled`/`not_found`/`active`/`error`) and include it in run_context consistently.
  - Record write outputs via `_write_df` returning metadata, preserving fallback error details for manifest use.

## Commands run (post-fix)
- `./.venv/bin/python -m unittest tests.test_auditability_artifacts_integration` (PASS)
