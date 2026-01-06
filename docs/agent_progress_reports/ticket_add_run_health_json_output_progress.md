# Ticket: ticket_add_run_health_json_output — Progress

## Goal
Add `logs/run_health.json` to `scripts/collect_data.py` summarizing:
- files scanned
- tensors observed
- extracted-by-rule vs fallback counts
- unmatched count
- (if index-active) missing/extra shard/tensor counts
- also: config variables used, name formats/examples for raw/fallback tensors, date+time, and model name (if available)

## Plan (Phase 1: tests only)
1. Add an integration test that runs `scripts/collect_data.py` on a tiny `.npz` model fixture with an active `model.safetensors.index.json`.
2. Make the fixture include:
   - at least one rule-extracted tensor
   - at least one fallback-extracted tensor
   - at least one unmatched expertish tensor
   - index-report categories that are non-empty (missing shard, missing tensor, extra tensor)
   - a “poison pill” invalid `.npz` that should be ignored when index limits the scan
3. Assert `logs/run_health.json` exists and contains stable counts + key fields (model id/name, config_used.model_path reflecting CLI override, and index counts).

## Changes made
- Added `tests/test_run_health_json_integration.py`
  - Runs `collect_data.py` via subprocess on a temp run dir + model dir fixture.
  - Asserts a concrete JSON schema + specific counts for:
    - `files_scanned == 1`
    - `tensors_observed == 3`
    - `extracted_by_rule == 1`, `extracted_by_fallback == 1`, `unmatched_expertish == 1`
    - index counts: expected=2, scanned=1, missing_shards=1, missing_tensors=2, extra_tensors=1
  - Asserts `config_used.model_path` matches the `--model-path` CLI override.

## Non-vacuity / why the test matters
The test fails unless `collect_data.py` actually emits a real `logs/run_health.json` report with correctly computed counts (it can’t be satisfied by a stub that only writes an empty file or only checks “ran without crashing”).

## Commands run
- `python -m unittest tests.test_run_health_json_integration`
  - Failed due to missing deps in the non-venv interpreter: `ModuleNotFoundError: No module named 'numpy'`
- `uv run python -m unittest tests.test_run_health_json_integration`
  - Failed due to sandbox/cache access issues and then an `uv` panic on this machine.
- `./.venv/bin/python -m unittest tests.test_run_health_json_integration`
  - Fails pre-fix as expected:
    - `AssertionError: False is not true` at `self.assertTrue(health_path.exists())` (no `logs/run_health.json` yet)

## Open questions / assumptions
- Assumption: the run-health report is a single JSON file at `run_dir/logs/run_health.json` (not CSV/parquet) and should be written even when other outputs are CSV.
- Assumption: “name formats” can be satisfied by reporting representative raw tensor name examples per category (rule-extracted vs fallback-extracted vs unmatched expertish).

---

## Phase 2 — Implementation

## Changes made
- Updated `scripts/collect_data.py`
  - Always writes `logs/run_health.json` at end of a successful scan.
  - Report includes: `generated_at`, `run` (model_id/run_name from manifest when available, plus resolved model_path),
    `config_used` (post-CLI-override), `scan_summary`, `extraction_summary`, `tensor_name_examples`,
    `derived_tensor_formats`, and `index_summary` (counts + index_path/metadata when active).
  - Extraction counts are tracked independent of `debug.dump_unmatched_tensors` so `unmatched_expertish` remains accurate even when unmatched dumps are disabled.

## Test adjustment
- Updated `tests/test_run_health_json_integration.py` to fix regex escaping in the generated config fixture.
  - The prior fixture patterns were double-escaped (e.g. `\\.`) which prevented rule matching.
  - New patterns match the same intent but reflect how the JSON config is normally written/parsed.

## Commands run
- `./.venv/bin/python -m unittest tests.test_run_health_json_integration`
  - PASS
- `./.venv/bin/python -m unittest tests.test_safetensors_index_integration`
  - PASS

## Follow-up tweak
- Added `tensor_name_formats` to `logs/run_health.json` so the key regexes and enabled rule match patterns are visible without digging through `config_used`.
- Re-ran: `./.venv/bin/python -m unittest tests.test_run_health_json_integration` (PASS)

---

## Robustness follow-up (tests-first)

## Motivation
- Ensure `scan_summary.tensors_observed` reflects the number of tensors actually iterated (not deduped names, and not index-only).
- Ensure `index_summary` remains safe even if future refactors introduce a weird “index_active but weight_map missing” state.
- Add a single-bit completion marker (`status: "success"`) + `duration_seconds` and basic output row-counts for easier partial-run detection in the future.

## Tests added/updated
- Updated `tests/test_run_health_json_integration.py`
  - Added assertions for `status`, `duration_seconds`, and `outputs_written` keys.
  - Added a directory-walk mode test that scans two `.npz` files containing the same tensor name and asserts `tensors_observed == 2` (this fails if the implementation incorrectly dedupes by name).

## Fail-first signal (pre-fix)
- `./.venv/bin/python -m unittest tests.test_run_health_json_integration`
  - Failed with: `AssertionError: None != 'success'` (missing `status` field in `run_health.json`)

## Implementation changes
- Updated `scripts/collect_data.py`
  - Tracks `tensors_observed` as an integer counter incremented for every `(name, arr)` yielded by `_iter_tensors_from_file`.
  - Uses `observed_tensor_names` only for index diffing (and only populates it when index is truly active/usable).
  - Introduces `index_ready = index_active and weight_map is not None` and uses it consistently for all `index_summary` guards.
  - Writes `run_health.json` after outputs are written and includes:
    - `status: "success"`
    - `duration_seconds`
    - `outputs_written` row counts + basic booleans

## Commands run (post-fix)
- `./.venv/bin/python -m unittest tests.test_run_health_json_integration` (PASS)
- `./.venv/bin/python -m unittest tests.test_safetensors_index_integration` (PASS)
