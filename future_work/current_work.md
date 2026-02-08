Remaining Identified Issues

- Hard errors are inconsistently surfaced (SystemExit vs tracebacks) and usually produce no structured failure artifacts. See `future_work/error_logging_and_hard_error_consistency.md`.
- "Optional" deps are not actually optional yet according to `pyproject.toml`.

Planned Changes

- Harden hard-error surfacing to be consistent and produce structured failure artifacts.
- Add CLI entry point to `main.py` and remove placeholder.
- Add license file.
- Modify `pyproject.toml` to canonicalize optional deps.

Open Questions

- We need to pick a license.

Recently Resolved (doc drift cleanup)
- `packed_split.projs` values are canonicalized through `parsing.proj_aliases` for packed splits, so known aliases (e.g., `w1`/`w2`) now emit canonical `proj`/`derived_tensor` labels.
- Unmapped proj canonicalization uncertainty is now surfaced end-of-run via `logs/proj_canonicalization_report.{parquet|csv}` with aggregated rows, plus warning summaries in `logs/warnings.*`:
  - kept-raw summary: `[proj] unmapped proj tokens kept raw: ... See ...` (counts from `action="kept_raw"` only)
  - strict-drop summary: `[proj] strict proj_group dropped tensors due to unmapped proj tokens: ... See ...` (from `action="dropped_strict"` rows)
- Add context manager around opening .npz files
- Add `logs/run_health.json` summarizing: files scanned, tensors observed, extracted-by-rule vs fallback counts, unmatched count, and (if index-active) missing/extra shard/tensor counts.
- Add run-level visibility for fallback usage (via `logs/run_health.json` counts for rule vs fallback extraction).
- `proj_group` captures are canonicalized via `parsing.proj_aliases` (tests: `tests/test_proj_group_normalization.py`).
- `model_path` supports single-file checkpoints via `_iter_weight_files` (tests: `tests/test_proj_group_normalization.py`).
- `scan.strict_index` now requires an active index (missing/invalid is a hard error), and single-file `model_path` no longer expands via index; index discovery is still logged explicitly.
- Delta math in `build_tables.py` is covered with deterministic values (tests: `tests/test_optional_mlx.py`).

Optional Improvements
- write a minimal “started” run_health early (status "running", start_time) and then overwrite it at the end with "success" + outputs_written. That way even crashes leave something behind.
- Emit a warning when `delta_pairs` references schemes not present in `quant_sim` (helps catch typos early).
