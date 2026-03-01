## 2026-03-01 - Phase 1 (tests-first) start
- Goal: Prepare a safe refactor path for scripts/collect_data.py by first adding tests that define expected helper-module boundaries and compatibility.
- Plan: Add a focused unittest module that requires new helper modules to exist and verifies key helpers remain reachable from collect_data.
- Changes made: Created branch codex/collect-data-refactor-phase1.
- Assumptions: The refactor will keep script behavior stable while moving helpers into small modules under scripts/.
- Open questions: None yet.
- Changes made: Added tests/test_collect_data_helper_module_split_contract.py to define expected helper-module split boundaries and collect_data compatibility re-exports.
- Why non-vacuous: Tests assert concrete symbol-level API contracts and function identity wiring, not just script execution.
- Risk of test loophole: A superficial pass via empty modules is prevented by required symbol checks and callable behavior checks on _split_along_axis.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract
- Result: FAIL (expected pre-fix). Key signal: FileNotFoundError for scripts/collect_extract.py and assertion that helper module file is missing.
- Interpretation: The tests are fail-first for the planned refactor because they require helper modules and symbol wiring that do not exist in current collect_data.py-only layout.

## 2026-03-01 - Review of phase-1 test robustness
- Goal: Assess whether newly added tests robustly cover phase-1 helper extraction work without over-constraining later phases.
- Changes made: Reviewed tests/test_collect_data_helper_module_split_contract.py against existing helper-focused tests and current scripts layout.
- Findings: New contract tests are fail-first, but currently assert presence of helpers/modules planned for later phases (for example _apply_rules/_fallback_extract and collect_quant), which can block incremental phase-1 completion.
- Findings: Existing suite already has strong behavioral tests for _split_along_axis, _canonicalize_layout, _per_expert_weight_stats, _mlx_quant_sim, _iter_tensors_from_file, and _iter_weight_files; coverage gaps remain for direct behavior of _get_sample_indices, _safe_read_json_dict, _write_json, and _write_df as independently moved helpers.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: FAIL. Key signals: missing scripts/collect_extract.py and FileNotFoundError loading helper modules.
- Commands run: uv run python -m unittest tests.test_split_along_axis tests.test_canonicalize_layout tests.test_weight_stats tests.test_iter_tensors_from_file_npz_close tests.test_mlx_quant_sim_error tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_accepts_file_path tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_filters_extensions -v
- Result: PASS (18 tests OK).
- Assumptions: Phase 1 scope remains pure/stateless helper moves first, with _apply_rules/_fallback_extract and optional quant split deferred.

## 2026-03-01 - Follow-up review of reviewer comments
- Goal: Validate reviewer feedback on phase-1 test scope and determine whether test contract should be narrowed before implementation.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: FAIL (expected in current tree). Key lines: missing scripts/collect_extract.py; FileNotFoundError loading helper modules.
- Commands run: uv run python -m unittest tests.test_split_along_axis tests.test_canonicalize_layout tests.test_weight_stats tests.test_iter_tensors_from_file_npz_close tests.test_mlx_quant_sim_error -v
- Result: PASS (16 tests OK), confirming existing behavior coverage for several planned helper moves.
- Assessment: Reviewer concern is valid; current phase-1 contract test enforces helper modules/functions likely targeted for later extraction phases and can force non-minimal implementation.
- Suggested adjustment (not yet applied): split contract assertions by phase and keep phase-1 test focused on pure/stateless helper modules only.
- Changes made: Narrowed tests/test_collect_data_helper_module_split_contract.py to phase-1 scope (collect_extract minimal + collect_stats + collect_io); removed phase-later collect_quant/_apply_rules/_fallback_extract requirements.
- Changes made: Added behavioral assertions for helper gaps highlighted in review: _safe_read_json_dict, _write_json, _write_df (csv path), and _get_sample_indices determinism/cache contract.
- Why non-vacuous: New assertions check parsed dict content, sorted JSON key ordering, exact write metadata/path and row content for CSV writes, and deterministic cached index arrays.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: FAIL (expected pre-fix). Key signal remains missing phase-1 helper files (collect_extract.py, collect_stats.py, collect_io.py).
- Interpretation: Updated tests now fail specifically on phase-1 module absence while defining concrete post-extraction behavior for moved helpers.

## 2026-03-01 - Review pass after test updates
- Goal: Re-assess updated phase-1 helper split tests for robustness and phase alignment.
- Changes made: Reviewed revised tests/test_collect_data_helper_module_split_contract.py and reran it.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: FAIL (expected pre-fix). Key signals: missing scripts/collect_extract.py, scripts/collect_stats.py, scripts/collect_io.py.
- Findings: Scope now matches phase 1 substantially better (no collect_quant or _apply_rules/_fallback_extract requirement).
- Residual risk: _get_sample_indices cache behavior assertion is weaker than intended cache contract because second call can pass even if indices are recomputed rather than loaded.
- Review follow-up: addressed residual loophole in _get_sample_indices cache test (determinism-only was insufficient).
- Changes made: strengthened test_collect_stats_get_sample_indices_is_deterministic_and_cached by overwriting cache file with an impossible-to-recompute sentinel (duplicate indices) before second call; second call must return sentinel to prove true cache read.
- Why non-vacuous: recompute-with-same-seed no longer passes; test now distinguishes cache-hit behavior from deterministic regeneration.

## 2026-03-01 - Phase 2 implementation (minimal extraction)
- Goal: Implement smallest helper extraction slice to satisfy phase-1 contracts while preserving existing collect_data behavior.
- Changes made:
  - Added scripts/collect_extract.py with _split_along_axis and _canonicalize_layout.
  - Added scripts/collect_stats.py with _get_sample_indices and _per_expert_weight_stats.
  - Added scripts/collect_io.py with _safe_read_json_dict, _write_json, _write_df, _iter_weight_files, and _iter_tensors_from_file.
  - Updated scripts/collect_data.py to load sibling helper modules by absolute path and re-export moved helpers for compatibility.
  - Updated tests/test_collect_data_helper_module_split_contract.py loader to reuse already-loaded modules when path/name match, preventing duplicate-module identity mismatches.
- Tradeoff: collect_data.py still contains legacy copies of moved helper definitions, but runtime names are now rebound to helper-module implementations to keep blast radius low in this phase.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: PASS (6 tests OK).
- Commands run: uv run python -m unittest tests.test_split_along_axis tests.test_canonicalize_layout tests.test_weight_stats tests.test_iter_tensors_from_file_npz_close tests.test_mlx_quant_sim_error tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_accepts_file_path tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_filters_extensions -v
- Result: PASS (18 tests OK).
- Commands run: uv run python -m unittest tests.test_optional_mlx.OptionalMlxIntegrationTests.test_collect_data_without_mlx_warns_and_writes_outputs -v
- Result: ERROR due wrong class selector (OptionalMlxIntegrationTests does not exist; suggested OptionalMlxPipelineTests).
- Commands run: uv run python -m unittest tests.test_optional_mlx.OptionalMlxPipelineTests.test_collect_data_without_mlx_warns_and_writes_outputs -v
- Result: PASS (1 test OK), providing a collect_data CLI smoke check after helper extraction wiring.

## 2026-03-01 - Post-implementation review
- Goal: Review phase-2 implementation for phase-1 helper extraction slice and validate regression risk.
- Scope reviewed: scripts/collect_data.py, scripts/collect_extract.py, scripts/collect_stats.py, scripts/collect_io.py, tests/test_collect_data_helper_module_split_contract.py.
- Findings: No functional regressions identified in reviewed scope; helper re-export wiring and module-loader identity behavior look correct for script execution and importlib-based tests.
- Residual risk: Temporary duplicate helper definitions remain in collect_data.py by design; future edits must keep helper modules as the source of truth to avoid drift.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: PASS (6 tests OK).
- Commands run: uv run python -m unittest tests.test_split_along_axis tests.test_canonicalize_layout tests.test_weight_stats tests.test_iter_tensors_from_file_npz_close tests.test_mlx_quant_sim_error tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_accepts_file_path tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_filters_extensions -v
- Result: PASS (18 tests OK).
- Commands run: uv run python -m unittest tests.test_optional_mlx.OptionalMlxPipelineTests.test_collect_data_without_mlx_warns_and_writes_outputs -v
- Result: PASS (1 test OK).
- Commands run: uv run python -m unittest -v
- Result: NO TESTS RAN (project uses unittest discover via make test).
- Commands run: uv run make test
- Result: PASS (80 tests OK).
- Changes made: Removed duplicate local implementations of moved phase-1 helpers from scripts/collect_data.py (io/stat/split/layout bodies), leaving helper-module re-exports as source of truth.
- Resulting shape: scripts/collect_data.py reduced from 1814 lines to 1536 lines after duplicate-removal cleanup.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v; uv run python -m unittest tests.test_split_along_axis tests.test_canonicalize_layout tests.test_weight_stats tests.test_iter_tensors_from_file_npz_close tests.test_mlx_quant_sim_error tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_accepts_file_path tests.test_proj_group_normalization.ProjInferenceUnitTests.test_iter_weight_files_filters_extensions -v
- Result: PASS for both commands.
- Commands run: uv run make test
- Result: PASS (80 tests OK). Note: initial sandbox attempt failed due uv cache permission; reran with escalation and succeeded.

## 2026-03-01 - Additional extraction slice (parsing/extraction helpers)
- Goal: Continue shrinking collect_data.py by moving parsing and extraction helper classes/functions into collect_extract.py while preserving public helper access from collect_data.
- Changes made:
  - Expanded scripts/collect_extract.py to include Rule, PackedSplitError, ExtractedBank, _compile_rules, _parse_int_from_regex, _is_shared_expert, _infer_proj, _suggest_proj, _record_proj_issue, _apply_rules, and _fallback_extract.
  - Removed the above local definitions from scripts/collect_data.py and re-exported them from _collect_extract_mod.
  - Expanded tests/test_collect_data_helper_module_split_contract.py expected collect_extract API and added identity checks for Rule, _infer_proj, and _apply_rules.
- Resulting shape: scripts/collect_data.py reduced further to 1174 lines.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract tests.test_proj_group_normalization tests.test_packed_split_strictness -v
- Result: PASS (27 tests OK).
- Commands run: uv run make test
- Result: PASS (80 tests OK).
- Commit: `f22e34c2b5269f5845ab357ef15f5f2307492f20` — Refactor collect_data into helper modules with compatibility exports

## 2026-03-01 - Next slice phase-1 tests (quant helper extraction)
- Goal: Add fail-first tests for extracting quant simulation helpers into a dedicated collect_quant module while preserving collect_data compatibility.
- Plan: Extend helper split contract tests to require collect_quant.py + quant API exports, verify collect_data/collect_quant quant column schema alignment, and preserve collect_data.mx monkeypatch behavior for _mlx_quant_sim.
- Changes made: Updated tests/test_collect_data_helper_module_split_contract.py with collect_quant module contract assertions and a quant-entrypoint monkeypatch behavior test.
- Why non-vacuous: tests require concrete module/symbol existence, schema alignment, and behavioral error-row propagation under a stub quantize failure path.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: FAIL (expected pre-fix for quant slice). Key signals:
  - FileNotFoundError: scripts/collect_quant.py
  - Assertion failure: expected helper module file missing scripts/collect_quant.py
- Interpretation: Tests are fail-first for the next extraction slice and already confirm existing collect_data.mx monkeypatch quant-entrypoint behavior is preserved pre-refactor.
## 2026-03-01 - Quant extraction slice implementation
- Goal: Move quant simulation helpers into scripts/collect_quant.py while preserving collect_data entrypoint behavior and existing monkeypatch patterns.
- Changes made:
  - Added scripts/collect_quant.py with QUANT_SIM_COLUMNS and _mlx_quant_sim.
  - Updated scripts/collect_data.py to load _collect_quant_mod and source QUANT_SIM_COLUMNS from helper module.
  - Replaced in-file quant implementation in collect_data.py with a wrapper that delegates to collect_quant._mlx_quant_sim using load_mlx=_load_mlx to preserve collect_data.mx monkeypatch behavior.
- Changes made (tests): Extended tests/test_collect_data_helper_module_split_contract.py for collect_quant module contract and quant-entrypoint behavior lock.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract -v
- Result: PASS (7 tests OK).
- Commands run: uv run python -m unittest tests.test_mlx_quant_sim_error tests.test_optional_mlx -v
- Result: PASS (5 tests OK).
- Commands run: uv run make test
- Result: PASS (81 tests OK).
## 2026-03-01 - Next slice phase-1 tests (main scan-loop decomposition)
- Goal: Add fail-first tests for extracting scan-loop helper logic out of collect_data main flow into a dedicated pipeline helper module.
- Plan: Require new scripts/collect_pipeline.py with explicit API (`record_example`, `process_one_bank`) and lock concrete behavior for dedupe/limit semantics and shared-expert row emission semantics.
- Changes made: Added tests/test_collect_pipeline_split_contract.py.
- Why non-vacuous: Tests assert deterministic row-level fields (`expert_id`, routed/shared flags, layer/block, rows/cols, metric propagation) and list semantics (dedupe+cap), not merely symbol existence.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract -v
- Result: FAIL (expected pre-fix).
- Key signals:
  - Assertion failure: missing scripts/collect_pipeline.py
  - FileNotFoundError importing scripts/collect_pipeline.py in behavior tests
- Interpretation: Fail-first signal is specific to the new main-loop helper module absence; once module exists, tests will enforce concrete behavior (record dedupe/limit and shared-expert row emission fields).

## 2026-03-01 - Review of collect_pipeline split contract tests
- Goal: Review new tests intended to guard extraction of _record_example and nested process_one bank row logic into collect_pipeline.py.
- Scope reviewed: tests/test_collect_pipeline_split_contract.py and corresponding current behavior in scripts/collect_data.py main() nested helpers.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract -v
- Result: FAIL (expected pre-implementation). Key signal: missing scripts/collect_pipeline.py.
- Findings: Tests provide good fail-first module-boundary signal and core shared-expert/no-quant row-shape guardrails.
- Residual risk: process_one_bank test does not currently assert source identifier columns (file/source_tensor/derived_tensor/proj/dtype) or layer_idx fallback semantics (layer_idx=None -> bank_obj.layer_base), so certain behavioral regressions could slip through extraction.
- Review follow-up: strengthened pipeline split contract tests per P2 findings.
- Changes made: in tests/test_collect_pipeline_split_contract.py, added assertions for source-identity fields (`file`, `source_tensor`, `derived_tensor`, `proj`, `dtype`) in process_one_bank output rows.
- Changes made: added test_process_one_bank_layer_fallback_uses_layer_base_then_unknown to constrain layer fallback semantics for layer_idx=None (use layer_base; if still None, use -1 with block4=None).
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract -v
- Result: FAIL (expected pre-fix). Key signals remain missing scripts/collect_pipeline.py (assertion + FileNotFoundError).
- Note: strengthened assertions are now in place and will activate once collect_pipeline.py exists.
## 2026-03-01 - Pipeline helper extraction slice implementation
- Goal: Extract nested main-loop helper logic (`_record_example` and nested `process_one`) from collect_data into a dedicated helper module while preserving output contracts.
- Changes made:
  - Added scripts/collect_pipeline.py with `record_example` and `process_one_bank`.
  - Updated scripts/collect_data.py to load `_collect_pipeline_mod` and alias `record_example` / `process_one_bank`.
  - Replaced calls to nested `_record_example` with `record_example`.
  - Removed nested `process_one` closure and replaced with calls to `process_one_bank` for both 3D and 4D bank paths.
  - `process_one_bank` preserves matrix row/quant row field contracts and layer fallback semantics (`layer_idx=None` -> `bank_obj.layer_base` -> `-1`).
- Changes made (tests):
  - Added tests/test_collect_pipeline_split_contract.py.
  - Strengthened it with source-identity field assertions and layer fallback branch coverage.
- Resulting shape: scripts/collect_data.py reduced to 974 lines.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract tests.test_collect_data_helper_module_split_contract -v
- Result: PASS (11 tests OK).
- Commands run: uv run python -m unittest tests.test_optional_mlx tests.test_proj_group_normalization tests.test_packed_split_strictness -v
- Result: PASS (25 tests OK).
- Commands run: uv run make test
- Result: PASS (85 tests OK).
- Review follow-up: added direct unit coverage for quant-row field mapping in pipeline helper.
- Changes made: tests/test_collect_pipeline_split_contract.py now includes test_process_one_bank_quant_rows_map_qdf_fields to assert qdf->quant_rows field propagation, expert_id mapping, layer/block propagation, and warn_log forwarding.
## 2026-03-01 - Additional pipeline decomposition (extracted-banks iterator)
- Goal: Continue reducing collect_data main-loop complexity by moving per-extracted-bank ndim/layer iteration into collect_pipeline helper module.
- Changes made:
  - Added collect_pipeline.process_extracted_banks.
  - Updated collect_data.py to alias process_extracted_banks and replace inline per-bank iteration block with helper call.
  - Added unit test test_process_extracted_banks_layer_progression_and_unsupported_warning in tests/test_collect_pipeline_split_contract.py.
- Behavioral constraints now covered: 4D layer progression (`layer_base + li`), 2D->3D path, and unsupported-ndim warning emission including derived tensor id.
- Resulting shape: scripts/collect_data.py reduced to 936 lines.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract tests.test_collect_data_helper_module_split_contract -v
- Result: PASS (13 tests OK).
- Commands run: uv run make test
- Result: PASS (87 tests OK).
- Commit: `da4ddc944dc575e6453f877dd7188f3f593a3ae3` — Extract quant and pipeline helpers from collect_data main loop
## 2026-03-01 - Next slice phase-1 tests (index/report synthesis extraction)
- Goal: Add fail-first tests for extracting index report and index summary synthesis out of collect_data main into a dedicated reporting helper module.
- Plan: Require scripts/collect_reporting.py with API (`build_index_report_data`, `build_index_summary`) and assert concrete set-diff semantics + index-used gating semantics.
- Changes made: Added tests/test_collect_reporting_split_contract.py.
- Why non-vacuous: Tests assert exact sorted list payload fields (`missing_shards`, `extra_tensors`, etc.) and count-zero gating when index is not used, which prevents superficial pass-through implementations.
- Commands run: uv run python -m unittest tests.test_collect_reporting_split_contract -v
- Result: FAIL (expected pre-fix). Key signals:
  - Assertion failure for missing scripts/collect_reporting.py
  - FileNotFoundError importing scripts/collect_reporting.py in behavior tests
- Interpretation: fail-first contract is specific to the new reporting helper module boundary; behavior assertions are ready for implementation phase.
## 2026-03-01 - Test re-review (higher-reasoning pass)
- Goal: Re-review current split-contract tests before reporting readiness and next implementation slice.
- Scope reviewed: tests/test_collect_data_helper_module_split_contract.py, tests/test_collect_pipeline_split_contract.py, tests/test_collect_reporting_split_contract.py, and current index/report logic in scripts/collect_data.py.
- Commands run: uv run python -m unittest tests.test_collect_data_helper_module_split_contract tests.test_collect_pipeline_split_contract tests.test_collect_reporting_split_contract -v
- Result: PARTIAL PASS / EXPECTED FAIL-FIRST.
  - `test_collect_data_helper_module_split_contract`: PASS.
  - `test_collect_pipeline_split_contract`: PASS.
  - `test_collect_reporting_split_contract`: FAIL/ERROR because scripts/collect_reporting.py does not exist yet.
- Key review finding: reporting summary contract currently constrains only the `index_used_for_scan=False` path; counts for `index_used_for_scan=True` are not yet pinned, so regressions in active index counting could slip through a refactor.
- Suggested follow-up test tightening: add one positive-path assertion set for build_index_summary with `index_used_for_scan=True` and non-empty inputs, verifying exact expected counts.
## 2026-03-01 - Added contract tightenings requested in re-review
- Goal: Close two residual P2 test gaps: (1) index_summary active-branch counts, (2) process_one_bank quant expert-id branch mapping.
- Changes made:
  - tests/test_collect_reporting_split_contract.py:
    - Added `test_build_index_summary_counts_inputs_when_index_used_for_scan` to pin active index counting semantics and metadata passthrough when `index_used_for_scan=True`.
  - tests/test_collect_pipeline_split_contract.py:
    - Added `test_process_one_bank_quant_rows_use_minus_one_for_shared_expert` to pin quant expert-id mapping for shared-expert banks.
    - Added `test_process_one_bank_quant_rows_use_expert_single_id_for_singleton_bank` to pin quant expert-id mapping for `expert_single_id` singleton path.
- Why non-vacuous:
  - New reporting test asserts exact count values derived from non-empty inputs, so a refactor cannot silently zero/count wrong while index scan is active.
  - New pipeline tests assert specific quant expert-id values on branch-specific fixtures, so generic passthrough of `expert_id_in_bank` is no longer sufficient.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract tests.test_collect_reporting_split_contract -v
- Result: PARTIAL PASS / EXPECTED FAIL-FIRST.
  - Pipeline module tests: PASS.
  - Reporting module tests: FAIL/ERROR due to missing scripts/collect_reporting.py (expected pre-implementation signal).
- Key failure lines:
  - `Expected helper module file is missing: .../scripts/collect_reporting.py`
  - `FileNotFoundError: .../scripts/collect_reporting.py`

## 2026-03-01 - Review of last-stage refactor tests
- Goal: Review new tests intended to guard final-stage decomposition of pipeline/reporting helpers.
- Scope reviewed: tests/test_collect_pipeline_split_contract.py and tests/test_collect_reporting_split_contract.py against current scripts/collect_data.py behavior.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract -v
- Result: PASS (8 tests OK).
- Commands run: uv run python -m unittest tests.test_collect_reporting_split_contract -v
- Result: FAIL (expected pre-implementation). Key signal: missing scripts/collect_reporting.py.
- Findings: Pipeline tests now strongly constrain row/quant mapping edge cases; reporting tests are useful fail-first coverage for pure set-diff and summary-count behavior.
- Residual risk: Reporting tests currently validate helper outputs but not collect_data wiring to prove the new helper functions are actually used for artifact/report emission.
## 2026-03-01 - Addressed additional review findings (reporting wiring + API shape checks)
- Goal: Close two new test-loophole findings:
  - ensure reporting split tests validate collect_data wiring/delegation (not helper logic alone)
  - strengthen API boundary checks from symbol-presence to callable/signature checks
- Changes made:
  - tests/test_collect_reporting_split_contract.py
    - Added callable/signature assertions for `build_index_report_data` and `build_index_summary` in module API test.
    - Added `test_collect_data_main_delegates_reporting_assembly_to_helpers` that runs `collect_data.main()` on a tiny fixture and monkeypatches `collect_data.build_index_report_data` / `collect_data.build_index_summary`; asserts both hooks are called and their sentinel outputs are used for `logs/index_report.json` and `logs/run_health.json[index_summary]`.
  - tests/test_collect_pipeline_split_contract.py
    - Upgraded API test for `record_example`, `process_one_bank`, and `process_extracted_banks` from `hasattr` checks to callable + parameter-name signature checks.
- Why non-vacuous:
  - Delegation test fails if `collect_data` keeps inline report/summary assembly and does not route through helper hooks.
  - Signature checks prevent non-callable placeholders or shape-drifted APIs from satisfying contract by name alone.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract tests.test_collect_reporting_split_contract -v
- Result: PARTIAL PASS / EXPECTED FAIL-FIRST.
  - Pipeline contract tests: PASS.
  - Reporting contract tests: FAIL/ERROR pre-implementation.
- Key fail-first signals:
  - `Expected helper module file is missing: .../scripts/collect_reporting.py`
  - `FileNotFoundError: .../scripts/collect_reporting.py`
  - `test_collect_data_main_delegates_reporting_assembly_to_helpers`: `AssertionError: 0 != 1` for helper-call count, showing collect_data has not delegated yet.
- Follow-up tweak: delegation hook stubs now accept `*args, **kwargs` to avoid over-constraining call style; intent remains delegation detection.
- Commands run (rerun): uv run python -m unittest tests.test_collect_pipeline_split_contract tests.test_collect_reporting_split_contract -v
- Result: unchanged; same fail-first signals (missing `scripts/collect_reporting.py` + delegation count assertion `0 != 1`).
## 2026-03-01 - Implementation: reporting helper extraction + collect_data delegation wiring
- Goal: Implement reporting decomposition so new fail-first contract tests pass, including collect_data wiring assertions.
- Changes made:
  - Added scripts/collect_reporting.py
    - `build_index_report_data(...)`: computes sorted index report payload (expected/scanned/missing/extra shard + tensor sets, extra on-disk safetensors, optional metadata).
    - `build_index_summary(...)`: computes `index_summary` map with active/parsed/used flags, index path/strict flag, gated count semantics, optional metadata.
  - Updated scripts/collect_data.py
    - Loads `_collect_reporting_mod` via `_load_local_helper_module`.
    - Re-exports `build_index_report_data` and `build_index_summary` for compatibility + monkeypatch-based contract tests.
    - Replaced inline index report assembly block with `build_index_report_data(...)` call.
    - Replaced inline `index_summary` dict assembly with `build_index_summary(...)` call.
    - Preserved warnings and artifact writes (`logs/index_report.json`, `logs/run_health.json`) while sourcing data from helper outputs.
- Why this fixes the tests:
  - `test_collect_reporting_module_exists_and_exports_expected_api` now resolves real helper module and API.
  - `test_collect_data_main_delegates_reporting_assembly_to_helpers` now sees helper calls and sentinel payload propagation into artifacts.
  - Existing index summary/count semantics remain intact via helper implementation mirroring prior behavior.
- Commands run:
  - `uv run python -m unittest tests.test_collect_reporting_split_contract tests.test_collect_pipeline_split_contract tests.test_collect_data_helper_module_split_contract -v`
  - Result: PASS (20 tests OK).
  - `uv run python -m unittest tests.test_index_found_semantics tests.test_auditability_artifacts_integration -v`
  - Result: PASS (5 tests OK).
  - `uv run make test` (initial attempt in sandbox)
  - Result: could not run due sandbox cache permission error (`Failed to initialize cache at /Users/noahlyons/.cache/uv ... Operation not permitted`).
  - `uv run make test` (escalated after approval)
  - Result: PASS (94 tests OK).
- Resulting size note:
  - scripts/collect_data.py is now 931 lines (down from prior 936 before this slice).

## 2026-03-01 - Refactor review (post reporting helper extraction)
- Goal: Review current refactor snapshot before next checkpoint commit.
- Scope reviewed: scripts/collect_data.py wiring changes, new scripts/collect_reporting.py helper, updated pipeline/reporting split-contract tests.
- Findings: No functional regressions identified in reviewed scope.
- Residual risk: New helper modules are currently untracked in working tree; checkpoint commit should include scripts/collect_reporting.py and updated tests together to preserve import/wiring contract.
- Commands run: uv run python -m unittest tests.test_collect_pipeline_split_contract -v
- Result: PASS (8 tests OK).
- Commands run: uv run python -m unittest tests.test_collect_reporting_split_contract -v
- Result: PASS (5 tests OK).
- Commands run: uv run python -m unittest tests.test_run_health_json_integration tests.test_safetensors_index_integration tests.test_index_found_semantics -v
- Result: PASS (15 tests OK).
- Commands run: uv run make test
- Result: PASS (94 tests OK).
