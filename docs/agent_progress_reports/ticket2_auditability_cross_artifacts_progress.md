# Ticket 2 - Auditability cross-artifact invariants
- Goal: Strengthen auditability tests with required-key subsets, cross-artifact invariants, and index parse-error coverage.
- Plan: Add helper assertions in existing integration tests, add index parse-error scenario, run targeted tests or `make test`.
- Changes made: (pending)
- Decisions/tradeoffs: (pending)
- Assumptions: (pending)
- Open questions/ambiguities: (pending)
- Risk of test loophole: (pending)
- Commands run: (pending)
- Update: created this progress report file.
- Update: added required-key subset helpers, CSV row counting, and cross-artifact invariant assertions in `tests/test_auditability_artifacts_integration.py` and `tests/test_run_health_json_integration.py`.
- Update: added index parse-error integration scenario to `tests/test_auditability_artifacts_integration.py`.
- Commands: `uv run make test` failed in sandbox (uv cache permission); reran with escalated permissions and all tests passed.
- Commands: `python -m py_compile tests/test_auditability_artifacts_integration.py tests/test_run_health_json_integration.py`.
- Decisions: enforce CSV row count checks only when manifest format is CSV to avoid parquet dependency.
- Assumptions: `matrix_stats_rows` equals 2 for the indexed run (rule + fallback extraction).
- Risk of test loophole: parquet outputs are not row-count-validated when format is not CSV.

- Commit: 25b7e76d4a56b82fa64e3efeaceeec0ed49f9151 Strengthen auditability cross-artifact tests
