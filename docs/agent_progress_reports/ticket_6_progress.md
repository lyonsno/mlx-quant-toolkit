Goal
- Add acceptance test that exercises unmatched_tensors emission when an expertish tensor does not match any proj rule.

Plan
- Extend the acceptance test harness to allow a custom tensor key.
- Add a new test that runs collect_data, then asserts unmatched_tensors.csv has the expected row and matrix_stats.csv is empty.

Changes made
- (started) Created this progress log.

Decisions / tradeoffs
- Will reuse existing acceptance helpers in tests/test_optional_mlx.py to minimize new scaffolding.

Assumptions
- None yet.

Commands run
- None.

Changes made
- Updated tests/test_optional_mlx.py to allow a custom tensor key/array and shallow-merge config overrides.
- Added acceptance test asserting unmatched_tensors.csv contents and empty matrix_stats.csv for an unmatched expertish tensor.

Decisions / tradeoffs
- Used a shallow merge for cfg_overrides to avoid rewriting full scan/debug dicts in tests.

Commands run
- None.
