Goal
- Add an acceptance test covering successful packed_split (happy path) with strict mode.

Plan
- Extend the packed split test helper to accept a valid split tensor.
- Add a new acceptance test that runs collect_data and asserts derived projs and shapes.
- Record warnings expectations (no packed_split failure).

Changes made
- (pending)

Decisions / tradeoffs
- (pending)

Assumptions
- (pending)

Commands run
- (pending)

Changes made
- Updated tests/test_packed_split_strictness.py to allow passing a valid packed tensor and added a happy-path acceptance test.

Decisions / tradeoffs
- Reused the existing collect_data harness to avoid duplicating CLI setup logic.

Assumptions
- The matrix_stats CSV includes proj, rows, and cols for each derived bank.

Commands run
- (not run; test-only change)
