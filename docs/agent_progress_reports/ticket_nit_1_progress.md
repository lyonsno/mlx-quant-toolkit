Goal
- Strengthen packed-split non-strict acceptance test to assert fallback emits real stats rows.

Plan
- Update the non-strict packed split test to parse matrix_stats.csv and assert rows/proj/shape.

Changes made
- Added matrix_stats.csv row/proj/shape assertions in tests/test_packed_split_strictness.py.

Decisions / tradeoffs
- Chose exact row count and shape assertions to prove fallback emitted stats for the gate_proj bank.

Assumptions
- Fallback extraction for a 3D expert bank treats the array as (E,R,C).

Commands run
- None.
