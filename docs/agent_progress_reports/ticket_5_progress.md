# Ticket 5 Progress

Goal:
- Add deterministic unit tests for _per_expert_weight_stats

Plan:
- Inspect _per_expert_weight_stats behavior and config knobs
- Add deterministic unit test(s) with tiny banks and explicit expected stats
- Record commands and outcomes

Changes made:
- Added deterministic unit tests for _per_expert_weight_stats with explicit expected stats and group outlier checks in tests/test_weight_stats.py

Decisions / tradeoffs:
- Used a tiny 2-expert bank with negative values to assert abs/percentile semantics and a separate float64 case to lock float32 casting behavior
- Kept sampling deterministic by setting sample_per_matrix >= R*C

Assumptions:
- None

Commands run:
- None (not run in this environment)

Updates:
- Adjusted expected g2_p50_outlier calculation to use float32 interpolation to match runtime float32 rounding in _per_expert_weight_stats

Updates:
- Relaxed g2_p50_outlier tolerance to 3e-6 to absorb float32 percentile rounding drift

Updates:
- Relaxed g2_max_outlier tolerance to 5e-6 to match float32 rounding
