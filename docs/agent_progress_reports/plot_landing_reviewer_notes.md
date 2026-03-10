# Plot Landing Reviewer Notes

Date: 2026-03-09  
Scope: Consolidated local-agent review findings into an actionable landing sequence.

## Pre-Plot Landing (must do before calling plotting ready)

1. Fix schema-less empty input handling in `build_tables`.
- Why: real run reproduced `KeyError: 'layer'` when `matrix_stats` had zero rows and no schema columns.
- Evidence:
  - Crash site: `scripts/build_tables.py:276` (`ms["layer"]` access when `layer` missing).
  - Fail-first test: `tests/test_build_tables_contract.py:test_build_tables_handles_schema_less_empty_loader_frames`.
- Required outcome:
  - `build_tables` succeeds when `_read_df` returns `DataFrame()` for both inputs.
  - Emits empty-but-valid A/B tables (stable headers, zero rows).
  - Emits valid `logs/tables_write_manifest.json`.

2. Tighten the new schema-less empty contract test.
- Why: current test proves no crash + headers, but misses audit-path and read-target precision.
- Evidence:
  - Current assertions only check `len(read_calls) == 2` and table headers/rows.
  - No manifest assertions in `test_build_tables_handles_schema_less_empty_loader_frames`.
- Required outcome:
  - Assert read paths are exactly the expected matrix/quant inputs for that run.
  - Assert per-artifact manifest fields for zero-row outputs (`rows`, `format`, `fallback`, `error`, `path`).

3. Add a run-smoke guard for "effectively empty collect" before plotting.
- Why: the reproduced run scanned 44 shards but wrote `matrix_stats rows: 0`, `quant_sim rows: 0`, `unmatched rows: 0`; plotting is downstream and should not hide this upstream failure mode.
- Required outcome:
  - At minimum, document and enforce a pre-plot smoke check in runbook/PR checklist:
    - `data/matrix_stats.*` row count > 0 for expected MoE runs.
    - If zero, block plotting and investigate extraction/rule config first.

## Immediately After Landing (next hardening batch)

1. Expand numeric contract checks for `build_tables` aggregation outputs.
- Gap: several tests assert schema/rows/order but do not pin enough aggregate values.
- Add:
  - Deterministic numeric assertions for `median/mean/std/p90/p99/p01/min/max` on known fixtures.
  - At least one non-trivial percentile fixture (more than two points per group).

2. Strengthen quant error-row coverage across all aggregate metric columns.
- Gap: current checks focus on a subset of `*_median` fields.
- Add:
  - Iterate expected B metric aggregate columns and assert empty/NA behavior consistently for error rows.

3. Add multi-delta-pair coverage.
- Gap: current delta tests exercise one pair at a time.
- Add:
  - Two-pair config and assertions that both named delta outputs are present and correct.

4. Add explicit compression success-path contracts.
- Gap: fallback/error path is well-covered; successful compression behavior is not.
- Add:
  - Valid compression settings with assertions on written format/metadata/readability.

5. Add explicit malformed collect-manifest contracts for `build_tables` input resolution.
- Gap: absent/legacy behavior is exercised implicitly; malformed `logs/write_manifest.json` shape/JSON should be pinned explicitly.
- Add:
  - Invalid JSON and wrong-schema cases with deterministic fallback/behavior assertions.

## Review-process note (local rig reliability)

Observed divergence between local search tools (`grep`, `rg`, editor "grep") can produce false "not covered" conclusions.  
For reviewer runs, require a canary query (`block4`, `B_quant_deltas`, or a known test name) before making any `Not Covered` claim.

