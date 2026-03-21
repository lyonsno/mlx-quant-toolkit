# Quant Metric Direct-Plot Scope (2026-03-20)

## Status
- Current local slice is uncommitted as of 2026-03-20.
- Intent is reviewer-facing and narrower than a full quant reporting expansion.

## Intended Contract
- Add two new scalar `quant_sim` metrics:
  - `w_rel_spectral`
  - `w_gram_cos_drift_sampled_rms`
- Keep these metrics first-class in:
  - raw `data/quant_sim.{csv,parquet}`
  - direct plot consumers in `build_plots.py`
- Do not extend `build_tables.py` or `B_quant_*` / `B_quant_deltas.*` in this slice.
  - If a reviewer expects table aggregation, that is a separate follow-up rather than an accidental omission.

## Metric Semantics
- `w_rel_spectral`
  - relative operator-norm distortion: `||W - W_hat||_2 / (||W||_2 + eps)`
- `w_gram_cos_drift_sampled_rms`
  - sampled, normalized off-diagonal cosine-Gram drift
  - computed separately for row space and column space
  - public scalar is `max(row_rms, col_rms)`
- The `_rms` suffix is intentional.
  - The earlier `_max` name was misleading because the implementation is an RMS/Frobenius-style summary over sampled off-diagonal cosine drift, not a worst-pair drift.

## Runtime / Implementation Notes
- Sampling is intentionally bounded by `stats.quant_gram_sample_k` when present.
- Spectral estimation uses exact `||.||_2` only on small matrices and a power-iteration estimate otherwise.
- The current local implementation avoids one avoidable whole-bank float32 copy by computing the new metrics from per-expert float32 slices after dequantization.

## Reviewer Notes
- The most relevant code paths are:
  - `scripts/collect_quant.py`
  - `scripts/collect_pipeline.py`
  - `tests/test_collect_quant_metric_contract.py`
  - `tests/test_optional_mlx.py`
  - `tests/test_build_plots_contract.py`
- The strongest contract test is the RMS-vs-pairwise-max distinction in `tests/test_collect_quant_metric_contract.py`.
- Plot coverage is intentionally consumer-side.
  - It proves direct quant plotting stays metric-generic once the raw `quant_sim` column exists.
  - It is not meant to replace the collect-path success test.

## Explicit Non-Goals For This Slice
- No `build_tables.py` aggregation for the new metrics yet.
- No template/doc exposure of `stats.quant_gram_sample_k` or `stats.quant_spectral_power_iters` yet.
- No per-channel raw artifact yet.
