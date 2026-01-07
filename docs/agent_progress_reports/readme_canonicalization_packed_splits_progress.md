# Ticket: readme_canonicalization_packed_splits — Progress

## Goal
Add a README section that explains (user-facing + developer-facing) what “canonicalization” means in this repo,
what “packed splits” are, why they add complexity, and how to reason about/validate them.

## Changes made
- Updated `README.md`
  - Added `### Canonicalization and packed splits (mental model)` describing canonical axis order, provenance
    (`source_tensor`/`derived_tensor`), proj alias resolution, packed-split slicing semantics, and practical
    sanity checks using `data/matrix_stats.*` and `logs/warnings.*`.

## Commands run
- None (documentation-only change).

## Notes
- During the README edit, I only used read-only inspection commands (e.g. `rg`, `sed`) and did not run tests.
