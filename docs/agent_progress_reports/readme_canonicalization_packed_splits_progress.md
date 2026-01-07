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

---

## Follow-up — progress log policy clarification

## Goal
Clarify the repo-level progress-log policy so unrelated work doesn’t get appended to an arbitrary open file.

## Changes made
- Updated `agents.md`
  - Default behavior: create a new `docs/agent_progress_reports/<ticket_slug>_progress.md` per new ticket/issue.
  - Continuation allowed only when a specific prior log is referenced, or when the agent already created a log
    earlier in the current session/thread.
  - Added an explicit exception for user-requested migrations/cleanup (e.g., moving a mis-filed entry).

## Commands run
- None (docs/policy-only changes).

## commit
[main 857740a] docs(readme): explain canonicalization and packed splits
[main 39ebc8b] docs(agents): clarify progress log creation policy
