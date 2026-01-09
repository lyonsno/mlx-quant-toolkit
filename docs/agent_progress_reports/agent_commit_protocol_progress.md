# Ticket: agent_commit_protocol - Progress

- Goal: update `AGENTS.md` to define a consistent commit protocol when the user explicitly asks the agent to commit.
- Plan: add a short “commit protocol (when requested)” section describing staging scope, commit message requirements, progress-log exclusion, hash propagation, and the follow-up log-only commit; keep the change minimal and unambiguous.
- Changes made: none yet.
- Decisions / tradeoffs:
  - Define “all work” in git terms (“all current changes vs HEAD”) rather than tracking “files touched”.
  - Avoid infinite regress by explicitly not logging the follow-up “progress log update” commit hash.
  - Keep multi-commit support opt-in (agent must ask before splitting into multiple substantive commits).
- Assumptions: none yet.
- Open questions / ambiguities: none yet.
- Risk of test loophole: not applicable (process/doc-only change).
- Commands run: none.

- Changes made:
  - Updated `AGENTS.md` to add a “Commit protocol (when requested)” section covering commit-message preview, staging scope (“all current changes vs HEAD”), declarative commit message requirements, progress-log exclusion, and the follow-up log-only commit that records the substantive hash in the progress report without infinite regress.
- Decisions / tradeoffs:
  - Kept multi-commit behavior opt-in (agent must ask) to avoid noisy histories and repetitive log-update commits.
  - Explicitly documented sandbox/permission escalation for git operations to avoid index lock failures blocking progress.
