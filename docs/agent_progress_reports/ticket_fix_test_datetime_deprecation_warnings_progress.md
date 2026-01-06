Goal: remove datetime.utcnow() deprecation warning in init_run manifest creation.
Plan: swap to timezone-aware UTC timestamp while preserving Z suffix.
Changes made: updated created_at to use dt.datetime.now(dt.timezone.utc) with Z normalization in scripts/init_run.py.
Decisions/tradeoffs: used dt.timezone.utc for broader compatibility; kept Z suffix to avoid output format drift.
Assumptions: output timestamp format with trailing Z is part of current contract and should remain stable.
Open questions/ambiguities: none.
Risk of test loophole: low; change is isolated to timestamp formatting.
Commands run: rg -n "utcnow()" scripts (found usage).
Tests: not run (user requested quick fix without adding tests).
