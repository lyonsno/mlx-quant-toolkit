Goal: Add tests for strict index requirement and file model_path index ignore semantics for PR1.
Plan:
- Add integration tests covering strict_index missing index failure and file model_path no index expansion + scan_plan flag.
- Run targeted unittest module to capture expected failing output pre-fix.
Notes:
- Assumption: strict_index means index must be active when use_index=true, even before scan starts.
Changes made:
- Added new tests in tests/test_safetensors_index_integration.py for strict_index missing index failure and file model_path ignoring index expansion (poison npz).
- Extended test helper to accept scan extensions so npz-only scans can be exercised.
Notes:
- Poison pill is an invalid .npz file referenced by the index; test expects success so it fails pre-fix when index expansion occurs.
Open questions:
- None yet; strict_index enforced as active-index requirement via test expectation.
Commands run:
- (pending)
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration (failed; strict_index missing index returned 0, file model_path scanned poison npz)
Changes made:
- Added strict index invalid-index failure test, file model_path strict_index requires active index test, and file model_path strict_index ignores missing index shards test in tests/test_safetensors_index_integration.py.
Decisions / tradeoffs:
- Encoded strict_index as requiring an active index even when model_path is a file (missing index -> failure).
- Encoded strict_index as not enforcing missing-index-shard completeness when model_path is a file (anchor scan stays strict about index validity only).
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration (failed; missing/invalid strict_index returned 0, file model_path strict missing index returned 0, file model_path missing shard strict failed, file model_path index still expanded to poison)
Changes made:
- Added test in tests/test_safetensors_index_integration.py to assert strict_index=true with use_index=false fails loudly (expects non-zero + message).
Assumptions:
- strict_index with use_index=false should be a hard error, not just a warning.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration (failed; strict_index missing/invalid/use_index=false returned 0; file model_path strict missing index returned 0; file model_path missing shard strict failed; file model_path index still expanded to poison)
Changes made:
- Implemented strict_index enforcement and file model_path index ignore logic in scripts/collect_data.py (new scan_plan flag + warning message).
- Updated README.md to document strict_index semantics and file model_path index handling; removed resolved pending decisions.
- Updated future_work/current_work.md to mark strict_index + file model_path index behavior as resolved.
Decisions / tradeoffs:
- Index reports and index_summary counts are emitted only when the index is used for scanning (file model_path ignores index for scan).
- strict_index + use_index=false is a hard error.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration (passed)
Changes made:
- Added run_context index parsed/active assertions to file model_path integration test in tests/test_safetensors_index_integration.py.
Assumptions:
- run_context.index.active should reflect index usage for scan, not parse success.
Changes made:
- Added index_parsed and index_used_for_scan fields to run_context and run_health outputs in scripts/collect_data.py, keeping index.active as used-for-scan.
- Documented parsed/active/used_for_scan semantics in README.md and auditability section.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration (passed)
Changes made:
- Swapped file-model-path poison-pill fixtures in tests/test_safetensors_index_integration.py from .npz to .safetensors (shard_ok/poison/missing) and removed .npz-only extensions override.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration -v (passed)
Changes made:
- Added run_health index_summary parsed/active/used_for_scan assertions to the file-model-path test in tests/test_safetensors_index_integration.py.
Notes:
- Behavior already matches this requirement, so the new assertions are expected to pass pre-fix; they lock in the contract.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration -v (passed)
Changes made:
- Added run_context and run_health index parsed/used_for_scan/active assertions to the index-active integration test in tests/test_safetensors_index_integration.py.
Notes:
- Existing behavior already provides these fields, so the new assertions are expected to pass; they lock in the contract.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration -v (passed)
Changes made:
- Added scan_plan.scan_mode assertions to index-active and file-model-path integration tests in tests/test_safetensors_index_integration.py.
Notes:
- These assertions are expected to pass with current behavior; they tighten the contract.
Commands run:
- uv run python -m unittest tests.test_safetensors_index_integration -v (passed)
Changes made:
- Updated README.md strict_index description, anchored-file index note, and troubleshooting entry to reflect new semantics and warning locations.
Changes made:
- Moved fallback-visibility item out of Remaining/Planned and into Recently Resolved in future_work/current_work.md.
- Fixed README.md typo in index field definitions.
Commit: 6845b8a Harden index semantics and logging
