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
