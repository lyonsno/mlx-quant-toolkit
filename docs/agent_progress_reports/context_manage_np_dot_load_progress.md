# Ticket: context_manage_np_dot_load — progress

## Goal
- Prevent file-descriptor leaks when `scripts/collect_data.py::_iter_tensors_from_file()` loads many `.npz` files via `np.load(...)` by ensuring the returned object is closed.

## Plan (TDD protocol)
- Add a unit test that fails on current behavior (no close).
- After confirmation, implement the minimal fix (likely `with np.load(...) as data:` or explicit `try/finally: data.close()`).

## Changes made (Phase 1 — tests only)
- Added `tests/test_iter_tensors_from_file_npz_close.py`
  - Asserts that `.npz` iteration closes the object returned by `np.load`.
  - Also asserts it closes even if an exception occurs mid-iteration (this is the “non-happy path” leak case).

## Why the test is non-vacuous
- The test uses a fake `np.load` return object with an explicit `close_called` flag; this flag is only set if `close()` is actually invoked (not by “it didn’t crash” behavior).
- Current `collect_data._iter_tensors_from_file()` never calls `close()`, so the assertions fail for a specific reason.

## Commands run
- `uv run python -m unittest tests.test_iter_tensors_from_file_npz_close`
  - Failed due to sandbox/cache permissions: `failed to open file ... ~/.cache/uv/... Operation not permitted`
- `UV_CACHE_DIR="$(pwd)/.uv_cache" uv run python -m unittest tests.test_iter_tensors_from_file_npz_close`
  - `uv` panicked (Tokio executor / system-configuration dynamic_store)
- `./.venv/bin/python -m unittest tests.test_iter_tensors_from_file_npz_close`
  - Fails as intended:
    - `AssertionError: ... Expected _iter_tensors_from_file to close the np.load() result for .npz inputs`
    - `AssertionError: ... Expected _iter_tensors_from_file to close the np.load() result even when iteration errors`

## Assumptions / notes
- I’m assuming `.npz` closure semantics should match `.safetensors` (already using a context manager).
- I’m not changing behavior yet (tests-only phase).

## Phase 2 — implementation
- Updated `scripts/collect_data.py::_iter_tensors_from_file()` to use `with np.load(..., allow_pickle=False) as data:` for `.npz` files so the underlying file handle is reliably closed:
  - on normal exhaustion,
  - on exceptions mid-iteration,
  - and when the generator is closed early.

## Commands run (post-fix)
- `./.venv/bin/python -m unittest tests.test_iter_tensors_from_file_npz_close` (PASS)
- `./.venv/bin/python -m unittest tests.test_optional_mlx` (PASS)
