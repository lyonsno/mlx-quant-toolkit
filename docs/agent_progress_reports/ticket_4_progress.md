# Ticket 4 Progress

## Goal
- Add unit tests for `_canonicalize_layout` to lock down axis reorder behavior.

## Plan
- Add unit tests for 2D identity, 3D expert/rows/cols reorder, 4D layer/expert/rows/cols reorder, and invalid layout axes.

## Changes made
- (pending)

## Decisions / tradeoffs
- (pending)

## Assumptions
- (pending)

## Commands run
- (not run yet)

## Update 1

## Changes made
- Added unit tests for `_canonicalize_layout` covering 2D identity, 3D reorder, 4D reorder, and invalid layout axes in `tests/test_canonicalize_layout.py`.

## Decisions / tradeoffs
- Used full-array equality checks to lock down axis ordering without relying on print output.

## Assumptions
- None.

## Commands run
- Not run (tests only).

## Update 2

## Commands run
- `uv run python -m unittest tests.test_canonicalize_layout` (failed: uv cache access denied in sandbox).
- `python -m unittest tests.test_canonicalize_layout` (failed: `ModuleNotFoundError: No module named 'numpy'`).
