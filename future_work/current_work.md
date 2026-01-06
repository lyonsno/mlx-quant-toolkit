Remaining Identified Issues

- `scripts/collect_data.py`: `packed_split.projs` values are not canonicalized through `parsing.proj_aliases` (e.g., ["w1","w2"] stays raw), which can silently fragment aggregates by `proj`.
- `scripts/collect_data.py`: when `model_path` is a *file* and `scan.use_safetensors_index_json=true`, index discovery in the parent directory can expand the scan beyond that single file; this behavior is currently implicit and should be an explicit contract (and logged).
- `scripts/collect_data.py`: fallback extraction is intentionally heuristic, but runs provide little visibility into how often it was used (risk: “plausible but wrong” axis assumptions). Add run-level visibility.
- `scripts/collect_data.py`: `scan.strict_index` only enforces shard presence after a valid index parse; decide what “strict” means when the index is missing or invalid (fail vs warn + fallback).

Planned Changes

- Canonicalize `packed_split.projs` via the same alias inference path used elsewhere:
  - map known aliases -> canonical (`w1` -> `gate_proj`, etc.)
  - when unknown: either warn + keep raw, or treat as strict failure (policy decision below).
- Make file `model_path` + index behavior explicit:
  - emit an explicit log line when a file path results in scanning multiple indexed shards
  - add an integration test that locks down the chosen contract.
- Add visibility for fallback usage:
  - minimally: a single warning / summary count (“fallback_extract used for N tensors”)
  - optionally: a column like `extraction_method = rule|fallback` in `matrix_stats` / `quant_sim`.
- Clarify and enforce `strict_index` semantics:
  - decide whether strict requires (a) index exists + parses, or (b) only enforces shard presence *if* an index is found
  - encode the decision in behavior + tests.

Open Questions

- File `model_path` with index present: should default behavior mean “scan only this file” or “treat file as an anchor and scan all index shards”? If the latter, what’s the minimal logging we require so it can’t be missed?
- For `packed_split.projs` that don’t map to a canonical proj: should we (a) keep raw with a warning (permissive), or (b) drop/fail under strict mode to prevent fragmentation?

Recently Resolved (doc drift cleanup)
- Add context manager around opening .npz files
- Add `logs/run_health.json` summarizing: files scanned, tensors observed, extracted-by-rule vs fallback counts, unmatched count, and (if index-active) missing/extra shard/tensor counts.
- `proj_group` captures are canonicalized via `parsing.proj_aliases` (tests: `tests/test_proj_group_normalization.py`).
- `model_path` supports single-file checkpoints via `_iter_weight_files` (tests: `tests/test_proj_group_normalization.py`).
- Delta math in `build_tables.py` is covered with deterministic values (tests: `tests/test_optional_mlx.py`).

Optional Improvements
- write a minimal “started” run_health early (status "running", start_time) and then overwrite it at the end with "success" + outputs_written. That way even crashes leave something behind.
- Emit a warning when `delta_pairs` references schemes not present in `quant_sim` (helps catch typos early).