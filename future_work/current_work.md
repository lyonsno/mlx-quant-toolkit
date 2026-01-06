Remaining Identified Issues

- `scripts/collect_data.py`: `proj_group` alias normalization still treats regex captures like `w1` as a distinct proj name; results can fragment aggregates by `proj`.
- `scripts/collect_data.py`: `_iter_weight_files` assumes `model_path` is a directory; a single-file checkpoint path results in a 0-file scan.

Planned Changes

- Normalize `proj_group` captures through the same alias map used for inference (e.g., map `w1` -> `gate_proj`).
- Optionally support `model_path` pointing directly to a single `.safetensors`/`.npz` file.

Open Questions

- Should `model_path` accept a file path as first-class input, or remain directory-only?
- Do you want `proj_group` normalization to be strict (only allow aliases) or permissive (allow raw names when unrecognized)?

Optional Improvements

- Strengthen the `tests/test_optional_mlx.py` coverage by validating delta math in
  `scripts/build_tables.py`. The current test forces `delta_pairs` so
  `B_quant_deltas.csv` is always emitted, but it uses scheme names that are not
  present in the quant output, so the delta columns remain `None`. A more robust
  check would configure `quant_schemes` with known names and provide a tiny,
  deterministic `quant_sim` dataset (or run a minimal collect step with a stub
  quant output) so both schemes appear in `quant_sim`. Then the test can assert
  that `delta_w_rel_fro` and `delta_w_rel_max` equal the expected differences.
  This ties the test directly to the README-documented behavior that
  `delta_pairs` compares two schemes within the `B_*` tables.
