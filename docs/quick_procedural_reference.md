# Custom MLX Quant Tools - Procedural Reference

This guide covers **how to use** the pipeline: configuration, execution workflows, testing, troubleshooting, and operational conventions.

---

## Configuration Summary (`analysis_config.json`)

Top-level keys:

| Section | Purpose | Important Sub-keys |
|---------|---------|-------------------|
| `model_path` | Path to model dir or single shard file | - |
| `scan` | File discovery and filtering | `extensions`, `max_files`, `experts_only`, `include_shared_expert`, `inventory_all_tensors`, `use_safetensors_index_json`, `strict_index` |
| `parsing` | Tensor name parsing rules | `layer_regex`, `expert_regex`, `proj_aliases`, `shared_expert_keywords`, `proj_group_strict`, `strict_packed_split` |
| `extract_rules` | Array of rule objects (match, ndim, layout, groups, packed_split) | See Core Concepts in navigation_reference.md |
| `stats` | Sampling and metrics | `sample_per_matrix` (set ≥ rows*cols for deterministic percentiles), `metrics` (mean_abs, max_abs, p99_abs, outlier_ratios) |
| `mlx` + `quant_schemes` | MLX simulation config (optional) | `enabled`, `schemes` array (name, mode, bits, group_size) |
| `delta_pairs` | Scheme comparison pairs for `build_tables.py` | `[{"name": "...", "a": "...", "b": "..."}]` |
| `output` | Format and compression | `format` ("parquet" or "csv"), `compression` (e.g., "snappy") |
| `metadata` | Config.json parsing | `enabled`, `config_path`, `mode` (reserved) |
| `debug` | Diagnostic artifacts | `dump_unmatched_tensors`, `print_progress_every_files` |

---

## Output Artifacts

### Data (`data/`)
- `tensor_inventory.*`: one row per observed tensor (`file`, `tensor_name`, `dtype`, `shape`, `ndim`, `nbytes`, plus `in_index`/`index_shard` if index active)
- `matrix_stats.*`: one row per extracted expert matrix (`file`, `source_tensor`, `derived_tensor`, `layer`, `block4`, `proj`, `expert_id`, `is_routed_expert`, `is_shared_expert`, `rows`, `cols`, `dtype`, `mean_abs`, `max_abs`, `p99_abs`, `g32_*` outlier ratios)
- `quant_sim.*`: one row per (expert, scheme) (`file`, `source_tensor`, `derived_tensor`, `layer`, `block4`, `proj`, `expert_id`, `is_shared_expert`, `scheme`, `mode`, `bits`, `group_size`, `w_rel_fro`, `w_rel_max`, `w_rel_spectral`, `w_gram_cos_drift_sampled_rms`, `scale_*`, `bias_*`, `error`)
  - Current March 2026 scope note: the new quant metrics are present in raw `quant_sim` and direct `build_plots.py` quant views, but are not yet aggregated into `B_quant_*` / `B_quant_deltas.*`.
- `unmatched_tensors.*` (optional): tensors that matched expertish heuristics but failed extraction

### Tables (`tables/`)
- `A_weight_*.*`: aggregated weight stats by layer/block/global
- `B_quant_*.*`: aggregated quant simulation stats
- `B_quant_deltas.*` (optional): deltas between scheme pairs

### Logs (`logs/`)
- `run_health.json`: scan/extraction/output summary, effective config, example tensor names
- `run_context.json`: scan plan, index status, CLI overrides
- `write_manifest.json`: collect-stage artifact writes from `collect_data.py` (paths, row counts, format, fallbacks)
- `tables_write_manifest.json`: tables-stage artifact writes from `build_tables.py` (paths, row counts, format, fallbacks)
- `warnings.*` (if any warnings): includes proj canonicalization summaries and index warnings
- `proj_canonicalization_report.*` (if proj issues): detailed unresolved projection mappings
- `index_report.json` (if index active): missing/extra shards and tensors
- `model_config.raw.json`, `model_shape_budget.json` (if metadata enabled)

**Contract surfaces:** Files tagged with `CONTRACT SURFACE:` in code are stable interfaces.

---

## Testing Strategy

### Unit Tests (pure helpers)
Targeted tests for:
- Array splitting (`test_split_along_axis.py`)
- Canonicalization (`test_canonicalize_layout.py`)
- Proj group normalization (`test_proj_group_normalization.py`)
- Weight stats computation (`test_weight_stats.py`)
- Packed split strictness (`test_packed_split_strictness.py`)
- Optional MLX handling (`test_optional_mlx.py`)

Pattern: import script via `importlib.util.spec_from_file_location`, monkeypatch globals (e.g., `mx` stub).

### Acceptance Tests (integration)
- End-to-end pipeline on tiny `.npz` fixtures
- Verify artifacts exist and have expected rows/columns
- Test index behaviors (`test_safetensors_index_integration.py`, `test_index_found_semantics.py`)
- Test run health logging (`test_run_health_json_integration.py`)
- Test metadata parsing (`test_metadata_integration.py`)

**Test design guardrails:**
- No float nondeterminism: use small deterministic arrays or exact integers
- For sampling: set `sample_per_matrix ≥ rows*cols` to compute exact percentiles
- Assert concrete invariants (row counts, columns, numeric values), not just file existence
- Include "poison pill" fixtures for filtering/selection tests
- Ensure each report category has non-empty examples when feasible

---

## Important Conventions

1. **Progress notes**: When working on a ticket, maintain `docs/agent_progress_reports/<slug>_progress.md` (append-only).
2. **Contract surfaces**: Look for `CONTRACT SURFACE:` comments in `scripts/` to identify stable outputs.
3. **Error handling**: For continue-on-error paths, record useful context in logs (not just generic messages).
4. **Minimal diffs**: Avoid drive-by refactors; keep changes scoped to ticket.
5. **MLX optional**: Pipeline must run without MLX; tests should stub MLX when needed.
6. **No new test frameworks**: Stick to `unittest`.

---

## Common Workflows

### Quick start
```bash
# 1. Initialize a run
uv run python scripts/init_run.py --root ./runs --model-id test-model --run-name test-run --model-path /path/to/model

# 2. Edit runs/test-model/test-run/analysis_config.json (customize extract_rules, etc.)

# 3. Run data collection
uv run python scripts/collect_data.py --run-dir ./runs/test-model/test-run

# 4. Build tables
uv run python scripts/build_tables.py --run-dir ./runs/test-model/test-run

# 5. Optional: build baseline plots
uv run python scripts/build_plots.py --run-dir ./runs/test-model/test-run

# 6. Inspect outputs in runs/test-model/test-run/{data,tables,plots,logs}
```

### Running tests
```bash
make test           # all tests
make verbose-test   # verbose
uv run python -m unittest tests.test_packed_split_strictness  # specific module
```

### Debugging extraction
- Check `logs/run_health.json` for rule vs fallback counts
- If `unmatched_tensors.*` exists, examine `reason` column
- Check `logs/proj_canonicalization_report.*` for projection mapping issues
- Verify `data/matrix_stats.*` has expected `proj` and `derived_tensor` values

---

## Troubleshooting

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| Fewer extracted rows than expected | Extraction rules not matching | `logs/run_health.json` (fallback counts), `data/unmatched_tensors.*` |
| Wrong `proj` labels | `proj_aliases` missing or `proj_group` index wrong | `logs/proj_canonicalization_report.*`, warnings |
| Packed split produces wrong shapes | `splits` don't sum to canonical rows/cols | Inspect `matrix_stats` rows vs expected sizes |
| Parquet write failures | Missing pyarrow or compression issue | `logs/write_manifest.json` (collect stage) and `logs/tables_write_manifest.json` (tables stage) for fallbacks |
| Index mode not scanning all shards | `model_path` is a file, not a directory | `logs/run_context.json` → `index_used_for_scan` |
| MLX simulation errors | MLX not installed or scheme invalid | `quant_sim` rows with `error` column populated |
| Deterministic percentile mismatch | `sample_per_matrix` too small | Set `sample_per_matrix ≥ rows*cols` |

---

## Notes on Strictness Flags

| Flag | When true | When false |
|------|-----------|------------|
| `parsing.proj_group_strict` | Unmapped proj_group → drop + warn | Unmapped → keep raw + warn |
| `parsing.strict_packed_split` | Packed split mismatch → error | Mismatch → warn + fall back (no split) |
| `scan.strict_index` | Missing/invalid index or shards → error | Issues → warn + continue |
| `scan.use_safetensors_index_json` | Enables index mode (required for `strict_index`) | Disables index handling |

---

## Version & Environment

- Python ≥ 3.12.9
- Base dependencies: `ml-dtypes`, `numpy`, `pandas`, `safetensors`
- Optional extras: `mlx` (`mlx`, `mlx-lm`), `parquet` (`pyarrow`), and `plot` (`matplotlib`, `pyarrow`)
- Uses `uv` for package management (Makefile assumes `.venv/bin/python`)
- Tests are `unittest`-based, no pytest
