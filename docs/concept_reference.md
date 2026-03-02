# Custom MLX Quant Tools - Concept Reference

## Core Concepts

### 1. Canonicalization

**Goal:** Convert arbitrary tensor shapes/axis orders into a uniform representation for downstream analysis.

**Canonical forms:**
- `(E, R, C)` for 3D tensors (expert, rows, cols)
- `(L, E, R, C)` for 4D tensors (layer, expert, rows, cols)
- `(R, C)` for 2D tensors (no expert dimension)

**Process:**
1. Match tensor name against an extraction rule (regex)
2. Rule specifies `layout`: mapping of input axes to `{layer_axis, expert_axis, rows_axis, cols_axis}`
3. Transpose/reshape to canonical order
4. Keep `source_tensor` (original name) and generate `derived_tensor` (e.g., `source::gate_proj`)

### 2. Extraction Rules

Each rule in `analysis_config.json` defines:
- `match`: regex pattern for tensor names
- `ndim`: expected number of dimensions
- `layout`: axis mapping to canonical positions
- `proj_group` (optional): regex capture group index that identifies the projection (e.g., gate_proj, up_proj)
- `expert_group` (optional): capture group for expert ID
- `packed_split` (optional): how to split fused matrices (see below)

If no rule matches, a heuristic fallback runs:
- 3D → `(E,R,C)` (guess expert axis is dim 0)
- 2D → `(R,C)` with expert ID parsed from name if possible

**Rule vs fallback counts** are tracked in `logs/run_health.json`.

### 3. Packed Splits

Some models fuse multiple projections (e.g., gate+up) into one tensor. `packed_split` breaks the canonical matrix into slices:

```json
"packed_split": {
  "axis": "rows" | "cols",
  "splits": [size1, size2, ...],
  "projs": ["proj_name1", "proj_name2", ...]
}
```

- Applied **after** canonicalization
- Each slice becomes a separate extracted matrix with its own `proj` label
- `derived_tensor` becomes `source_tensor::split[axis]::proj_name`
- `proj_group` and `packed_split.projs` are canonicalized via `parsing.proj_aliases`

### 4. Projection Canonicalization

Projection names (e.g., `w1`, `gate_proj`, `up`) are normalized using `parsing.proj_aliases`:
- `proj_aliases`: map of canonical names → list of alias patterns

Two strictness modes:
- `parsing.proj_group_strict = true`: unmapped `proj_group` tokens are **dropped** and reported as warnings (testable via `test_proj_group_normalization.py` and `test_collect_reporting_split_contract.py`)
- `parsing.proj_group_strict = false`: unmapped tokens are kept raw (logged as `kept_raw`)

### 5. Safetensors Index Support

When `scan.use_safetensors_index_json=true`:
- Discovers `model.safetensors.index.json`
- Uses it to select which shard files to scan
- Enriches `tensor_inventory` with `in_index` and `index_shard` columns
- Writes `logs/index_report.json`

`scan.strict_index` controls behavior:
- `true`: fail on missing/invalid index or missing shards (when index is active)
- `false`: warn and continue

**Index is active** when:
1. Index found and parsed successfully
2. `model_path` is a **directory** (index controls scan set)

If `model_path` is a single file, index is used only for reporting.

---