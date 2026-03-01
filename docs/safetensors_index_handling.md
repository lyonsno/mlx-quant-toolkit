# Safetensors Index Handling and `strict_index` Behavior

## Overview

When working with large language models stored in `.safetensors` format, Hugging Face and other frameworks often provide a `model.safetensors.index.json` file that maps tensor names to the specific shard files that contain them. This index file is essential for efficiently locating tensors across multiple shards.

The `custom_mlx_quant_tools` pipeline supports optional index-based scanning through the `scan.use_safetensors_index_json` configuration flag. When enabled, the pipeline can:
- Discover and parse the index file
- Use it to limit which shard files are scanned
- Enrich the tensor inventory with `in_index` and `index_shard` metadata
- Report missing/extra shards and tensors

The `scan.strict_index` flag controls how strictly the pipeline enforces index consistency. This document explains the detailed behavior.

## Key Concepts

### Index Discovery
**Index discovery** is the process of searching for a `model.safetensors.index.json` file (or `*.safetensors.index.json`) in the model directory. Discovery is enabled by `scan.use_safetensors_index_json=true`. Even if the model_path points to a single file, discovery still occurs in the parent directory for reporting purposes.

### Index Mode Active
**Index mode is active** when:
1. Index discovery is enabled (`use_safetensors_index_json=true`)
2. An index file was found and successfully parsed
3. The index is actually used to determine which shard files to scan

This third condition depends on the `model_path`:
- If `model_path` is a **directory**, and an index is found → **active** (the index controls the scan set)
- If `model_path` is a **single file**, the index is discovered but **not active** (the scan is anchored to that file regardless of what the index says)

The distinction is crucial: some `strict_index` checks apply only when the index is *active* (i.e., when missing shards would actually cause data loss).

### What Makes an Index "Valid"?
An index is considered **valid** when:
1. It exists and is parseable as JSON
2. Its `weight_map` dictionary can be loaded
3. (For active mode) All shard files referenced in the index exist on disk

If any of these fail, the index is considered invalid or not fully usable.

## The `strict_index` Flag

When `scan.strict_index=true`, the following rules apply:

| Condition | Behavior |
|-----------|----------|
| `use_safetensors_index_json=false` | Immediate failure: "strict_index requires use_safetensors_index_json=true" |
| `use_safetensors_index_json=true` but no index file found | Failure: "strict_index requires an active index (status: ...)" |
| `use_safetensors_index_json=true`, index found and parsed, but `model_path` is a file | **Allowed** — index validity is enforced (it must exist and be parseable), but the scan is not expanded to other shards. |
| Index mode is active (directory + parsed index) and any referenced shard file is missing | Immediate failure with list of missing shards |
| Index mode is active and index is malformed (e.g., not a valid JSON) | Failure via "active index" check |

When `strict_index=false` (the default):
- Missing or invalid indexes trigger a **warning** instead of an error
- Missing shards (in active mode) are recorded in `logs/warnings.*` and `logs/run_health.json`, and the pipeline continues by scanning all candidate files (or just the anchor file for file model_path)

## Behavior Matrix

| `use_index` | `strict_index` | `model_path` | Index found? | Outcome |
|-------------|----------------|--------------|--------------|---------|
| false | any | dir or file | N/A | Index discovery disabled; no index logic runs |
| true | false | dir | no | Warning: missing index; scan all safetensors/npz files |
| true | false | dir | yes, parsed | Use index; missing shards → warning + continue |
| true | false | dir | yes, parse error | Warning: invalid index; scan all files |
| true | false | file | no | Warning: missing index; scan only the anchor file |
| true | false | file | yes, parsed | Index used for reporting only; scan only the anchor file |
| true | false | file | yes, parse error | Warning: invalid index; scan only the anchor file |
| true | true | dir | no | **ERROR**: strict_index requires an active index |
| true | true | dir | yes, parsed | **Active**; missing shards → **ERROR** |
| true | true | dir | yes, parse error | **ERROR**: strict_index requires an active index |
| true | true | file | no | **ERROR**: strict_index requires an active index |
| true | true | file | yes, parsed | **Reporting only**; no missing-shard check (scan stays on anchor file) |
| true | true | file | yes, parse error | **ERROR**: strict_index requires an active index |

**Note**: "Active" means the index determines the scan set. The `strict_index requires an active index` error occurs whenever `use_index=true` but `index_parsed=false` (i.e., index not found or parse failure), regardless of file/dir.

## Error Scenarios and Messages

### 1. `strict_index` without `use_safetensors_index_json`
```
Error: strict_index requires use_safetensors_index_json=true
```
**When**: `scan.strict_index=true` but `scan.use_safetensors_index_json=false`.
**Fix**: Either set `use_safetensors_index_json=true` or disable `strict_index`.

### 2. No index found when required
```
Error: strict_index requires an active index (status: not_found)
```
**When**: `use_safetensors_index_json=true`, `strict_index=true`, and no index file exists in the model directory (or parent directory for file `model_path`).
**Fix**: Provide an index file or set `strict_index=false`.

### 3. Index parse failure
```
Error: strict_index requires an active index (status: parse_error: ...)
```
**When**: Index file exists but contains invalid JSON or does not conform to expected schema.
**Fix**: Repair the index file or set `strict_index=false`.

### 4. Missing shards (active mode only)
```
Error: [index] missing shard(s) referenced by index: model-00002.safetensors, model-00003.safetensors
```
**When**: `model_path` is a directory, index is active, and one or more shard files listed in the index are absent from disk.
**Fix**: Ensure all indexed shards are present, or set `strict_index=false`.

## Logs and Diagnostics

When index handling is active, the pipeline writes:

- `logs/index_report.json`: Contains `missing_shards`, `extra_scanned_shards`, `missing_tensors`, `extra_tensors`, and metadata about the index.
- `logs/run_context.json`: Records `index_discovered_but_ignored_due_to_file_model_path`, `index_used_for_scan`, `index_parsed`, `index_status`.
- `logs/run_health.json`: Includes `expected_shards_count`, `scanned_shards_count`, `missing_shards_count`, `extra_scanned_shards_count`, `missing_tensors_count`.
- `data/tensor_inventory.*`: Adds `in_index` (bool) and `index_shard` (filename) columns when index is active.

When `strict_index=false` and issues are found, `logs/warnings.*` will contain entries such as:
```
[index] missing shard(s) referenced by index: model-00002.safetensors
[index] extra safetensors file(s) on disk not in index: orphan.safetensors
```

## Troubleshooting

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `strict_index` error when index exists | Index file is malformed or missing required fields | Validate JSON structure; ensure `weight_map` is present |
| Missing shard error despite having all files | Index lists shards with different paths/names than actual files | Align shard filenames with index entries |
| No index activity even though `use_safetensors_index_json=true` | `model_path` points to a single file; index is only used for reporting | Pass the directory containing the shards instead |
| Want to scan all tensors without index interference | Index file is stale or incomplete | Set `use_safetensors_index_json=false` or `strict_index=false` |
| Need to enforce index consistency in production | Accidental shard deletion or index mismatch | Use `strict_index=true` to fail fast |

## Relationship to Other Flags

- `scan.use_safetensors_index_json`: Enables index discovery and usage. `strict_index` is meaningless without it.
- `scan.inventory_all_tensors`: Controls whether tensors not in the index are still inventoried when index mode is active. This does not affect strictness.
- `parsing.strict_packed_split` and `parsing.proj_group_strict`: These handle tensor name parsing strictness and are independent of index handling.

## Examples

### Example 1: Directory with strict index (happy path)
```
model_dir/
├── model.safetensors.index.json  # lists shards: ["model-00001.safetensors", "model-00002.safetensors"]
├── model-00001.safetensors
└── model-00002.safetensors

Config: {"scan": {"use_safetensors_index_json": true, "strict_index": true}}
Result: Index parsed, both shards scanned, inventory enriched with index metadata.
```

### Example 2: Directory with missing shard, strict = false
Same setup but `model-00002.safetensors` is absent.
```
Result: Warning in logs/warnings.* about missing shard. Pipeline scans only model-00001.safetensors.
```

### Example 3: Directory with missing shard, strict = true
```
Result: Non-zero exit immediately upon discovering missing shard.
```

### Example 4: File model_path with strict = true
```
model_dir/
├── model.safetensors.index.json  # lists shards: ["model-00001.safetensors", "model-00002.safetensors"]
├── model-00001.safetensors
└── model-00002.safetensors

Command: python collect_data.py --model-path /path/to/model_dir/model-00001.safetensors
Config: {"scan": {"use_safetensors_index_json": true, "strict_index": true}}
Result: Index is parsed (validity enforced), but scan is anchored to the single file. 
        No check for missing shards because index is not active for scanning.
```

## Future Considerations

As the pipeline evolves, index handling may be extended to support:
- Partial index validation (e.g., warn on extra shards but don't fail)
- Index format version compatibility checks
- Automatic index regeneration from scanned tensors

These features would likely be controlled by additional configuration flags.