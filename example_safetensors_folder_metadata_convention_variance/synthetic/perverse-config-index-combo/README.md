# perverse-config-index-combo

This is a synthetic example intended to be "maximally annoying" while remaining valid JSON.

Goals:
- Demonstrate that multiple equivalent keys may coexist (`num_hidden_layers`, `n_layer`, `num_layers`) with conflicting values.
- Demonstrate that values may be strings or floats where you expected ints.
- Demonstrate that an index file's `weight_map` may reference shard names with odd casing, path separators, relative prefixes,
  and subdirectories.
- Demonstrate that tensor naming conventions can vary wildly even within a single index.

There are no actual `.safetensors` shards here. It's metadata-only.
