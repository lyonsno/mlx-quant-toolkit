# example_safetensors_folder_metadata_convention_variance

This folder is a small corpus of static, non-canonical examples of how model directories encode metadata around weights.

These files are not here to define a schema. They exist to demonstrate variance and failure modes (empty config files,
different key naming conventions, different index metadata layouts, and similar weirdness).

## What’s inside

- Vendor-style first-level folders (for example `Qwen/`, `lmstudio-community/`), then model folders.
- Each model folder contains some subset of:
  - `config.json`
  - `model.safetensors.index.json`
  - `manifest.json` (generated; see below)

There are no `.safetensors` weight shards in this corpus.

## Manifests

- `manifest.json` at the root summarizes the corpus and lists each model entry.
- Each model folder also has a `manifest.json` that records:
  - file sizes and sha256 hashes
  - whether `config.json` / the index JSON parses cleanly
  - a small “summary” subset of keys that tend to matter for tooling
  - tags for “interesting quirks” (for example: empty config, `eos_token_id` is a list)

The manifests are descriptive, not prescriptive.

## Distribution

This folder exists for local development convenience (especially when iterating quickly with agentic tooling).
It is expected to be removed from the final repository and replaced with links or external references.
