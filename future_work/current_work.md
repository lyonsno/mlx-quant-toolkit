Current State

- The packaged CLI surface is now the supported entrypoint:
  - `mlx-quant-init`
  - `mlx-quant-collect`
  - `mlx-quant-build-tables`
  - `mlx-quant-build-plots`
- The placeholder top-level `main.py` entrypoint has been removed.
- The repo now ships an MIT [LICENSE], and `pyproject.toml` declares modern project-level MIT license metadata.
- `build_tables.py` now has:
  - duplicate-key hardening for `B_quant_deltas`
  - rerun-safe stale-output cleanup
  - cleanup containment for symlinks, poisoned manifests, and malformed owned paths
  - structured hard-failure artifacts via `logs/tables_failure.json`
- `collect_data.py` and `build_tables.py` both have stronger hard-failure audit coverage.
- `scan.strict_index` now requires an active index when enabled, and docs/tests are aligned to that contract.

Remaining High-Value Work

- Refresh [future_work/error_logging_and_hard_error_consistency.md](./error_logging_and_hard_error_consistency.md)
  to match what has already landed:
  - `build_tables.py` hard-failure artifacts are implemented
  - `strict_index` contract has been chosen and enforced
  - the main unresolved hard-error auditability gap is now `collect_data.py` follow-on polish, not the original broad inventory
- Decide whether to add an early `logs/run_health.json` "running" record that gets overwritten on success/failure.
  - This is still optional, but it is the clearest remaining auditability improvement for crash scenarios.

Refactor-Adjacent Optional Improvements

- Automate the release smoke we already ran manually:
  - clean `build/` and `dist/`
  - build wheel/sdist
  - install into a temp target
  - assert `import mlx_quant_toolkit.scripts.init_run` works
  - assert `import scripts` does not become the install surface
- Emit a warning when `delta_pairs` references schemes not present in `quant_sim`.
- Review whether any script/package-local duplication should get a sync guard test during the refactor.

Recently Resolved

- Optional dependency packaging is cleanly split between base dependencies and capability-scoped extras.
- `packed_split.projs` values are canonicalized through `parsing.proj_aliases`.
- Unmapped proj canonicalization uncertainty is surfaced via `logs/proj_canonicalization_report.{parquet|csv}` and warning summaries.
- `.npz` reading now uses a context manager.
- `logs/run_health.json` summarizes scan/extraction/output state, including index-aware counts.
- Run-level fallback visibility is exposed through `logs/run_health.json`.
- `proj_group` captures are canonicalized via `parsing.proj_aliases`.
- `model_path` supports single-file checkpoints via `_iter_weight_files`.
- Relative quant metrics now support denominator-floor stabilization, with defaults wired through normal `init_run -> collect_data` usage and null treated as unset.
- MLX quant compute dtype fallback behavior is hardened for no-`dtype` array runtimes, including fail-closed behavior for unsupported `bf16` paths.
- CLI packaging is package-local under `mlx_quant_toolkit`, with stronger namespace-isolation coverage and more descriptive help text.
- `build_tables.py` duplicate-delta cleanup is hardened across reruns, stale manifests, unrelated sidecars, malformed owned paths, and symlink containment.
- `build_tables.py` now emits `logs/tables_failure.json` on hard failures and clears stale failure artifacts on later success.

