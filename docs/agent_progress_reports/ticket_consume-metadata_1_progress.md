Goal: Add metadata reader for model config.json and persist shape budget into run outputs.
Plan:
- Inspect existing collect/init config paths and tests.
- Add unit tests for metadata parsing + config discovery + collect_data logging behavior.
- Implement metadata module and wire into init_run/collect_data.
Changes made:
- None yet.
Decisions / tradeoffs:
- None yet.
Assumptions:
- None yet.
Commands run:
- None yet.

Changes made:
- Added tests for metadata module parsing/discovery and collect_data integration.
  - tests/test_metadata.py
  - tests/test_metadata_integration.py
Decisions / tradeoffs:
- Decided to assert metadata log JSON structure with keys: config_path + shape_budget for integration tests.
- Added alias support test for experts_per_token -> num_experts_per_tok to cover gpt-oss config.
Assumptions:
- model_shape_budget.json will include top-level keys config_path and shape_budget.
- collect_data will emit a "[meta]" line when metadata is found.
Commands run:
- None yet.

---
Context update: first time revisiting this ticket in this session. Implement metadata module + wiring in init_run/collect_data; tests currently failing because metadata.py and logging aren’t implemented yet.
Changes made:
- Added scripts/metadata.py with config discovery/parsing, shape budget dataclass, and log trimming.
- Wired metadata section into scripts/init_run.py defaults.
- Added metadata handling in scripts/collect_data.py to locate/parse config, write logs, and emit [meta]/warnings.
Commands run:
- None yet.
Changes made:
- Switched MLX import to lazy loading to avoid Metal crash when mlx disabled.
- Adjusted metadata integration test to compare resolved config paths.
Commands run:
- ./.venv/bin/python -m unittest tests.test_metadata tests.test_metadata_integration (pass; init_run emits DeprecationWarning about datetime.utcnow)
