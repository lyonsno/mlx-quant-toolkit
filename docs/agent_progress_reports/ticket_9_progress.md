Ticket 9 progress
- Goal: strengthen build_tables test by asserting delta math/global stats without depending on collect_data.
- Plan: craft manual run dir fixtures (config + CSVs), update test to assert deltas and summaries, run test, then adjust if needed.
- Changes made: rewrote build_tables test in tests/test_optional_mlx.py to build matrix_stats/quant_sim CSVs with known values and verify B_quant_deltas and B_quant_global_summary outputs.
- Decisions / assumptions: use csv output format with delta pair scheme_a vs scheme_b; single base index is sufficient to exercise pivot/delta logic.
- Commands run: `python -m unittest tests.test_optional_mlx.OptionalMlxPipelineTests.test_build_tables_computes_quant_deltas_from_manual_inputs` (failed: numpy not installed in base interpreter); `UV_CACHE_DIR=.uv-cache uv run python -m unittest tests.test_optional_mlx.OptionalMlxPipelineTests.test_build_tables_computes_quant_deltas_from_manual_inputs` (failed: uv panic when using local cache under sandbox).
- Changes made: tightened quantize-failure test in tests/test_optional_mlx.py to assert expert_id coverage stays 0..E-1 even when quant simulation errors.
- Decisions / assumptions: use explicit 2x4x4 array input so expected expert count is deterministic; keeping stub quantize failure path for coverage.
- Commands run: none (tests not executed yet this round).
- Commands run (update): `python - <<'PY'\nimport numpy as np\nprint(np.__version__)\nPY` (failed: ModuleNotFoundError: No module named 'numpy').
