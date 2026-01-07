# Ticket: readme_repo_staleness_review - Progress

- Goal: review `README.md` against current repo behavior and identify stale/missing info; propose edits before changing the README.
- Plan: scan README, core scripts, tests, and config/deps; compare documented behavior to actual outputs; draft proposed README adjustments.
- Changes made: none (review only).
- Decisions / tradeoffs: defer any README edits until proposals are approved by the user.
- Assumptions: none.
- Open questions / ambiguities: none yet.
- Risk of test loophole: not applicable (doc-only review).
- Commands run: `ls`, `cat README.md`, `rg --files`, `sed -n '1,220p' scripts/collect_data.py`, `sed -n '220,520p' scripts/collect_data.py`, `sed -n '520,1040p' scripts/collect_data.py`, `sed -n '1040,1560p' scripts/collect_data.py`, `sed -n '1,200p' scripts/init_run.py`, `sed -n '1,240p' scripts/build_tables.py`, `sed -n '1,260p' scripts/metadata.py`, `sed -n '1,200p' main.py`, `cat Makefile`, `cat pyproject.toml`, `ls -a`, `sed -n '1,260p' tests/test_auditability_artifacts_integration.py`, `sed -n '1,240p' tests/test_index_found_semantics.py`, `rg -n "model_path" tests`, `sed -n '1,260p' tests/test_run_health_json_integration.py`, `sed -n '1,240p' tests/test_mlx_quant_sim_error.py`, `sed -n '1,240p' tests/test_optional_mlx.py`, `cat future_work/current_work.md`, `cat docs/agent_progress_reports/readme_staleness_review_progress.md`, `sed -n '1,260p' tests/test_metadata.py`, `sed -n '1,260p' tests/test_metadata_integration.py`, `sed -n '1,260p' tests/test_safetensors_index_integration.py`
- Notes: README appears mostly consistent with current behavior; potential gaps include index report fields (`extra_safetensors_files_on_disk`, `index_metadata`), run_health fields (`config_used`, `outputs_written`, tensor name examples/formats), and schema flags (`is_shared_expert`, `is_routed_expert`, `dtype`) in `matrix_stats` and identifiers in `quant_sim`. Optional-deps wording may be misleading given `pyproject.toml` declares `mlx`/`pyarrow` as required.
- Commands run: `rg -n "extra_safetensors|index_metadata|outputs_written|config_used|tensor_name_examples|derived_tensor_formats" README.md`, `rg -n "is_shared_expert|is_routed_expert|dtype" README.md`

- Update: proposed README edits in chat, then checked whether any were downstream of pending policy decisions in `future_work/current_work.md`; conclusion: no, because these edits are about documenting currently-emitted fields/columns and aligning runtime tolerance wording with current packaging.
- Commands run: `nl -ba future_work/current_work.md`

- Changes made: updated `README.md` to (a) clarify runtime-tolerant deps vs `pyproject.toml` packaging, (b) document additional `index_report.json` fields, (c) expand `run_context.json` and `run_health.json` descriptions, (d) expand `matrix_stats`/`quant_sim` schema bullets, and (e) add an optional `uv` test invocation line. (Per user request, did not modify `## Pending decisions`.)
- Commands run: none (edits via apply_patch only).

- Commit: `4dfa53c83142237186a1e34a1bca10ead9c14cf5` — README documents runtime-tolerant deps and audit artifacts
