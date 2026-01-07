# Ticket: readme_staleness_review - Progress

- Goal: review README for stale or missing information vs current repo behavior.
- Plan: scan README and key scripts/config; compare outputs and config keys; update README if mismatches found.
- Changes made: none yet.
- Decisions / tradeoffs: will keep diffs minimal and only document currently implemented behavior.
- Assumptions: none so far.
- Open questions / ambiguities: none.
- Risk of test loophole: not applicable (doc-only change).
- Commands run: `ls`, `ls -a`, `cat README.md`, `sed -n '1,220p' scripts/collect_data.py`, `sed -n '1,220p' scripts/init_run.py`, `sed -n '1,240p' scripts/build_tables.py`, `sed -n '70,190p' README.md`, `cat pyproject.toml`, `cat .python-version`, `rg -n "run_context|run_health|write_manifest|index_report|analysis_config|strict_index|strict_packed_split|packed_split|quant_sim|tensor_inventory|matrix_stats|parquet" -S README.md scripts docs tests`, `rg -n "proj_group_strict|proj_group" -n scripts/collect_data.py`, `rg -n "metadata" -S scripts tests README.md`

- Changes made: updated `README.md` with parsing/metadata/debug config keys, output compression note, run output tree updates for plots + unmatched_tensors note, and clarified index_report condition.
- Decisions / tradeoffs: documented only fields present in config defaults or outputs; avoided deep schema detail to keep README concise.
- Assumptions: none.
- Open questions / ambiguities: none.
- Risk of test loophole: not applicable (doc-only change).
- Commands run: none after README edits.
