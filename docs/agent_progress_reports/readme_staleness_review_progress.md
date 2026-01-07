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

- Changes made: expanded `README.md` metadata section to document `metadata.mode` as a reserved/informational field.
- Decisions / tradeoffs: kept the description minimal since current code does not consume `metadata.mode`.
- Assumptions: none.
- Open questions / ambiguities: none.
- Risk of test loophole: not applicable (doc-only change).
- Commands run: `rg -n "metadata" -n scripts/collect_data.py scripts/metadata.py scripts/init_run.py`, `rg -n "metadata" -S README.md docs tests`, `sed -n '130,190p' README.md`

- Changes made: annotated README statements that depend on unresolved decisions with asterisks and added a brief "Pending decisions (*)" note.
- Decisions / tradeoffs: used a single footnote-style line to avoid cluttering the main sections.
- Assumptions: none.
- Open questions / ambiguities: none.
- Risk of test loophole: not applicable (doc-only change).
- Commands run: `cat future_work/current_work.md`

- Changes made: added an introductory "Pending decisions" section to `README.md` that complements the existing `*` markers.
- Decisions / tradeoffs: kept the section short and scoped to the three open policy questions to avoid duplicating detailed docs below.
- Assumptions: none.
- Open questions / ambiguities: none.
- Risk of test loophole: not applicable (doc-only change).
- Commands run: none after pending-decisions section edit.
