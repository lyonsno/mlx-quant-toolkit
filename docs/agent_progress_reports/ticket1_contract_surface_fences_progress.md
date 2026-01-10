## Ticket: CONTRACT SURFACE fences + README alignment

### Goal
- Add short, consistent `CONTRACT SURFACE:` fences adjacent to contract-writing sites (logs + stable output naming) so contract surfaces are harder to “forget”.
- Align `README.md` wording so it does not incorrectly claim some logs are “always” written in early-failure modes.

### Plan
1) Locate all write sites called out in the ticket (collect/build/init + README).
2) Add minimal comment-only fences near each contract-writing site (no schema dumps).
3) Update README “always” language and add a pointer to the markers.
4) Run unit tests to confirm no behavioral regression.

### Changes made
- `scripts/collect_data.py`
  - Added `CONTRACT SURFACE:` fences adjacent to writes of:
    - `logs/index_report.json`
    - `logs/model_config.raw.json`
    - `logs/model_shape_budget.json`
    - `logs/write_manifest.json`
    - `logs/run_health.json`
    - `logs/run_context.json`
    - `logs/warnings.{parquet|csv}` plus `write_manifest.artifacts["warnings"]`
  - Added a `CONTRACT SURFACE:` fence at the stable `write_manifest.artifacts` key map assignments.
- `scripts/build_tables.py`
  - Added `CONTRACT SURFACE:` fences for the Parquet→CSV fallback helper (and the explicit “no tables write_manifest; ask before adding” note).
  - Added `CONTRACT SURFACE:` fences adjacent to the hardcoded table output names (`A_*`, `B_*`, `B_quant_deltas`).
- `scripts/init_run.py`
  - Added `CONTRACT SURFACE:` fences adjacent to bootstrap artifact writes (`manifest.json`, `analysis_config.json` template).
- `README.md`
  - Changed `run_context.json`, `run_health.json`, `write_manifest.json` from “(always)” to “(written on successful completion of collect_data.py)”.
  - Added a short note that these logs are contract surfaces, may be absent on early failure, and are tagged in code with `CONTRACT SURFACE:` markers.
  - Added a clarification about the index gotcha: `index.index_path` may be set even when `status == "error"` (discovered but parse failed).

### Assumptions / notes
- This ticket is intentionally comment-only + README wording; no behavioral changes to outputs, schemas, or selection logic.

### Commands run
- `make test`
  - Result: PASS (`Ran 56 tests in 6.041s`, `OK`)

