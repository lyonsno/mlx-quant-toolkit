import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildTablesContractTests(unittest.TestCase):
    MATRIX_STATS_INPUT_COLUMNS = [
        "layer",
        "proj",
        "mean",
        "std",
        "mean_abs",
        "rms",
        "max_abs",
        "p50_abs",
        "p99_abs",
        "p999_abs",
        "outlier_max_over_mean",
        "outlier_p99_over_median",
        "outlier_p999_over_median",
    ]

    QUANT_SIM_INPUT_COLUMNS = [
        "derived_tensor",
        "layer",
        "proj",
        "expert_id",
        "rows",
        "cols",
        "scheme",
        "w_rel_fro",
        "w_rel_max",
        "scale_mean",
        "scale_max",
        "bias_mean",
        "bias_max",
        "error",
    ]

    A_STATS = [
        "mean",
        "std",
        "mean_abs",
        "rms",
        "max_abs",
        "p50_abs",
        "p99_abs",
        "p999_abs",
        "outlier_max_over_mean",
        "outlier_p99_over_median",
        "outlier_p999_over_median",
    ]

    B_METRICS = [
        "w_rel_fro",
        "w_rel_max",
        "scale_mean",
        "scale_max",
        "bias_mean",
        "bias_max",
    ]

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _run(self, args, env=None, check=True):
        return subprocess.run(
            args,
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _write_config(self, run_dir: Path, *, output_format: str, compression, delta_pairs=None):
        if delta_pairs is None:
            delta_pairs = []
        cfg = {
            "output": {"format": output_format, "compression": compression},
            "delta_pairs": delta_pairs,
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def _write_matrix_stats_csv(self, data_dir: Path, rows):
        with (data_dir / "matrix_stats.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.MATRIX_STATS_INPUT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self.MATRIX_STATS_INPUT_COLUMNS})

    def _write_quant_sim_csv(self, data_dir: Path, rows):
        with (data_dir / "quant_sim.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.QUANT_SIM_INPUT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self.QUANT_SIM_INPUT_COLUMNS})

    def _write_csv(self, path: Path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _run_build_tables(self, run_dir: Path):
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        result = self._run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "build_tables.py"),
                "--run-dir",
                str(run_dir),
            ],
            env=env,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.returncode, 0, f"build_tables failed: {output}")

    def _build_tables_path(self) -> Path:
        return self.repo_root / "scripts" / "build_tables.py"

    def _read_csv(self, path: Path):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return reader.fieldnames, rows

    def _expected_a_columns(self, group_cols, agg_cols):
        cols = list(group_cols)
        for stat in self.A_STATS:
            for agg in agg_cols:
                cols.append(f"{stat}__{agg}")
        return cols

    def _expected_b_columns(self, group_cols, agg_cols):
        cols = list(group_cols)
        for metric in self.B_METRICS:
            for agg in agg_cols:
                cols.append(f"{metric}__{agg}")
        return cols

    def _sample_matrix_rows(self):
        return [
            {
                "layer": 0,
                "proj": "down_proj",
                "mean": 1.0,
                "std": 0.1,
                "mean_abs": 1.0,
                "rms": 1.0,
                "max_abs": 1.2,
                "p50_abs": 1.0,
                "p99_abs": 1.2,
                "p999_abs": 1.2,
                "outlier_max_over_mean": 1.2,
                "outlier_p99_over_median": 1.2,
                "outlier_p999_over_median": 1.2,
            },
            {
                "layer": 1,
                "proj": "down_proj",
                "mean": 2.0,
                "std": 0.2,
                "mean_abs": 2.0,
                "rms": 2.0,
                "max_abs": 2.2,
                "p50_abs": 2.0,
                "p99_abs": 2.2,
                "p999_abs": 2.2,
                "outlier_max_over_mean": 1.1,
                "outlier_p99_over_median": 1.1,
                "outlier_p999_over_median": 1.1,
            },
        ]

    def _sample_quant_rows(self):
        return [
            {
                "derived_tensor": "layers.0.experts.0.down_proj.weight",
                "layer": 0,
                "proj": "down_proj",
                "expert_id": 0,
                "rows": 2,
                "cols": 2,
                "scheme": "scheme_a",
                "w_rel_fro": 0.10,
                "w_rel_max": 0.15,
                "scale_mean": 0.0,
                "scale_max": 0.0,
                "bias_mean": 0.0,
                "bias_max": 0.0,
                "error": "",
            },
            {
                "derived_tensor": "layers.0.experts.0.down_proj.weight",
                "layer": 0,
                "proj": "down_proj",
                "expert_id": 0,
                "rows": 2,
                "cols": 2,
                "scheme": "scheme_b",
                "w_rel_fro": 0.20,
                "w_rel_max": 0.25,
                "scale_mean": 0.0,
                "scale_max": 0.0,
                "bias_mean": 0.0,
                "bias_max": 0.0,
                "error": "",
            },
            {
                "derived_tensor": "layers.1.experts.0.down_proj.weight",
                "layer": 1,
                "proj": "down_proj",
                "expert_id": 0,
                "rows": 2,
                "cols": 2,
                "scheme": "scheme_a",
                "w_rel_fro": 0.30,
                "w_rel_max": 0.35,
                "scale_mean": 0.0,
                "scale_max": 0.0,
                "bias_mean": 0.0,
                "bias_max": 0.0,
                "error": "",
            },
            {
                "derived_tensor": "layers.1.experts.0.down_proj.weight",
                "layer": 1,
                "proj": "down_proj",
                "expert_id": 0,
                "rows": 2,
                "cols": 2,
                "scheme": "scheme_b",
                "w_rel_fro": 0.40,
                "w_rel_max": 0.45,
                "scale_mean": 0.0,
                "scale_max": 0.0,
                "bias_mean": 0.0,
                "bias_max": 0.0,
                "error": "",
            },
        ]

    def test_build_tables_a_table_schema_invariants(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            a_layer_cols, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            a_block4_cols, a_block4_rows = self._read_csv(run_dir / "tables" / "A_weight_block4_summary.csv")
            a_global_cols, a_global_rows = self._read_csv(run_dir / "tables" / "A_weight_global_summary.csv")

            self.assertEqual(
                a_layer_cols,
                self._expected_a_columns(["layer", "proj"], ["median", "mean", "std", "p90", "p99"]),
            )
            self.assertEqual(
                a_block4_cols,
                self._expected_a_columns(["block4", "proj"], ["median", "mean", "std", "p90", "p99"]),
            )
            self.assertEqual(
                a_global_cols,
                self._expected_a_columns(["proj"], ["min", "p01", "median", "p99", "max"]),
            )

            self.assertEqual(len(a_layer_rows), 2)
            self.assertEqual(len(a_block4_rows), 1)
            self.assertEqual(len(a_global_rows), 1)

    def test_build_tables_zero_row_inputs_preserve_headers_and_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)
            self._write_matrix_stats_csv(data_dir, [])
            self._write_quant_sim_csv(data_dir, [])

            self._run_build_tables(run_dir)

            expected_headers = {
                "A_weight_layer_summary": self._expected_a_columns(["layer", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_block4_summary": self._expected_a_columns(["block4", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_global_summary": self._expected_a_columns(["proj"], ["min", "p01", "median", "p99", "max"]),
                "B_quant_layer_summary": self._expected_b_columns(["layer", "proj", "scheme"], ["median", "mean", "p90", "p99"]),
                "B_quant_block4_summary": self._expected_b_columns(["block4", "proj", "scheme"], ["median", "mean", "p90", "p99"]),
                "B_quant_global_summary": self._expected_b_columns(["proj", "scheme"], ["min", "p01", "median", "p99", "max"]),
            }

            for artifact, expected in expected_headers.items():
                cols, rows = self._read_csv(run_dir / "tables" / f"{artifact}.csv")
                self.assertEqual(cols, expected)
                self.assertEqual(len(rows), 0)

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), sorted(expected_headers))
            for name in expected_headers:
                entry = artifacts[name]
                self.assertEqual(entry.get("format"), "csv")
                self.assertFalse(entry.get("fallback"))
                self.assertEqual(entry.get("error"), "")
                self.assertEqual(entry.get("rows"), 0)
                self.assertEqual(entry.get("path"), f"tables/{name}.csv")

    def test_build_tables_handles_schema_less_empty_loader_frames(self):
        mod = _load_module("build_tables_schema_less_empty_contract", self._build_tables_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            read_calls: list[Path] = []

            # Distinct from zero-row/header-only input: this simulates a loader path that
            # yields a DataFrame with no schema columns at all.
            def _fake_read_df(path: Path, empty_columns=None):
                read_calls.append(path)
                return mod.pd.DataFrame()

            orig_read_df = mod._read_df
            old_argv = sys.argv
            try:
                mod._read_df = _fake_read_df
                sys.argv = ["build_tables.py", "--run-dir", str(run_dir)]
                mod.main()
            finally:
                mod._read_df = orig_read_df
                sys.argv = old_argv

            self.assertEqual(len(read_calls), 2, "build_tables should read matrix_stats and quant_sim once each")
            expected_read_paths = [
                (run_dir / "data" / "matrix_stats.parquet").resolve(),
                (run_dir / "data" / "quant_sim.parquet").resolve(),
            ]
            self.assertEqual(
                Counter(path.resolve() for path in read_calls),
                Counter(expected_read_paths),
            )

            # Contract note: in fully schema-less mode, A summaries keep their full metric
            # columns (stat_cols are fixed in build_tables), while B summaries remain
            # axis-only because quant metric columns are discovered from input availability.
            expected_headers = {
                "A_weight_layer_summary": self._expected_a_columns(["layer", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_block4_summary": self._expected_a_columns(["block4", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_global_summary": self._expected_a_columns(["proj"], ["min", "p01", "median", "p99", "max"]),
                "B_quant_layer_summary": ["layer", "proj", "scheme"],
                "B_quant_block4_summary": ["block4", "proj", "scheme"],
                "B_quant_global_summary": ["proj", "scheme"],
            }

            for artifact, expected_cols in expected_headers.items():
                cols, rows = self._read_csv(run_dir / "tables" / f"{artifact}.csv")
                self.assertEqual(cols, expected_cols)
                self.assertEqual(rows, [])

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), sorted(expected_headers))
            for name in expected_headers:
                entry = artifacts[name]
                self.assertEqual(entry.get("format"), "csv")
                self.assertFalse(entry.get("fallback"))
                self.assertEqual(entry.get("error"), "")
                self.assertEqual(entry.get("rows"), 0)
                self.assertEqual(entry.get("path"), f"tables/{name}.csv")

    def test_build_tables_handles_schema_less_matrix_stats_with_valid_quant_rows(self):
        mod = _load_module("build_tables_schema_less_matrix_contract", self._build_tables_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            read_calls: list[Path] = []

            def _fake_read_df(path: Path, empty_columns=None):
                read_calls.append(path)
                if path.name.startswith("matrix_stats"):
                    return mod.pd.DataFrame()
                if path.name.startswith("quant_sim"):
                    return mod.pd.DataFrame(
                        [
                            {
                                "derived_tensor": "layers.0.experts.0.down_proj.weight",
                                "layer": 0,
                                "proj": "down_proj",
                                "expert_id": 0,
                                "rows": 2,
                                "cols": 2,
                                "scheme": "scheme_a",
                                "w_rel_fro": 0.1,
                                "w_rel_max": 0.2,
                                "scale_mean": 0.0,
                                "scale_max": 0.0,
                                "bias_mean": 0.0,
                                "bias_max": 0.0,
                            }
                        ]
                    )
                raise AssertionError(f"Unexpected input path in fake _read_df: {path}")

            orig_read_df = mod._read_df
            old_argv = sys.argv
            try:
                mod._read_df = _fake_read_df
                sys.argv = ["build_tables.py", "--run-dir", str(run_dir)]
                mod.main()
            finally:
                mod._read_df = orig_read_df
                sys.argv = old_argv

            self.assertEqual(len(read_calls), 2, "build_tables should read matrix_stats and quant_sim once each")
            expected_read_paths = [
                (run_dir / "data" / "matrix_stats.parquet").resolve(),
                (run_dir / "data" / "quant_sim.parquet").resolve(),
            ]
            self.assertEqual(
                Counter(path.resolve() for path in read_calls),
                Counter(expected_read_paths),
            )

            expected_headers = {
                "A_weight_layer_summary": self._expected_a_columns(["layer", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_block4_summary": self._expected_a_columns(["block4", "proj"], ["median", "mean", "std", "p90", "p99"]),
                "A_weight_global_summary": self._expected_a_columns(["proj"], ["min", "p01", "median", "p99", "max"]),
                "B_quant_layer_summary": self._expected_b_columns(["layer", "proj", "scheme"], ["median", "mean", "p90", "p99"]),
                "B_quant_block4_summary": self._expected_b_columns(["block4", "proj", "scheme"], ["median", "mean", "p90", "p99"]),
                "B_quant_global_summary": self._expected_b_columns(["proj", "scheme"], ["min", "p01", "median", "p99", "max"]),
            }
            expected_rows = {
                "A_weight_layer_summary": 0,
                "A_weight_block4_summary": 0,
                "A_weight_global_summary": 0,
                "B_quant_layer_summary": 1,
                "B_quant_block4_summary": 1,
                "B_quant_global_summary": 1,
            }
            for artifact, expected_cols in expected_headers.items():
                cols, rows = self._read_csv(run_dir / "tables" / f"{artifact}.csv")
                self.assertEqual(cols, expected_cols)
                self.assertEqual(len(rows), expected_rows[artifact])

            _, b_global_rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")
            self.assertEqual(b_global_rows[0]["proj"], "down_proj")
            self.assertEqual(b_global_rows[0]["scheme"], "scheme_a")
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_fro__median"]), 0.1)
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_max__median"]), 0.2)

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), sorted(expected_headers))
            for name in expected_headers:
                entry = artifacts[name]
                self.assertEqual(entry.get("format"), "csv")
                self.assertFalse(entry.get("fallback"))
                self.assertEqual(entry.get("error"), "")
                self.assertEqual(entry.get("rows"), expected_rows[name])
                self.assertEqual(entry.get("path"), f"tables/{name}.csv")

    def test_build_tables_handles_non_empty_matrix_stats_missing_numeric_columns(self):
        mod = _load_module("build_tables_partial_schema_numeric_contract", self._build_tables_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            def _fake_read_df(path: Path, empty_columns=None):
                if path.name.startswith("matrix_stats"):
                    # Non-empty and intentionally partial schema: numeric stat columns like
                    # "mean"/"std" are missing, but one numeric metric is present.
                    return mod.pd.DataFrame(
                        [
                            {
                                "layer": 0,
                                "proj": "down_proj",
                                "mean_abs": 1.5,
                            }
                        ]
                    )
                if path.name.startswith("quant_sim"):
                    # Keep quant side empty so this test isolates A-table robustness.
                    return mod.pd.DataFrame(
                        columns=[
                            "derived_tensor",
                            "layer",
                            "proj",
                            "expert_id",
                            "rows",
                            "cols",
                            "scheme",
                        ]
                    )
                raise AssertionError(f"Unexpected input path in fake _read_df: {path}")

            orig_read_df = mod._read_df
            old_argv = sys.argv
            try:
                mod._read_df = _fake_read_df
                sys.argv = ["build_tables.py", "--run-dir", str(run_dir)]
                mod.main()
            finally:
                mod._read_df = orig_read_df
                sys.argv = old_argv

            a_layer_cols, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertIn("mean__median", a_layer_cols)
            self.assertIn("mean_abs__median", a_layer_cols)
            self.assertEqual(len(a_layer_rows), 1)
            row = a_layer_rows[0]
            self.assertEqual(row["layer"], "0")
            self.assertEqual(row["proj"], "down_proj")
            self.assertEqual(row["mean__median"], "")
            self.assertAlmostEqual(float(row["mean_abs__median"]), 1.5)

            # Hardening: verify that added columns have proper missing-value sentinels and numeric dtypes.
            # Load the CSV with pandas to inspect dtypes and NA values.
            df = mod.pd.read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            # The mean__median column should be NA/NaN because "mean" was missing from input schema.
            self.assertTrue(mod.pd.isna(df["mean__median"].iloc[0]))
            # The column dtype should be numeric (float64), not object.
            self.assertTrue(
                mod.pd.api.types.is_numeric_dtype(df["mean__median"].dtype),
                f"mean__median should have numeric dtype, got {df['mean__median'].dtype}",
            )
            # mean_abs__median should have a valid numeric value.
            self.assertTrue(mod.pd.api.types.is_numeric_dtype(df["mean_abs__median"].dtype))
            self.assertFalse(mod.pd.isna(df["mean_abs__median"].iloc[0]))

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(artifacts["A_weight_layer_summary"]["rows"], 1)

    def test_build_tables_block4_derived_from_layer_is_na_when_layer_is_na(self):
        """
        Contract test: when layer column exists but contains all NA values,
        block4 computed via floordiv(4) should also be all NA.

        This catches the case where _ensure_columns adds layer with pd.NA
        and block4 = layer.floordiv(4) would propagate NA incorrectly.
        """
        mod = _load_module("build_tables_block4_na_contract", self._build_tables_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            def _fake_read_df(path: Path, empty_columns=None):
                if path.name.startswith("matrix_stats"):
                    # layer exists but is all NA; proj has a value
                    return mod.pd.DataFrame(
                        [
                            {
                                "layer": mod.pd.NA,
                                "proj": "down_proj",
                            }
                        ]
                    )
                if path.name.startswith("quant_sim"):
                    return mod.pd.DataFrame(
                        columns=[
                            "derived_tensor",
                            "layer",
                            "proj",
                            "expert_id",
                            "rows",
                            "cols",
                            "scheme",
                        ]
                    )
                raise AssertionError(f"Unexpected input path in fake _read_df: {path}")

            orig_read_df = mod._read_df
            old_argv = sys.argv
            try:
                mod._read_df = _fake_read_df
                sys.argv = ["build_tables.py", "--run-dir", str(run_dir)]
                mod.main()
            finally:
                mod._read_df = orig_read_df
                sys.argv = old_argv

            # Load A_weight_block4_summary and verify block4 is all NA when layer is NA
            block4_cols, block4_rows = self._read_csv(run_dir / "tables" / "A_weight_block4_summary.csv")
            self.assertIn("block4", block4_cols)
            self.assertEqual(len(block4_rows), 1)

            # The block4 value should be NA/empty since it's derived from NA layer
            self.assertEqual(block4_rows[0]["block4"], "")

            # Also verify via pandas that it's genuinely NA, not just empty string
            df = mod.pd.read_csv(run_dir / "tables" / "A_weight_block4_summary.csv")
            self.assertTrue(
                mod.pd.isna(df["block4"].iloc[0]) or df["block4"].iloc[0] == "",
                f"block4 should be NA when layer is NA, got {df['block4'].iloc[0]}",
            )

            # Verify layer is also NA in the layer summary
            layer_cols, layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertEqual(len(layer_rows), 1)
            self.assertEqual(layer_rows[0]["layer"], "")

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(artifacts["A_weight_block4_summary"]["rows"], 1)
            self.assertEqual(artifacts["A_weight_layer_summary"]["rows"], 1)

    def test_build_tables_parquet_fallback_contract(self):
       with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="parquet", compression="invalid-codec")
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())
            self._run_build_tables(run_dir)

            expected_rows = {
                "A_weight_layer_summary": 2,
                "A_weight_block4_summary": 1,
                "A_weight_global_summary": 1,
                "B_quant_layer_summary": 4,
                "B_quant_block4_summary": 2,
                "B_quant_global_summary": 2,
            }

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), sorted(expected_rows))

            for name, expected_count in expected_rows.items():
                entry = artifacts[name]
                self.assertEqual(entry.get("format"), "csv")
                self.assertTrue(entry.get("fallback"))
                self.assertIsInstance(entry.get("error"), str)
                self.assertTrue(entry.get("error"))
                self.assertEqual(entry.get("path"), f"tables/{name}.csv")
                self.assertEqual(entry.get("rows"), expected_count)

                csv_cols, csv_rows = self._read_csv(run_dir / "tables" / f"{name}.csv")
                self.assertGreater(len(csv_cols), 0)
                self.assertEqual(len(csv_rows), expected_count)

                parquet_path = run_dir / "tables" / f"{name}.parquet"
                self.assertFalse(parquet_path.exists(), f"Unexpected stale parquet file: {parquet_path}")

            a_global_cols, a_global_rows = self._read_csv(run_dir / "tables" / "A_weight_global_summary.csv")
            self.assertIn("mean__median", a_global_cols)
            self.assertEqual(len(a_global_rows), 1)
            self.assertAlmostEqual(float(a_global_rows[0]["mean__median"]), 1.5)

    def test_build_tables_prefers_collect_manifest_paths_over_stale_extension_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            logs_dir = run_dir / "logs"
            data_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            # Current intended inputs (CSV): mean is intentionally high to detect stale reads.
            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 100.0,
                        "std": 0.1,
                        "mean_abs": 100.0,
                        "rms": 100.0,
                        "max_abs": 101.0,
                        "p50_abs": 100.0,
                        "p99_abs": 101.0,
                        "p999_abs": 101.0,
                        "outlier_max_over_mean": 1.01,
                        "outlier_p99_over_median": 1.01,
                        "outlier_p999_over_median": 1.01,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.10,
                        "w_rel_max": 0.15,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    }
                ],
            )

            # Poison-pill stale defaults: if build_tables ignores the collect manifest path map
            # and prefers extension-based defaults, it will try to open these files and fail.
            (data_dir / "matrix_stats.parquet").write_text("stale invalid parquet")
            (data_dir / "quant_sim.parquet").write_text("stale invalid parquet")

            collect_manifest = {
                "generated_at": "2026-03-05T00:00:00Z",
                "requested_format": "parquet",
                "requested_compression": "invalid-codec",
                "artifacts": {
                    "matrix_stats": {
                        "path": "data/matrix_stats.csv",
                        "format": "csv",
                        "fallback": True,
                        "error": "ValueError: invalid codec",
                        "rows": 1,
                    },
                    "quant_sim": {
                        "path": "data/quant_sim.csv",
                        "format": "csv",
                        "fallback": True,
                        "error": "ValueError: invalid codec",
                        "rows": 1,
                    },
                },
            }
            (logs_dir / "write_manifest.json").write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            _, rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["mean__median"]), 100.0)

    def test_build_tables_manifest_windows_separator_paths_still_override_stale_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            logs_dir = run_dir / "logs"
            data_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 123.0,
                        "std": 0.1,
                        "mean_abs": 123.0,
                        "rms": 123.0,
                        "max_abs": 124.0,
                        "p50_abs": 123.0,
                        "p99_abs": 124.0,
                        "p999_abs": 124.0,
                        "outlier_max_over_mean": 1.0,
                        "outlier_p99_over_median": 1.0,
                        "outlier_p999_over_median": 1.0,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.10,
                        "w_rel_max": 0.15,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    }
                ],
            )

            # Poison-pill stale defaults to prove manifest path normalization is used.
            (data_dir / "matrix_stats.parquet").write_text("stale invalid parquet")
            (data_dir / "quant_sim.parquet").write_text("stale invalid parquet")

            # Simulate a collect manifest produced on Windows.
            collect_manifest = {
                "generated_at": "2026-03-05T00:00:00Z",
                "requested_format": "parquet",
                "requested_compression": "invalid-codec",
                "artifacts": {
                    "matrix_stats": {
                        "path": "data\\matrix_stats.csv",
                        "format": "csv",
                        "fallback": True,
                        "error": "ValueError: invalid codec",
                        "rows": 1,
                    },
                    "quant_sim": {
                        "path": "data\\quant_sim.csv",
                        "format": "csv",
                        "fallback": True,
                        "error": "ValueError: invalid codec",
                        "rows": 1,
                    },
                },
            }
            (logs_dir / "write_manifest.json").write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            _, rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["mean__median"]), 123.0)

    def test_build_tables_ignores_manifest_data_paths_with_unsupported_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            logs_dir = run_dir / "logs"
            data_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            # Valid default inputs that should be used when manifest entries are invalid.
            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 42.0,
                        "std": 0.1,
                        "mean_abs": 42.0,
                        "rms": 42.0,
                        "max_abs": 43.0,
                        "p50_abs": 42.0,
                        "p99_abs": 43.0,
                        "p999_abs": 43.0,
                        "outlier_max_over_mean": 1.0,
                        "outlier_p99_over_median": 1.0,
                        "outlier_p999_over_median": 1.0,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_default",
                        "w_rel_fro": 0.10,
                        "w_rel_max": 0.15,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    }
                ],
            )

            # Poison-pill files with unsupported suffixes under data/.
            (data_dir / "poison_matrix_stats.txt").write_text("not a table")
            (data_dir / "poison_quant_sim.txt").write_text("not a table")

            collect_manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "matrix_stats": {
                        "path": "data/poison_matrix_stats.txt",
                        "format": "txt",
                        "fallback": False,
                        "error": "",
                        "rows": 0,
                    },
                    "quant_sim": {
                        "path": "data/poison_quant_sim.txt",
                        "format": "txt",
                        "fallback": False,
                        "error": "",
                        "rows": 0,
                    },
                },
            }
            (logs_dir / "write_manifest.json").write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            a_layer_cols, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertIn("mean__median", a_layer_cols)
            self.assertEqual(len(a_layer_rows), 1)
            self.assertAlmostEqual(float(a_layer_rows[0]["mean__median"]), 42.0)

            _, b_global_rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")
            self.assertEqual(len(b_global_rows), 1)
            self.assertEqual(b_global_rows[0]["scheme"], "scheme_default")
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_fro__median"]), 0.10)
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_max__median"]), 0.15)

    def test_build_tables_ignores_manifest_unsupported_suffix_for_matrix_stats_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            logs_dir = run_dir / "logs"
            data_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 77.0,
                        "std": 0.1,
                        "mean_abs": 77.0,
                        "rms": 77.0,
                        "max_abs": 78.0,
                        "p50_abs": 77.0,
                        "p99_abs": 78.0,
                        "p999_abs": 78.0,
                        "outlier_max_over_mean": 1.0,
                        "outlier_p99_over_median": 1.0,
                        "outlier_p999_over_median": 1.0,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_q_only",
                        "w_rel_fro": 0.33,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    }
                ],
            )
            (data_dir / "poison_matrix_stats.txt").write_text("not a table")

            collect_manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "matrix_stats": {
                        "path": "data/poison_matrix_stats.txt",
                        "format": "txt",
                        "fallback": False,
                        "error": "",
                        "rows": 0,
                    },
                    "quant_sim": {
                        "path": "data/quant_sim.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                },
            }
            (logs_dir / "write_manifest.json").write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            _, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertEqual(len(a_layer_rows), 1)
            self.assertAlmostEqual(float(a_layer_rows[0]["mean__median"]), 77.0)

            _, b_global_rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")
            self.assertEqual(len(b_global_rows), 1)
            self.assertEqual(b_global_rows[0]["scheme"], "scheme_q_only")
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_fro__median"]), 0.33)
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_max__median"]), 0.44)

    def test_build_tables_ignores_manifest_unsupported_suffix_for_quant_sim_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            logs_dir = run_dir / "logs"
            data_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 88.0,
                        "std": 0.1,
                        "mean_abs": 88.0,
                        "rms": 88.0,
                        "max_abs": 89.0,
                        "p50_abs": 88.0,
                        "p99_abs": 89.0,
                        "p999_abs": 89.0,
                        "outlier_max_over_mean": 1.0,
                        "outlier_p99_over_median": 1.0,
                        "outlier_p999_over_median": 1.0,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_q_fallback",
                        "w_rel_fro": 0.55,
                        "w_rel_max": 0.66,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    }
                ],
            )
            (data_dir / "poison_quant_sim.txt").write_text("not a table")

            collect_manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "matrix_stats": {
                        "path": "data/matrix_stats.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                    "quant_sim": {
                        "path": "data/poison_quant_sim.txt",
                        "format": "txt",
                        "fallback": False,
                        "error": "",
                        "rows": 0,
                    },
                },
            }
            (logs_dir / "write_manifest.json").write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            _, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            self.assertEqual(len(a_layer_rows), 1)
            self.assertAlmostEqual(float(a_layer_rows[0]["mean__median"]), 88.0)

            _, b_global_rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")
            self.assertEqual(len(b_global_rows), 1)
            self.assertEqual(b_global_rows[0]["scheme"], "scheme_q_fallback")
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_fro__median"]), 0.55)
            self.assertAlmostEqual(float(b_global_rows[0]["w_rel_max__median"]), 0.66)

    def test_build_tables_handles_quant_sim_rows_without_metric_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())

            # Contract: B tables should still be writable when quant metric columns are absent
            # (for example from partially degraded upstream output).
            self._write_csv(
                data_dir / "quant_sim.csv",
                fieldnames=[
                    "derived_tensor",
                    "layer",
                    "block4",
                    "proj",
                    "expert_id",
                    "rows",
                    "cols",
                    "scheme",
                    "error",
                ],
                rows=[
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "block4": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "error": "",
                    }
                ],
            )

            self._run_build_tables(run_dir)

            b_layer_cols, b_layer_rows = self._read_csv(run_dir / "tables" / "B_quant_layer_summary.csv")
            b_block4_cols, b_block4_rows = self._read_csv(run_dir / "tables" / "B_quant_block4_summary.csv")
            b_global_cols, b_global_rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")

            self.assertEqual(b_layer_cols, ["layer", "proj", "scheme"])
            self.assertEqual(b_block4_cols, ["block4", "proj", "scheme"])
            self.assertEqual(b_global_cols, ["proj", "scheme"])

            self.assertEqual(len(b_layer_rows), 1)
            self.assertEqual(len(b_block4_rows), 1)
            self.assertEqual(len(b_global_rows), 1)

            manifest = json.loads((run_dir / "logs" / "tables_write_manifest.json").read_text())
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(artifacts["B_quant_layer_summary"]["rows"], 1)
            self.assertEqual(artifacts["B_quant_block4_summary"]["rows"], 1)
            self.assertEqual(artifacts["B_quant_global_summary"]["rows"], 1)

    def test_build_tables_handles_delta_pairs_when_quant_metric_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())

            # Deliberately omit w_rel_fro/w_rel_max and other quant metric columns.
            self._write_csv(
                data_dir / "quant_sim.csv",
                fieldnames=[
                    "derived_tensor",
                    "layer",
                    "block4",
                    "proj",
                    "expert_id",
                    "rows",
                    "cols",
                    "scheme",
                    "error",
                ],
                rows=[
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "block4": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "error": "",
                    }
                ],
            )

            self._run_build_tables(run_dir)

            cols, rows = self._read_csv(run_dir / "tables" / "B_quant_deltas.csv")
            self.assertEqual(
                cols,
                [
                    "derived_tensor",
                    "layer",
                    "block4",
                    "proj",
                    "expert_id",
                    "rows",
                    "cols",
                    "delta_name",
                    "delta_w_rel_fro",
                    "delta_w_rel_max",
                ],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["delta_name"], "delta_ab")
            self.assertEqual(rows[0]["delta_w_rel_fro"], "")
            self.assertEqual(rows[0]["delta_w_rel_max"], "")

    def test_build_tables_fails_fast_without_writing_outputs_when_quant_delta_base_keys_are_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.15,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        # CONTRACT: duplicate base keys for B_quant_deltas must not
                        # silently collapse through first-value semantics.
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.31,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.09,
                        "w_rel_max": 0.12,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            # CONTRACT: duplicate delta base keys are treated as early validation
            # errors, not as order-sensitive first-value collapses.
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertIn("scheme_a", output)
            self.assertIn("down_proj", output)
            self.assertFalse(
                (run_dir / "tables").exists(),
                "Duplicate delta-key failures should happen before writing tables",
            )
            self.assertFalse(
                (run_dir / "logs" / "tables_write_manifest.json").exists(),
                "Duplicate delta-key failures should happen before writing tables manifests",
            )

    def test_build_tables_duplicate_quant_delta_key_failures_clear_stale_outputs_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            stale_table_paths = [
                run_dir / "tables" / "A_weight_layer_summary.csv",
                run_dir / "tables" / "A_weight_block4_summary.csv",
                run_dir / "tables" / "A_weight_global_summary.csv",
                run_dir / "tables" / "B_quant_layer_summary.csv",
                run_dir / "tables" / "B_quant_block4_summary.csv",
                run_dir / "tables" / "B_quant_global_summary.csv",
                run_dir / "tables" / "B_quant_deltas.csv",
            ]
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"

            for path in stale_table_paths:
                self.assertTrue(path.exists(), f"Expected first successful build to write {path.name}")
            self.assertTrue(manifest_path.exists(), "Expected first successful build to write tables manifest")

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.15,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        # CONTRACT: on reruns, duplicate delta-key validation must
                        # clear any stale prior tables/manifest instead of leaving a
                        # previous successful build visible to downstream tooling.
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.31,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.09,
                        "w_rel_max": 0.12,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertIn("scheme_a", output)
            self.assertIn("down_proj", output)
            for path in stale_table_paths:
                self.assertFalse(
                    path.exists(),
                    f"Duplicate delta-key reruns should clear stale table output {path.name}",
                )
            self.assertFalse(
                manifest_path.exists(),
                "Duplicate delta-key reruns should clear stale tables manifest",
            )

    def test_build_tables_missing_input_rerun_preserves_previous_success_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            preserved_table_paths = [
                run_dir / "tables" / "A_weight_layer_summary.csv",
                run_dir / "tables" / "A_weight_block4_summary.csv",
                run_dir / "tables" / "A_weight_global_summary.csv",
                run_dir / "tables" / "B_quant_layer_summary.csv",
                run_dir / "tables" / "B_quant_block4_summary.csv",
                run_dir / "tables" / "B_quant_global_summary.csv",
                run_dir / "tables" / "B_quant_deltas.csv",
            ]
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            first_manifest = manifest_path.read_text()

            for path in preserved_table_paths:
                self.assertTrue(path.exists(), f"Expected first successful build to write {path.name}")
            self.assertTrue(manifest_path.exists(), "Expected first successful build to write tables manifest")

            (data_dir / "quant_sim.csv").unlink()

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            # CONTRACT: a rerun that fails before new tables are validated/written
            # must not wipe the previous successful table set.
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("quant_sim", output)
            for path in preserved_table_paths:
                self.assertTrue(
                    path.exists(),
                    f"Input-read failures should preserve previous successful table output {path.name}",
                )
            self.assertTrue(
                manifest_path.exists(),
                "Input-read failures should preserve previous successful tables manifest",
            )
            self.assertEqual(
                manifest_path.read_text(),
                first_manifest,
                "Input-read failures should leave the previous tables manifest unchanged",
            )

    def test_build_tables_duplicate_delta_cleanup_preserves_unrelated_table_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            custom_sidecar = run_dir / "tables" / "custom" / "manual_notes.csv"
            custom_sidecar.parent.mkdir(parents=True, exist_ok=True)
            custom_sidecar.write_text("note,value\ncustom,1\n")

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.15,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        # CONTRACT: duplicate-delta cleanup should remove only
                        # build_tables-owned outputs, not unrelated sidecars in
                        # tables/ created by other tooling or manual workflows.
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.31,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.09,
                        "w_rel_max": 0.12,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertFalse(
                (run_dir / "tables" / "B_quant_deltas.csv").exists(),
                "Duplicate-delta cleanup should still clear stale canonical outputs",
            )
            self.assertFalse(
                (run_dir / "logs" / "tables_write_manifest.json").exists(),
                "Duplicate-delta cleanup should still clear the stale tables manifest",
            )
            self.assertTrue(
                custom_sidecar.exists(),
                "Duplicate-delta cleanup should preserve unrelated table sidecars",
            )
            self.assertEqual(
                custom_sidecar.read_text(),
                "note,value\ncustom,1\n",
                "Duplicate-delta cleanup should not rewrite unrelated table sidecars",
            )

    def test_build_tables_duplicate_delta_cleanup_ignores_poisoned_manifest_path_for_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            custom_sidecar = run_dir / "tables" / "custom" / "manual_notes.csv"
            custom_sidecar.parent.mkdir(parents=True, exist_ok=True)
            custom_sidecar.write_text("note,value\ncustom,1\n")

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["B_quant_deltas"]["path"] = "tables/custom/manual_notes.csv"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.15,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        # CONTRACT: stale/corrupt tables manifests must not be
                        # able to redirect duplicate-delta cleanup onto unrelated
                        # sidecars under tables/.
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.31,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.09,
                        "w_rel_max": 0.12,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertFalse(
                (run_dir / "tables" / "B_quant_deltas.csv").exists(),
                "Duplicate-delta cleanup should still clear stale canonical outputs",
            )
            self.assertFalse(
                manifest_path.exists(),
                "Duplicate-delta cleanup should still clear the stale tables manifest",
            )
            self.assertTrue(
                custom_sidecar.exists(),
                "Poisoned manifest paths should not let duplicate-delta cleanup delete unrelated sidecars",
            )
            self.assertEqual(
                custom_sidecar.read_text(),
                "note,value\ncustom,1\n",
                "Poisoned manifest paths should not rewrite unrelated sidecars",
            )

    def test_build_tables_duplicate_delta_cleanup_handles_malformed_owned_directory_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                output_format="csv",
                compression=None,
                delta_pairs=[
                    {"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"},
                ],
            )
            self._write_matrix_stats_csv(data_dir, self._sample_matrix_rows())
            self._write_quant_sim_csv(data_dir, self._sample_quant_rows())

            self._run_build_tables(run_dir)

            malformed_owned_path = run_dir / "tables" / "B_quant_deltas.csv"
            malformed_owned_path.unlink()
            malformed_owned_path.mkdir()

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.15,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        # CONTRACT: duplicate-delta cleanup must tolerate malformed
                        # stale owned paths like a directory at a canonical table
                        # filename, and still surface the duplicate-key validation
                        # error instead of masking it with a cleanup exception.
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.31,
                        "w_rel_max": 0.44,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.09,
                        "w_rel_max": 0.12,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("duplicate", output.lower())
            self.assertNotIn("PermissionError", output)
            self.assertFalse(
                malformed_owned_path.exists(),
                "Duplicate-delta cleanup should remove malformed owned directory paths",
            )
            self.assertFalse(
                (run_dir / "tables" / "A_weight_layer_summary.csv").exists(),
                "Duplicate-delta cleanup should still clear earlier stale canonical outputs",
            )
            self.assertFalse(
                (run_dir / "tables" / "B_quant_global_summary.csv").exists(),
                "Duplicate-delta cleanup should still clear later stale canonical outputs",
            )
            self.assertFalse(
                (run_dir / "logs" / "tables_write_manifest.json").exists(),
                "Duplicate-delta cleanup should still clear the stale tables manifest",
            )

    def test_build_tables_quant_error_rows_are_represented_in_b_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)
            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 1.0,
                        "std": 0.1,
                        "mean_abs": 1.0,
                        "rms": 1.0,
                        "max_abs": 1.2,
                        "p50_abs": 1.0,
                        "p99_abs": 1.2,
                        "p999_abs": 1.2,
                        "outlier_max_over_mean": 1.2,
                        "outlier_p99_over_median": 1.2,
                        "outlier_p999_over_median": 1.2,
                    }
                ],
            )
            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_ok",
                        "w_rel_fro": 0.1,
                        "w_rel_max": 0.2,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.1.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 1,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_err",
                        "w_rel_fro": None,
                        "w_rel_max": None,
                        "scale_mean": None,
                        "scale_max": None,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": "RuntimeError: stub quantize fail",
                    },
                ],
            )

            self._run_build_tables(run_dir)

            _, layer_rows = self._read_csv(run_dir / "tables" / "B_quant_layer_summary.csv")
            self.assertEqual(len(layer_rows), 2)
            layer_by_scheme = {row["scheme"]: row for row in layer_rows}
            self.assertEqual(sorted(layer_by_scheme), ["scheme_err", "scheme_ok"])

            layer_ok = layer_by_scheme["scheme_ok"]
            self.assertAlmostEqual(float(layer_ok["w_rel_fro__median"]), 0.1)
            self.assertAlmostEqual(float(layer_ok["w_rel_max__median"]), 0.2)

            layer_err = layer_by_scheme["scheme_err"]
            self.assertEqual(layer_err["w_rel_fro__median"], "")
            self.assertEqual(layer_err["w_rel_max__median"], "")
            self.assertEqual(layer_err["scale_mean__median"], "")

            _, block4_rows = self._read_csv(run_dir / "tables" / "B_quant_block4_summary.csv")
            self.assertEqual(len(block4_rows), 2)
            block4_by_scheme = {row["scheme"]: row for row in block4_rows}
            self.assertEqual(sorted(block4_by_scheme), ["scheme_err", "scheme_ok"])

            block4_ok = block4_by_scheme["scheme_ok"]
            self.assertAlmostEqual(float(block4_ok["w_rel_fro__median"]), 0.1)
            self.assertAlmostEqual(float(block4_ok["w_rel_max__median"]), 0.2)

            block4_err = block4_by_scheme["scheme_err"]
            self.assertEqual(block4_err["w_rel_fro__median"], "")
            self.assertEqual(block4_err["w_rel_max__median"], "")
            self.assertEqual(block4_err["scale_mean__median"], "")

            _, rows = self._read_csv(run_dir / "tables" / "B_quant_global_summary.csv")
            self.assertEqual(len(rows), 2)
            by_scheme = {row["scheme"]: row for row in rows}
            self.assertEqual(sorted(by_scheme), ["scheme_err", "scheme_ok"])

            ok = by_scheme["scheme_ok"]
            self.assertAlmostEqual(float(ok["w_rel_fro__median"]), 0.1)
            self.assertAlmostEqual(float(ok["w_rel_max__median"]), 0.2)

            err = by_scheme["scheme_err"]
            self.assertEqual(err["w_rel_fro__median"], "")
            self.assertEqual(err["w_rel_max__median"], "")
            self.assertEqual(err["scale_mean__median"], "")

    def test_build_tables_layer_and_scheme_row_order_is_deterministic_for_plotting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 1,
                        "proj": "z_proj",
                        "mean": 1.0,
                        "std": 0.1,
                        "mean_abs": 1.0,
                        "rms": 1.0,
                        "max_abs": 1.2,
                        "p50_abs": 1.0,
                        "p99_abs": 1.2,
                        "p999_abs": 1.2,
                        "outlier_max_over_mean": 1.2,
                        "outlier_p99_over_median": 1.2,
                        "outlier_p999_over_median": 1.2,
                    },
                    {
                        "layer": 0,
                        "proj": "b_proj",
                        "mean": 2.0,
                        "std": 0.2,
                        "mean_abs": 2.0,
                        "rms": 2.0,
                        "max_abs": 2.2,
                        "p50_abs": 2.0,
                        "p99_abs": 2.2,
                        "p999_abs": 2.2,
                        "outlier_max_over_mean": 1.1,
                        "outlier_p99_over_median": 1.1,
                        "outlier_p999_over_median": 1.1,
                    },
                    {
                        "layer": 0,
                        "proj": "a_proj",
                        "mean": 3.0,
                        "std": 0.3,
                        "mean_abs": 3.0,
                        "rms": 3.0,
                        "max_abs": 3.3,
                        "p50_abs": 3.0,
                        "p99_abs": 3.3,
                        "p999_abs": 3.3,
                        "outlier_max_over_mean": 1.1,
                        "outlier_p99_over_median": 1.1,
                        "outlier_p999_over_median": 1.1,
                    },
                ],
            )

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.1.experts.0.z_proj.weight",
                        "layer": 1,
                        "proj": "z_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.11,
                        "w_rel_max": 0.16,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.a_proj.weight",
                        "layer": 0,
                        "proj": "a_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.21,
                        "w_rel_max": 0.26,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.a_proj.weight",
                        "layer": 0,
                        "proj": "a_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.20,
                        "w_rel_max": 0.25,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.0.experts.0.b_proj.weight",
                        "layer": 0,
                        "proj": "b_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.30,
                        "w_rel_max": 0.35,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                ],
            )

            self._run_build_tables(run_dir)

            _, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            a_layer_order = [(int(row["layer"]), row["proj"]) for row in a_layer_rows]
            self.assertEqual(
                a_layer_order,
                [(0, "a_proj"), (0, "b_proj"), (1, "z_proj")],
            )

            _, b_layer_rows = self._read_csv(run_dir / "tables" / "B_quant_layer_summary.csv")
            b_layer_order = [(int(row["layer"]), row["proj"], row["scheme"]) for row in b_layer_rows]
            self.assertEqual(
                b_layer_order,
                [
                    (0, "a_proj", "scheme_a"),
                    (0, "a_proj", "scheme_b"),
                    (0, "b_proj", "scheme_a"),
                    (1, "z_proj", "scheme_b"),
                ],
            )

    def test_build_tables_axis_keys_remain_parseable_with_missing_quant_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, output_format="csv", compression=None)

            self._write_matrix_stats_csv(
                data_dir,
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 1.0,
                        "std": 0.1,
                        "mean_abs": 1.0,
                        "rms": 1.0,
                        "max_abs": 1.2,
                        "p50_abs": 1.0,
                        "p99_abs": 1.2,
                        "p999_abs": 1.2,
                        "outlier_max_over_mean": 1.2,
                        "outlier_p99_over_median": 1.2,
                        "outlier_p999_over_median": 1.2,
                    },
                    {
                        "layer": 4,
                        "proj": "down_proj",
                        "mean": 2.0,
                        "std": 0.2,
                        "mean_abs": 2.0,
                        "rms": 2.0,
                        "max_abs": 2.2,
                        "p50_abs": 2.0,
                        "p99_abs": 2.2,
                        "p999_abs": 2.2,
                        "outlier_max_over_mean": 1.1,
                        "outlier_p99_over_median": 1.1,
                        "outlier_p999_over_median": 1.1,
                    },
                ],
            )

            self._write_quant_sim_csv(
                data_dir,
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_ok",
                        "w_rel_fro": 0.10,
                        "w_rel_max": 0.20,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                        "error": "",
                    },
                    {
                        "derived_tensor": "layers.4.experts.0.down_proj.weight",
                        "layer": 4,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_err",
                        "w_rel_fro": None,
                        "w_rel_max": None,
                        "scale_mean": None,
                        "scale_max": None,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": "RuntimeError: stub quantize fail",
                    },
                ],
            )

            self._run_build_tables(run_dir)

            _, a_layer_rows = self._read_csv(run_dir / "tables" / "A_weight_layer_summary.csv")
            parsed_a_layers = [int(row["layer"]) for row in a_layer_rows]
            self.assertEqual(sorted(parsed_a_layers), [0, 4])

            _, a_block4_rows = self._read_csv(run_dir / "tables" / "A_weight_block4_summary.csv")
            parsed_a_block4 = [int(row["block4"]) for row in a_block4_rows]
            self.assertEqual(sorted(parsed_a_block4), [0, 1])

            _, b_layer_rows = self._read_csv(run_dir / "tables" / "B_quant_layer_summary.csv")
            parsed_b_layers = [int(row["layer"]) for row in b_layer_rows]
            self.assertEqual(sorted(parsed_b_layers), [0, 4])
            by_scheme = {row["scheme"]: row for row in b_layer_rows}
            self.assertEqual(by_scheme["scheme_err"]["w_rel_fro__median"], "")
            self.assertEqual(by_scheme["scheme_err"]["w_rel_max__median"], "")

            _, b_block4_rows = self._read_csv(run_dir / "tables" / "B_quant_block4_summary.csv")
            parsed_b_block4 = [int(row["block4"]) for row in b_block4_rows]
            self.assertEqual(sorted(parsed_b_block4), [0, 1])
            block4_by_scheme = {row["scheme"]: row for row in b_block4_rows}
            self.assertEqual(block4_by_scheme["scheme_err"]["w_rel_fro__median"], "")
            self.assertEqual(block4_by_scheme["scheme_err"]["w_rel_max__median"], "")


if __name__ == "__main__":
    unittest.main()
