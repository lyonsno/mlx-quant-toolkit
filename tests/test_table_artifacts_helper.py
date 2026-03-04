import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TablesArtifactHelperContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def _write_file(self, path: Path, content: str = "col\n1\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_table_artifacts_helper_module_exists_and_exports_expected_api(self):
        path = self.scripts_dir / "table_artifacts.py"
        self.assertTrue(path.exists(), f"Expected helper module file is missing: {path}")
        mod = _load_module("table_artifacts", path)

        self.assertTrue(
            hasattr(mod, "DEFAULT_TABLE_ARTIFACT_KEYS"),
            "Expected symbol DEFAULT_TABLE_ARTIFACT_KEYS",
        )
        self.assertTrue(
            hasattr(mod, "discover_table_artifacts"),
            "Expected symbol discover_table_artifacts",
        )
        self.assertTrue(callable(mod.discover_table_artifacts))
        self.assertEqual(
            mod.DEFAULT_TABLE_ARTIFACT_KEYS,
            [
                "A_weight_layer_summary",
                "A_weight_block4_summary",
                "A_weight_global_summary",
                "B_quant_layer_summary",
                "B_quant_block4_summary",
                "B_quant_global_summary",
                "B_quant_deltas",
            ],
        )
        self.assertEqual(
            len(mod.DEFAULT_TABLE_ARTIFACT_KEYS),
            len(set(mod.DEFAULT_TABLE_ARTIFACT_KEYS)),
            "DEFAULT_TABLE_ARTIFACT_KEYS should not contain duplicates",
        )

    def test_discover_table_artifacts_prefers_tables_manifest_metadata(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary", "B_quant_global_summary"]

            self._write_file(run_dir / "tables" / "A_weight_global_summary.csv", "col\n10\n")
            self._write_file(run_dir / "tables" / "A_weight_global_summary.parquet", "ignore\n")
            self._write_file(run_dir / "tables" / "B_quant_global_summary.parquet", "col\n20\n")

            manifest = {
                "generated_at": "2026-03-04T00:00:00+00:00",
                "requested_format": "parquet",
                "requested_compression": "invalid-codec",
                "artifacts": {
                    "A_weight_global_summary": {
                        "path": "tables/A_weight_global_summary.csv",
                        "format": "csv",
                        "fallback": True,
                        "error": "ValueError: invalid codec",
                        "rows": 1,
                    },
                    "B_quant_global_summary": {
                        "path": "tables/B_quant_global_summary.parquet",
                        "format": "parquet",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                },
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)

            self.assertEqual(sorted(discovered), sorted(expected))
            self.assertEqual(
                discovered["A_weight_global_summary"]["path"],
                "tables/A_weight_global_summary.csv",
            )
            self.assertEqual(discovered["A_weight_global_summary"]["format"], "csv")
            self.assertTrue(discovered["A_weight_global_summary"]["fallback"])
            self.assertEqual(
                discovered["B_quant_global_summary"]["path"],
                "tables/B_quant_global_summary.parquet",
            )
            self.assertEqual(discovered["B_quant_global_summary"]["source"], "manifest")

    def test_discover_table_artifacts_falls_back_to_legacy_scan_when_manifest_missing_or_invalid(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary", "B_quant_global_summary", "B_quant_deltas"]

            self._write_file(run_dir / "tables" / "A_weight_global_summary.parquet", "col\n1\n")
            self._write_file(run_dir / "tables" / "B_quant_global_summary.csv", "col\n2\n")
            self._write_file(run_dir / "tables" / "B_quant_deltas.csv", "col\n3\n")

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{not-json")

            discovered = mod.discover_table_artifacts(run_dir, expected)

            self.assertEqual(sorted(discovered), sorted(expected))
            self.assertEqual(
                discovered["A_weight_global_summary"]["path"],
                "tables/A_weight_global_summary.parquet",
            )
            self.assertEqual(
                discovered["B_quant_global_summary"]["path"],
                "tables/B_quant_global_summary.csv",
            )
            self.assertEqual(discovered["B_quant_deltas"]["format"], "csv")
            for key in expected:
                self.assertEqual(discovered[key]["source"], "legacy_scan")
                self.assertFalse(discovered[key]["fallback"])
                self.assertEqual(discovered[key]["error"], "")

    def test_discover_table_artifacts_mixes_manifest_entries_with_legacy_fallback_per_key(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary", "B_quant_global_summary", "B_quant_deltas"]

            self._write_file(run_dir / "tables" / "A_weight_global_summary.csv", "col\n1\n")
            self._write_file(run_dir / "tables" / "B_quant_global_summary.csv", "col\n2\n")
            # B_quant_deltas intentionally missing to prove graceful partial discovery.

            manifest = {
                "generated_at": "2026-03-04T00:00:00+00:00",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "A_weight_global_summary": {
                        "path": "tables/A_weight_global_summary.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    }
                },
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)

            self.assertEqual(
                sorted(discovered),
                sorted(["A_weight_global_summary", "B_quant_global_summary"]),
            )
            self.assertEqual(discovered["A_weight_global_summary"]["source"], "manifest")
            self.assertEqual(discovered["B_quant_global_summary"]["source"], "legacy_scan")

    def test_discover_table_artifacts_handles_non_numeric_rows_without_crashing(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary"]
            self._write_file(run_dir / "tables" / "A_weight_global_summary.csv", "col\n1\n")

            manifest = {
                "artifacts": {
                    "A_weight_global_summary": {
                        "path": "tables/A_weight_global_summary.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": "abc",
                    }
                }
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)

            self.assertEqual(sorted(discovered), expected)
            self.assertEqual(discovered["A_weight_global_summary"]["source"], "manifest")
            self.assertEqual(discovered["A_weight_global_summary"]["rows"], 0)

    def test_discover_table_artifacts_ignores_manifest_entry_without_path(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary"]

            manifest = {
                "artifacts": {
                    "A_weight_global_summary": {
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    }
                }
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)
            self.assertEqual(discovered, {})

    def test_discover_table_artifacts_ignores_manifest_paths_outside_run_dir(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / "run"
            expected = ["A_weight_global_summary", "B_quant_global_summary"]

            self._write_file(run_dir / "tables" / "A_weight_global_summary.csv", "col\n1\n")
            self._write_file(run_dir / "tables" / "B_quant_global_summary.csv", "col\n2\n")

            # Poison-pill files outside run_dir that should never be selected as discovered artifacts.
            absolute_poison = tmp_path / "outside_absolute.csv"
            traversal_poison = tmp_path / "outside_traversal.csv"
            self._write_file(absolute_poison, "col\n99\n")
            self._write_file(traversal_poison, "col\n88\n")

            manifest = {
                "artifacts": {
                    "A_weight_global_summary": {
                        "path": str(absolute_poison.resolve()),
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 99,
                    },
                    "B_quant_global_summary": {
                        "path": "../outside_traversal.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 88,
                    },
                }
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)
            self.assertEqual(sorted(discovered), sorted(expected))
            self.assertEqual(discovered["A_weight_global_summary"]["source"], "legacy_scan")
            self.assertEqual(discovered["A_weight_global_summary"]["path"], "tables/A_weight_global_summary.csv")
            self.assertEqual(discovered["B_quant_global_summary"]["source"], "legacy_scan")
            self.assertEqual(discovered["B_quant_global_summary"]["path"], "tables/B_quant_global_summary.csv")

    def test_discover_table_artifacts_rejects_manifest_non_table_targets(self):
        mod = _load_module("table_artifacts", self.scripts_dir / "table_artifacts.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            expected = ["A_weight_global_summary", "B_quant_global_summary", "B_quant_deltas"]

            # Valid legacy table artifacts that should be selected when manifest entries are invalid.
            self._write_file(run_dir / "tables" / "A_weight_global_summary.csv", "col\n1\n")
            self._write_file(run_dir / "tables" / "B_quant_global_summary.parquet", "col\n2\n")
            self._write_file(run_dir / "tables" / "B_quant_deltas.csv", "col\n3\n")

            # Poison-pill paths that exist but are not valid table artifact files.
            self._write_file(run_dir / "data" / "poison.csv", "col\n99\n")  # outside tables/
            self._write_file(run_dir / "tables" / "B_quant_deltas.json", '{"x": 1}\n')  # wrong extension
            (run_dir / "tables").mkdir(parents=True, exist_ok=True)  # directory path target

            manifest = {
                "artifacts": {
                    "A_weight_global_summary": {
                        "path": "data/poison.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 99,
                    },
                    "B_quant_global_summary": {
                        "path": "tables",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 0,
                    },
                    "B_quant_deltas": {
                        "path": "tables/B_quant_deltas.json",
                        "format": "json",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                }
            }
            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))

            discovered = mod.discover_table_artifacts(run_dir, expected)

            self.assertEqual(sorted(discovered), sorted(expected))
            self.assertEqual(discovered["A_weight_global_summary"]["source"], "legacy_scan")
            self.assertEqual(discovered["A_weight_global_summary"]["path"], "tables/A_weight_global_summary.csv")
            self.assertEqual(discovered["B_quant_global_summary"]["source"], "legacy_scan")
            self.assertEqual(
                discovered["B_quant_global_summary"]["path"],
                "tables/B_quant_global_summary.parquet",
            )
            self.assertEqual(discovered["B_quant_deltas"]["source"], "legacy_scan")
            self.assertEqual(discovered["B_quant_deltas"]["path"], "tables/B_quant_deltas.csv")


if __name__ == "__main__":
    unittest.main()
