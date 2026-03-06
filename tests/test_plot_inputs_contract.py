import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotInputsContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def _load_plot_inputs_module(self):
        module_path = self.scripts_dir / "plot_inputs.py"
        self.assertTrue(module_path.exists(), f"Expected helper module file is missing: {module_path}")
        return _load_module("plot_inputs", module_path)

    def test_plot_inputs_module_exports_axis_normalizer(self):
        mod = self._load_plot_inputs_module()
        self.assertTrue(
            hasattr(mod, "normalize_plot_axis_columns"),
            "Expected symbol normalize_plot_axis_columns",
        )
        self.assertTrue(callable(mod.normalize_plot_axis_columns))
        self.assertTrue(
            hasattr(mod, "load_plot_tables"),
            "Expected symbol load_plot_tables",
        )
        self.assertTrue(callable(mod.load_plot_tables))

    def test_normalize_plot_axis_columns_coerces_axis_keys_to_nullable_int(self):
        mod = self._load_plot_inputs_module()
        raw = pd.DataFrame(
            {
                "layer": ["0", "4", "", None],
                "block4": [0.0, "1", "", None],
                "scheme": ["a", "b", "a", "b"],
                "w_rel_fro__median": [0.1, 0.2, 0.3, 0.4],
            }
        )

        normalized = mod.normalize_plot_axis_columns(raw, axis_columns=("layer", "block4"))

        # Contract: do not mutate caller-owned frame in place.
        self.assertEqual(str(raw["layer"].dtype), "object")
        self.assertEqual(str(raw["block4"].dtype), "object")

        # Contract: plotting axis columns are normalized to nullable integer dtype.
        self.assertEqual(str(normalized["layer"].dtype), "Int64")
        self.assertEqual(str(normalized["block4"].dtype), "Int64")

        self.assertEqual(list(normalized["layer"][:2]), [0, 4])
        self.assertTrue(pd.isna(normalized["layer"][2]))
        self.assertTrue(pd.isna(normalized["layer"][3]))

        self.assertEqual(list(normalized["block4"][:2]), [0, 1])
        self.assertTrue(pd.isna(normalized["block4"][2]))
        self.assertTrue(pd.isna(normalized["block4"][3]))

        # Non-axis data should pass through unchanged.
        self.assertEqual(list(normalized["scheme"]), ["a", "b", "a", "b"])
        self.assertEqual(list(normalized["w_rel_fro__median"]), [0.1, 0.2, 0.3, 0.4])

    def test_normalize_plot_axis_columns_rejects_non_integer_tokens(self):
        mod = self._load_plot_inputs_module()
        raw = pd.DataFrame(
            {
                "layer": ["0", "oops"],
                "block4": ["0", "1"],
                "scheme": ["a", "b"],
            }
        )

        with self.assertRaisesRegex(ValueError, r"layer|oops"):
            mod.normalize_plot_axis_columns(raw, axis_columns=("layer", "block4"))

    def test_load_plot_tables_uses_manifest_discovery_and_normalizes_axis_columns(self):
        mod = self._load_plot_inputs_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            logs_dir = run_dir / "logs"
            tables_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            custom_table_path = tables_dir / "custom_layer_summary.csv"
            pd.DataFrame(
                {
                    "layer": ["0", "4", ""],
                    "block4": ["0", "1", ""],
                    "proj": ["a_proj", "a_proj", "a_proj"],
                    "mean__median": [0.1, 0.2, 0.3],
                }
            ).to_csv(custom_table_path, index=False)

            manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "A_weight_layer_summary": {
                        "path": "tables/custom_layer_summary.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 3,
                    }
                },
            }
            (logs_dir / "tables_write_manifest.json").write_text(json.dumps(manifest, indent=2))

            loaded = mod.load_plot_tables(run_dir, artifact_keys=("A_weight_layer_summary",))

            self.assertEqual(sorted(loaded.keys()), ["A_weight_layer_summary"])
            frame = loaded["A_weight_layer_summary"]
            self.assertEqual(str(frame["layer"].dtype), "Int64")
            self.assertEqual(str(frame["block4"].dtype), "Int64")
            self.assertEqual(list(frame["layer"][:2]), [0, 4])
            self.assertTrue(pd.isna(frame["layer"][2]))
            self.assertEqual(list(frame["block4"][:2]), [0, 1])
            self.assertTrue(pd.isna(frame["block4"][2]))
            self.assertEqual(list(frame["mean__median"]), [0.1, 0.2, 0.3])

    def test_load_plot_tables_surfaces_axis_normalization_errors(self):
        mod = self._load_plot_inputs_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            logs_dir = run_dir / "logs"
            tables_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            bad_table_path = tables_dir / "bad_layer_summary.csv"
            pd.DataFrame(
                {
                    "layer": ["0", "oops"],
                    "block4": ["0", "1"],
                    "proj": ["a_proj", "a_proj"],
                    "mean__median": [0.1, 0.2],
                }
            ).to_csv(bad_table_path, index=False)

            manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "A_weight_layer_summary": {
                        "path": "tables/bad_layer_summary.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 2,
                    }
                },
            }
            (logs_dir / "tables_write_manifest.json").write_text(json.dumps(manifest, indent=2))

            with self.assertRaisesRegex(ValueError, r"layer|oops"):
                mod.load_plot_tables(run_dir, artifact_keys=("A_weight_layer_summary",))

    def test_load_local_helper_module_cleans_sys_modules_on_exec_failure(self):
        mod = self._load_plot_inputs_module()
        module_name = "_missing_helper_for_plot_inputs_contract_test"
        prior = sys.modules.pop(module_name, None)
        try:
            with self.assertRaises(FileNotFoundError):
                mod._load_local_helper_module(module_name)
            self.assertNotIn(
                module_name,
                sys.modules,
                "failed helper load should not leave a poisoned sys.modules entry",
            )
        finally:
            sys.modules.pop(module_name, None)
            if prior is not None:
                sys.modules[module_name] = prior

    def test_load_local_helper_module_does_not_shadow_unrelated_loaded_module(self):
        mod = self._load_plot_inputs_module()
        module_name = "table_artifacts"
        prior = sys.modules.get(module_name)

        sentinel = types.ModuleType(module_name)
        sentinel.__file__ = str((self.repo_root / "not_the_local_helper.py").resolve())
        sys.modules[module_name] = sentinel

        try:
            loaded = mod._load_local_helper_module(module_name)
            self.assertIs(
                sys.modules.get(module_name),
                sentinel,
                "local helper loading should not overwrite unrelated preloaded module entries",
            )
            self.assertTrue(
                hasattr(loaded, "discover_table_artifacts"),
                "local helper module should still load successfully",
            )
            self.assertIsNot(loaded, sentinel)
        finally:
            current = sys.modules.get(module_name)
            if current is sentinel:
                if prior is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = prior
            elif prior is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = prior

    def test_load_plot_tables_falls_back_to_legacy_scan_when_manifest_entry_unavailable(self):
        mod = self._load_plot_inputs_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            logs_dir = run_dir / "logs"
            tables_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            legacy_table_path = tables_dir / "A_weight_layer_summary.csv"
            pd.DataFrame(
                {
                    "layer": ["0", "2", ""],
                    "block4": ["0", "0", ""],
                    "proj": ["a_proj", "a_proj", "a_proj"],
                    "mean__median": [1.0, 2.0, 3.0],
                }
            ).to_csv(legacy_table_path, index=False)

            # Manifest entry is unavailable to discovery (no usable path), forcing legacy scan fallback.
            manifest = {
                "generated_at": "2026-03-06T00:00:00Z",
                "requested_format": "csv",
                "requested_compression": None,
                "artifacts": {
                    "A_weight_layer_summary": {
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 3,
                    }
                },
            }
            (logs_dir / "tables_write_manifest.json").write_text(json.dumps(manifest, indent=2))

            loaded = mod.load_plot_tables(run_dir, artifact_keys=("A_weight_layer_summary",))

            self.assertEqual(sorted(loaded.keys()), ["A_weight_layer_summary"])
            frame = loaded["A_weight_layer_summary"]
            self.assertEqual(str(frame["layer"].dtype), "Int64")
            self.assertEqual(str(frame["block4"].dtype), "Int64")
            self.assertEqual(list(frame["layer"][:2]), [0, 2])
            self.assertTrue(pd.isna(frame["layer"][2]))
            self.assertEqual(list(frame["block4"][:2]), [0, 0])
            self.assertTrue(pd.isna(frame["block4"][2]))
            self.assertEqual(list(frame["mean__median"]), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
