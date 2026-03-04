import importlib.util
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


if __name__ == "__main__":
    unittest.main()
