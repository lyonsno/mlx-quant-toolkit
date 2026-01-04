import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_STUB_DIR = None


def _ensure_stub_mlx():
    global _STUB_DIR
    if _STUB_DIR is not None:
        return
    tmp = tempfile.TemporaryDirectory()
    stub_root = Path(tmp.name) / "mlx"
    stub_root.mkdir(parents=True, exist_ok=True)
    (stub_root / "__init__.py").write_text("")
    (stub_root / "core.py").write_text("raise ImportError('stub mlx not available')\n")
    sys.path.insert(0, tmp.name)
    _STUB_DIR = tmp


def _load_collect_data():
    _ensure_stub_mlx()
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "collect_data.py"
    spec = importlib.util.spec_from_file_location("collect_data", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Unable to load collect_data module")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CanonicalizeLayoutTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_2d_identity_layout(self):
        arr = np.arange(6, dtype=np.int32).reshape(2, 3)
        layout = {"rows_axis": 0, "cols_axis": 1}
        canon = self.collect_data._canonicalize_layout(arr, layout)
        self.assertEqual(canon.shape, arr.shape)
        np.testing.assert_array_equal(canon, arr)

    def test_3d_expert_rows_cols_reorder(self):
        experts, rows, cols = 2, 3, 4
        base = np.arange(experts * rows * cols, dtype=np.int32).reshape(
            experts, rows, cols
        )
        permuted = np.transpose(base, (1, 0, 2))  # (R, E, C)
        layout = {"expert_axis": 1, "rows_axis": 0, "cols_axis": 2}
        canon = self.collect_data._canonicalize_layout(permuted, layout)
        self.assertEqual(canon.shape, base.shape)
        np.testing.assert_array_equal(canon, base)

    def test_4d_layer_expert_rows_cols_reorder(self):
        layers, experts, rows, cols = 2, 2, 3, 4
        base = np.arange(layers * experts * rows * cols, dtype=np.int32).reshape(
            layers, experts, rows, cols
        )
        permuted = np.transpose(base, (2, 0, 1, 3))  # (R, L, E, C)
        layout = {"layer_axis": 1, "expert_axis": 2, "rows_axis": 0, "cols_axis": 3}
        canon = self.collect_data._canonicalize_layout(permuted, layout)
        self.assertEqual(canon.shape, base.shape)
        np.testing.assert_array_equal(canon, base)

    def test_layout_axes_do_not_cover_ndim(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        layout = {"rows_axis": 0, "cols_axis": 1}
        with self.assertRaisesRegex(ValueError, "do not cover ndim=3"):
            self.collect_data._canonicalize_layout(arr, layout)


if __name__ == "__main__":
    unittest.main()
