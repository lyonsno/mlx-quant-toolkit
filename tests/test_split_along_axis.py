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


class SplitAlongAxisTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_split_along_axis_valid(self):
        x = np.arange(12, dtype=np.int32).reshape(3, 4)
        parts = self.collect_data._split_along_axis(x, axis=1, splits=[1, 3])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].shape, (3, 1))
        self.assertEqual(parts[1].shape, (3, 3))
        np.testing.assert_array_equal(parts[0], x[:, :1])
        np.testing.assert_array_equal(parts[1], x[:, 1:])

    def test_split_along_axis_accepts_negative_axis(self):
        x = np.arange(12, dtype=np.int32).reshape(3, 4)
        parts = self.collect_data._split_along_axis(x, axis=-1, splits=[2, 2])
        self.assertEqual([p.shape for p in parts], [(3, 2), (3, 2)])
        np.testing.assert_array_equal(parts[0], x[:, :2])
        np.testing.assert_array_equal(parts[1], x[:, 2:])

    def test_split_along_axis_rejects_out_of_bounds_axis(self):
        x = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=2, splits=[1, 1])

    def test_split_along_axis_rejects_empty_splits(self):
        x = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[])

    def test_split_along_axis_rejects_non_integer_splits(self):
        x = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[1, 1.5])

    def test_split_along_axis_rejects_non_positive_splits(self):
        x = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[0, 2])
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[-1, 3])

    def test_split_along_axis_rejects_sum_mismatch(self):
        x = np.zeros((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[1, 1])
        with self.assertRaises(ValueError):
            self.collect_data._split_along_axis(x, axis=1, splits=[3, 2])


if __name__ == "__main__":
    unittest.main()
