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


class _FakeNpz:
    def __init__(self, *, fail_on_key: str | None = None):
        self.files = ["a", "b"]
        self._arrays = {
            "a": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "b": np.array([5, 6, 7], dtype=np.int32),
        }
        self._fail_on_key = fail_on_key
        self.close_called = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getitem__(self, key: str) -> np.ndarray:
        if self._fail_on_key is not None and key == self._fail_on_key:
            raise RuntimeError("boom during array load")
        return self._arrays[key]

    def close(self):
        self.close_called = True


class IterTensorsFromFileNpzCloseTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_iter_tensors_from_file_npz_closes_after_iteration(self):
        original_np_load = self.collect_data.np.load
        created: dict[str, _FakeNpz] = {}

        def fake_np_load(path: str, allow_pickle: bool):
            self.assertFalse(allow_pickle)
            obj = _FakeNpz()
            created["obj"] = obj
            return obj

        self.collect_data.np.load = fake_np_load
        try:
            pairs = list(self.collect_data._iter_tensors_from_file(Path("fake_weights.npz")))
        finally:
            self.collect_data.np.load = original_np_load

        self.assertEqual([n for n, _ in pairs], ["a", "b"])
        np.testing.assert_array_equal(pairs[0][1], np.array([[1, 2], [3, 4]], dtype=np.int32))
        np.testing.assert_array_equal(pairs[1][1], np.array([5, 6, 7], dtype=np.int32))

        self.assertIn("obj", created)
        self.assertTrue(
            created["obj"].close_called,
            "Expected _iter_tensors_from_file to close the np.load() result for .npz inputs",
        )

    def test_iter_tensors_from_file_npz_closes_on_error(self):
        original_np_load = self.collect_data.np.load
        created: dict[str, _FakeNpz] = {}

        def fake_np_load(path: str, allow_pickle: bool):
            self.assertFalse(allow_pickle)
            obj = _FakeNpz(fail_on_key="b")
            created["obj"] = obj
            return obj

        self.collect_data.np.load = fake_np_load
        try:
            with self.assertRaisesRegex(RuntimeError, "boom during array load"):
                list(self.collect_data._iter_tensors_from_file(Path("fake_weights.npz")))
        finally:
            self.collect_data.np.load = original_np_load

        self.assertIn("obj", created)
        self.assertTrue(
            created["obj"].close_called,
            "Expected _iter_tensors_from_file to close the np.load() result even when iteration errors",
        )


if __name__ == "__main__":
    unittest.main()
