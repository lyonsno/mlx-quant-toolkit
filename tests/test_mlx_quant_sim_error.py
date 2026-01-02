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


class MlxQuantSimErrorTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_quant_sim_error_rows_include_exception_message(self):
        class StubMx:
            cpu = object()
            gpu = object()

            def set_default_device(self, _device):
                return None

            def array(self, x):
                return x

            def quantize(self, *_args, **_kwargs):
                raise RuntimeError("stub quantize fail")

        stub_mx = StubMx()
        original_mx = self.collect_data.mx
        self.collect_data.mx = stub_mx
        try:
            bank = np.zeros((2, 4, 4), dtype=np.float32)
            schemes = [
                {
                    "name": "s1",
                    "mode": "symmetric",
                    "bits": 4,
                    "group_size": 32,
                    "enabled": True,
                }
            ]
            cfg_stats = {"eps": 1e-12}
            df, _ = self.collect_data._mlx_quant_sim(
                bank, schemes, cfg_stats, device="cpu"
            )
        finally:
            self.collect_data.mx = original_mx

        self.assertGreaterEqual(len(df), 2)
        for err in df["error"].tolist():
            self.assertIn("stub quantize fail", str(err))


if __name__ == "__main__":
    unittest.main()
