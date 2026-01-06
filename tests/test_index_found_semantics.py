import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_STUB_DIR = None


def _ensure_stub_mlx() -> None:
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


class IndexFoundSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collect_data = _load_collect_data()

    def _write_npz(self, path: Path, key: str, arr: np.ndarray) -> None:
        np.savez(path, **{key: arr})

    def test_index_found_false_when_candidate_path_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            self._write_npz(model_dir / "shard1.npz", tensor_name, arr)

            run_dir = tmp_path / "run"
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "test-model",
                        "run_name": "index-candidate-missing",
                        "created_at": "2000-01-01T00:00:00Z",
                        "version": 2,
                    },
                    indent=2,
                )
            )

            cfg = {
                "model_path": str(model_dir),
                "scan": {
                    "extensions": [".npz"],
                    "experts_only": True,
                    "include_shared_expert": True,
                    "inventory_all_tensors": True,
                    "use_safetensors_index_json": True,
                    "strict_index": False,
                    "max_files": None,
                },
                "parsing": {
                    "layer_regex": r"(?:^|\.)layers\.(\d+)(?:\.|$)",
                    "expert_regex": r"(?:^|\.)experts\.(\d+)(?:\.|$)",
                    "proj_aliases": {
                        "down_proj": ["down_proj"],
                        "gate_proj": ["gate_proj"],
                        "up_proj": ["up_proj"],
                    },
                    "shared_expert_keywords": ["shared", "expert"],
                    "strict_packed_split": True,
                    "proj_group_strict": False,
                },
                "extract_rules": [],
                "metadata": {"enabled": False, "mode": "validate", "config_path": None},
                "mlx": {"enabled": False, "device": "cpu"},
                "stats": {
                    "eps": 1e-12,
                    "sample_per_matrix": 4,
                    "sample_seed": 123,
                    "percentiles_abs": [50.0],
                    "group_outlier_percentile": 95.0,
                    "group_sizes_lastdim": [2],
                },
                "quant_schemes": [],
                "output": {"format": "csv", "compression": None},
                "debug": {"dump_unmatched_tensors": False, "print_progress_every_files": 0},
            }
            (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

            class StubMeta:
                def find_safetensors_index_json(self, _model_path: Path) -> Path:
                    return _model_path / "model.safetensors.index.json"

                def parse_safetensors_index(self, _path: Path):
                    raise RuntimeError("parse should not be called for missing index")

            old_loaded = self.collect_data._METADATA_LOADED
            old_module = self.collect_data._METADATA_MODULE
            old_argv = sys.argv
            try:
                self.collect_data._METADATA_LOADED = True
                self.collect_data._METADATA_MODULE = StubMeta()
                sys.argv = ["collect_data.py", "--run-dir", str(run_dir)]
                self.collect_data.main()
            finally:
                sys.argv = old_argv
                self.collect_data._METADATA_LOADED = old_loaded
                self.collect_data._METADATA_MODULE = old_module

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            index_info = context.get("index", {})
            self.assertEqual(index_info.get("status"), "not_found")
            self.assertEqual(index_info.get("searched"), True)
            self.assertEqual(index_info.get("found"), False)
            self.assertEqual(index_info.get("active"), False)
            self.assertIsNone(index_info.get("index_path"))
