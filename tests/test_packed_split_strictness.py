import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


class PackedSplitStrictnessTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _write_npz_with_key(self, path: Path, key: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{key}.npy", buf.getvalue())

    def _create_stub_mlx(self, root: Path) -> Path:
        stub_root = root / "stub_mlx"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text("raise ImportError('stub mlx not available')\n")
        return stub_root

    def _run_collect(self, run_dir: Path, env: dict, check: bool):
        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ],
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _write_config(self, run_dir: Path, model_dir: Path, strict_packed_split: bool) -> None:
        cfg = {
            "model_path": str(model_dir),
            "scan": {
                "extensions": [".npz"],
                "experts_only": True,
                "include_shared_expert": True,
                "inventory_all_tensors": True,
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
                "strict_packed_split": strict_packed_split,
            },
            "extract_rules": [
                {
                    "name": "packed_split_test",
                    "match": r".*experts.*\.(gate_proj)\.weight$",
                    "ndim": 3,
                    "layout": {"layer_axis": None, "expert_axis": 0, "rows_axis": 1, "cols_axis": 2},
                    "packed_split": {
                        "axis": "rows",
                        "splits": [3, 3],
                        "projs": ["gate_proj", "down_proj"],
                    },
                }
            ],
            "mlx": {"enabled": False, "device": "cpu"},
            "stats": {
                "eps": 1e-12,
                "sample_per_matrix": 8,
                "sample_seed": 123,
                "percentiles_abs": [50.0],
                "group_outlier_percentile": 95.0,
                "group_sizes_lastdim": [2],
            },
            "quant_schemes": [],
            "output": {"format": "csv", "compression": None},
            "debug": {"dump_unmatched_tensors": True, "print_progress_every_files": 0},
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def _setup_run(self, tmp_path: Path, strict_packed_split: bool) -> tuple[Path, dict]:
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        self._write_npz_with_key(
            model_dir / "weights.npz",
            "layers.0.experts.0.gate_proj.weight",
            arr,
        )

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(run_dir, model_dir, strict_packed_split)

        stub_root = self._create_stub_mlx(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"
        return run_dir, env

    def test_packed_split_mismatch_fails_when_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, env = self._setup_run(Path(tmp_dir), strict_packed_split=True)
            result = self._run_collect(run_dir, env, check=False)
            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("PackedSplitError", output)

    def test_packed_split_mismatch_warns_and_falls_back_when_non_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, env = self._setup_run(Path(tmp_dir), strict_packed_split=False)
            result = self._run_collect(run_dir, env, check=True)
            output = (result.stdout or "") + (result.stderr or "")

            warnings_path = run_dir / "logs" / "warnings.csv"
            self.assertTrue(warnings_path.exists())
            warnings_text = warnings_path.read_text()
            self.assertIn("packed_split failed", warnings_text)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            self.assertIn("matrix_stats rows", output)


if __name__ == "__main__":
    unittest.main()
