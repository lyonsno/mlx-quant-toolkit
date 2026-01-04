import csv
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
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
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load collect_data module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProjInferenceUnitTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_iter_weight_files_accepts_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            weights_path = tmp_path / "weights.npz"
            np.savez(weights_path, arr=np.zeros((1,), dtype=np.float32))

            files = list(
                self.collect_data._iter_weight_files(weights_path, exts=[".npz"])
            )

            self.assertEqual(files, [weights_path])

    def test_iter_weight_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            np.savez(tmp_path / "a.npz", arr=np.zeros((1,), dtype=np.float32))
            (tmp_path / "b.safetensors").write_bytes(b"")

            files = list(
                self.collect_data._iter_weight_files(tmp_path, exts=[".npz"])
            )

            self.assertEqual(set(files), {tmp_path / "a.npz"})

    def test_infer_proj_respects_aliases(self):
        alias_map = {
            "down_proj": ["w2"],
            "gate_proj": ["w1", ".gate."],
            "up_proj": ["w3"],
        }
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.w1.weight", alias_map
        )
        self.assertEqual(proj, "gate_proj")

    def test_infer_proj_does_not_match_substrings(self):
        alias_map = {
            "down_proj": ["w2"],
            "gate_proj": ["w1", ".gate."],
            "up_proj": ["w3"],
        }
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.w13.weight", alias_map
        )
        self.assertIsNone(proj)

    def test_infer_proj_keeps_sentinel_contains(self):
        alias_map = {"gate_proj": [".gate."]}
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.ffn.gate.weight", alias_map
        )
        self.assertEqual(proj, "gate_proj")


class ProjGroupNormalizationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _write_npz(self, path: Path, tensors: dict[str, np.ndarray]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for key, arr in tensors.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{key}.npy", buf.getvalue())

    def _create_stub_mlx(self, root: Path) -> Path:
        stub_root = root / "stub_mlx"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text("raise ImportError('stub mlx not available')\n")
        return stub_root

    def _write_config(
        self,
        run_dir: Path,
        model_dir: Path,
        rule: dict,
        proj_aliases: dict[str, list[str]],
        proj_group_strict: bool,
    ) -> None:
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
                "layer_regex": r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)",
                "expert_regex": r"(?:^|\\.)experts\\.(\\d+)(?:\\.|$)",
                "proj_aliases": proj_aliases,
                "shared_expert_keywords": ["shared", "expert"],
                "strict_packed_split": True,
                "proj_group_strict": proj_group_strict,
            },
            "extract_rules": [rule],
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
            "debug": {"dump_unmatched_tensors": True, "print_progress_every_files": 0},
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def _run_collect(self, run_dir: Path, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ],
            cwd=self.repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _setup_run(
        self,
        tmp_path: Path,
        tensors: dict[str, np.ndarray],
        rule: dict,
        proj_aliases: dict[str, list[str]],
        proj_group_strict: bool,
    ) -> tuple[Path, dict]:
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._write_npz(model_dir / "weights.npz", tensors)

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(run_dir, model_dir, rule, proj_aliases, proj_group_strict)

        stub_root = self._create_stub_mlx(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"
        return run_dir, env

    def test_proj_group_canonicalizes_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensors = {
                "layers.0.experts.0.w1.weight": arr,
                "layers.0.experts.0.w2.weight": arr,
                "layers.0.experts.0.w3.weight": arr,
            }
            rule = {
                "name": "proj_group_w123",
                "match": r".*experts.*\.(w1|w2|w3)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2"],
                "gate_proj": ["w1"],
                "up_proj": ["w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path, tensors, rule, proj_aliases, proj_group_strict=False
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 6)
            allowed = {"gate_proj", "down_proj", "up_proj"}
            self.assertTrue(all(row["proj"] in allowed for row in rows))
            for row in rows:
                source = row["source_tensor"]
                if source.endswith(".w1.weight"):
                    self.assertEqual(row["proj"], "gate_proj")
                elif source.endswith(".w2.weight"):
                    self.assertEqual(row["proj"], "down_proj")
                elif source.endswith(".w3.weight"):
                    self.assertEqual(row["proj"], "up_proj")
                else:
                    self.fail(f"Unexpected tensor name: {source}")

    def test_proj_group_strict_unmatched_when_alias_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensor_name = "layers.0.experts.0.w999.weight"
            tensors = {tensor_name: arr}
            rule = {
                "name": "proj_group_w999",
                "match": r".*experts.*\.(w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2"],
                "gate_proj": ["w1"],
                "up_proj": ["w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path, tensors, rule, proj_aliases, proj_group_strict=True
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            match = next(
                (row for row in unmatched_rows if row["tensor_name"] == tensor_name),
                None,
            )
            self.assertIsNotNone(match)
            self.assertEqual(match["reason"], "no_rule_match_or_proj_infer")


if __name__ == "__main__":
    unittest.main()
