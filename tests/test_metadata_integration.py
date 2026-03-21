import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


def _load_init_run(repo_root: Path):
    path = repo_root / "scripts" / "init_run.py"
    spec = importlib.util.spec_from_file_location("init_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load init_run module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetadataInitRunTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_init_run_writes_metadata_section(self):
        init_run = _load_init_run(self.repo_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = init_run.init_run(tmp_path, "model", "run", None)
            cfg = json.loads((run_dir / "analysis_config.json").read_text())

            self.assertIn("metadata", cfg)
            self.assertEqual(cfg["metadata"].get("enabled"), True)
            self.assertEqual(cfg["metadata"].get("mode"), "validate")
            self.assertIsNone(cfg["metadata"].get("config_path"))

    def test_init_run_writes_scan_index_defaults(self):
        init_run = _load_init_run(self.repo_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = init_run.init_run(tmp_path, "model", "run", None)
            cfg = json.loads((run_dir / "analysis_config.json").read_text())

            scan_cfg = cfg.get("scan", {})
            self.assertEqual(scan_cfg.get("use_safetensors_index_json"), True)
            self.assertEqual(scan_cfg.get("strict_index"), False)

    def test_init_run_writes_quant_compute_dtype_default(self):
        init_run = _load_init_run(self.repo_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = init_run.init_run(tmp_path, "model", "run", None)
            cfg = json.loads((run_dir / "analysis_config.json").read_text())

            stats_cfg = cfg.get("stats", {})
            self.assertEqual(stats_cfg.get("quant_compute_dtype"), "bf16")

    def test_init_run_writes_quant_rel_den_floor_default(self):
        init_run = _load_init_run(self.repo_root)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = init_run.init_run(tmp_path, "model", "run", None)
            cfg = json.loads((run_dir / "analysis_config.json").read_text())

            stats_cfg = cfg.get("stats", {})
            self.assertEqual(stats_cfg.get("quant_rel_den_floor"), 1.0)


class CollectDataMetadataTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _write_npz_with_key(self, path: Path, key: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{key}.npy", buf.getvalue())

    def _run(self, args, env=None):
        return subprocess.run(
            args,
            cwd=self.repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _setup_run(self, tmp_path: Path, model_dir: Path) -> Path:
        run_root = tmp_path / "runs"
        run_root.mkdir(parents=True, exist_ok=True)

        self._run([
            sys.executable,
            str(self.repo_root / "scripts" / "init_run.py"),
            "--root",
            str(run_root),
            "--model-id",
            "model",
            "--run-name",
            "run",
            "--model-path",
            str(model_dir),
        ])

        return run_root / "model" / "run"

    def _configure_run(self, run_dir: Path, metadata_enabled: bool = True) -> None:
        cfg_path = run_dir / "analysis_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["output"]["format"] = "csv"
        cfg["output"]["compression"] = None
        cfg["mlx"]["enabled"] = False
        cfg["metadata"] = {
            "enabled": metadata_enabled,
            "mode": "validate",
            "config_path": None,
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))

    def test_collect_data_writes_metadata_logs_when_config_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 6,
            }
            (model_dir / "config.json").write_text(json.dumps(cfg, indent=2))

            arr = np.arange(16, dtype=np.float32).reshape(4, 4)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                arr,
            )

            run_dir = self._setup_run(tmp_path, model_dir)
            self._configure_run(run_dir, metadata_enabled=True)

            result = self._run([
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ])

            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("[meta]", output)

            budget_path = run_dir / "logs" / "model_shape_budget.json"
            self.assertTrue(budget_path.exists())
            budget = json.loads(budget_path.read_text())
            self.assertEqual(
                Path(budget["config_path"]).resolve(),
                (model_dir / "config.json").resolve(),
            )
            self.assertEqual(budget["shape_budget"]["hidden_size"], 16)
            self.assertEqual(budget["shape_budget"]["num_hidden_layers"], 2)
            self.assertEqual(budget["shape_budget"]["num_experts"], 4)

            raw_path = run_dir / "logs" / "model_config.raw.json"
            self.assertTrue(raw_path.exists())
            raw_cfg = json.loads(raw_path.read_text())
            self.assertEqual(raw_cfg["hidden_size"], 16)

    def test_collect_data_missing_config_logs_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            arr = np.arange(16, dtype=np.float32).reshape(4, 4)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                arr,
            )

            run_dir = self._setup_run(tmp_path, model_dir)
            self._configure_run(run_dir, metadata_enabled=True)

            self._run([
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ])

            warnings_path = run_dir / "logs" / "warnings.csv"
            self.assertTrue(warnings_path.exists())

            with warnings_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertTrue(any("meta" in row["warning"].lower() for row in rows))

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
