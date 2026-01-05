import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


class SafetensorsIndexIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        return env

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

    def _write_config(self, run_dir: Path, model_dir: Path, use_index: bool, strict_index: bool) -> None:
        cfg = {
            "model_path": str(model_dir),
            "scan": {
                "extensions": [".safetensors"],
                "experts_only": True,
                "include_shared_expert": True,
                "inventory_all_tensors": True,
                "max_files": None,
                "use_safetensors_index_json": use_index,
                "strict_index": strict_index,
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

    def _write_safetensors(self, path: Path, tensors: dict[str, np.ndarray]) -> None:
        save_file(tensors, str(path))

    def _write_index(self, path: Path, weight_map: dict[str, str], metadata: dict | None = None) -> None:
        payload = {"weight_map": weight_map}
        if metadata is not None:
            payload["metadata"] = metadata
        path.write_text(json.dumps(payload, indent=2))

    def test_collect_data_uses_index_to_limit_scan_and_enrich_inventory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            t3 = "layers.0.experts.2.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "shard1.safetensors", {t1: arr})
            self._write_safetensors(model_dir / "shard2.safetensors", {t2: arr + 1})
            self._write_safetensors(model_dir / "extra.safetensors", {t3: arr + 2})

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: "shard1.safetensors", t2: "shard2.safetensors"},
                metadata={"format": "pt"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=False)

            self._run_collect(run_dir, self._env(), check=True)

            inv_path = run_dir / "data" / "tensor_inventory.csv"
            self.assertTrue(inv_path.exists())
            with inv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all("in_index" in row for row in rows))
            self.assertTrue(all("index_shard" in row for row in rows))

            names = {row["tensor_name"] for row in rows}
            self.assertEqual(names, {t1, t2})

            for row in rows:
                self.assertEqual(row["in_index"], "True")

            shard_map = {row["tensor_name"]: row["index_shard"] for row in rows}
            self.assertEqual(shard_map[t1], "shard1.safetensors")
            self.assertEqual(shard_map[t2], "shard2.safetensors")

            report_path = run_dir / "logs" / "index_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            self.assertEqual(report.get("missing_tensors"), [])
            self.assertEqual(report.get("extra_tensors"), [])
            self.assertEqual(report.get("missing_shards"), [])
            self.assertEqual(report.get("extra_shards"), [])

    def test_collect_data_missing_shard_warns_and_reports_when_non_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "shard1.safetensors", {t1: arr})

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: "shard1.safetensors", t2: "shard2.safetensors"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=False)

            self._run_collect(run_dir, self._env(), check=True)

            warnings_path = run_dir / "logs" / "warnings.csv"
            self.assertTrue(warnings_path.exists())
            warnings_text = warnings_path.read_text().lower()
            self.assertIn("index", warnings_text)
            self.assertIn("shard2.safetensors", warnings_text)

            report_path = run_dir / "logs" / "index_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            self.assertIn("shard2.safetensors", report.get("missing_shards", []))
            self.assertIn(t2, report.get("missing_tensors", []))

            inv_path = run_dir / "data" / "tensor_inventory.csv"
            with inv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tensor_name"], t1)
            self.assertEqual(rows[0]["in_index"], "True")
            self.assertEqual(rows[0]["index_shard"], "shard1.safetensors")

    def test_collect_data_missing_shard_errors_when_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "shard1.safetensors", {t1: arr})

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: "shard1.safetensors", t2: "shard2.safetensors"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=True)

            result = self._run_collect(run_dir, self._env(), check=False)
            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("shard2.safetensors", output)

    def test_collect_data_without_index_keeps_inventory_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "weights.safetensors", {t1: arr})

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=False)

            self._run_collect(run_dir, self._env(), check=True)

            inv_path = run_dir / "data" / "tensor_inventory.csv"
            with inv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                rows = list(reader)

            self.assertNotIn("in_index", fieldnames)
            self.assertNotIn("index_shard", fieldnames)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tensor_name"], t1)


if __name__ == "__main__":
    unittest.main()
