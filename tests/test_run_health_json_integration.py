import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


class RunHealthJsonIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        return env

    def _run_collect(self, run_dir: Path, model_path: Path, env: dict, check: bool):
        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
                "--model-path",
                str(model_path),
            ],
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_collect_data_writes_run_health_json_with_summary_and_index_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t_rule = "layers.0.experts.0.up_proj.weight"
            t_fallback = "layers.0.experts.0.w3.weight"
            t_unmatched = "layers.0.experts.0.foo.weight"

            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "shard1.npz", **{t_rule: arr, t_fallback: arr + 1, t_unmatched: arr + 2})

            (model_dir / "poison.npz").write_bytes(b"not a real npz zip file")

            t_missing_tensor_same_shard = "layers.0.experts.123.up_proj.weight"
            t_missing_shard_tensor = "layers.0.experts.999.up_proj.weight"
            index_payload = {
                "weight_map": {
                    t_rule: "shard1.npz",
                    t_fallback: "shard1.npz",
                    t_missing_tensor_same_shard: "shard1.npz",
                    t_missing_shard_tensor: "shard2.npz",
                },
                "metadata": {"format": "npz-test"},
            }
            (model_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2))

            run_dir = tmp_path / "run"
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "data").mkdir(parents=True, exist_ok=True)

            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "test-model",
                        "run_name": "test-run",
                        "created_at": "2000-01-01T00:00:00Z",
                        "version": 2,
                    },
                    indent=2,
                )
            )

            cfg = {
                "model_path": str(tmp_path / "does_not_exist"),
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
                        "down_proj": ["down_proj", "w2"],
                        "gate_proj": ["gate_proj", "w1"],
                        "up_proj": ["up_proj", "w3"],
                    },
                    "shared_expert_keywords": ["shared", "expert"],
                    "strict_packed_split": True,
                    "proj_group_strict": False,
                },
                "extract_rules": [
                    {
                      "name": "test_rule_match_up_proj_2d",
                        "match": r".*experts\.(\d+)\.up_proj\.weight$",
                        "ndim": 2,
                        "layout": {"layer_axis": None, "expert_axis": None, "rows_axis": 0, "cols_axis": 1},
                    }
                ],
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

            self._run_collect(run_dir, model_dir, self._env(), check=True)

            health_path = run_dir / "logs" / "run_health.json"
            self.assertTrue(health_path.exists())

            health = json.loads(health_path.read_text())
            self.assertEqual(health.get("status"), "success")
            self.assertIn("generated_at", health)
            self.assertIn("duration_seconds", health)
            self.assertIn("run", health)
            self.assertIn("config_used", health)
            self.assertIn("scan_summary", health)
            self.assertIn("extraction_summary", health)
            self.assertIn("outputs_written", health)
            self.assertIn("tensor_name_examples", health)

            # Basic type checks to ensure this is a real JSON report rather than a stub.
            self.assertIsInstance(health["run"], dict)
            self.assertIsInstance(health["config_used"], dict)
            self.assertIsInstance(health["scan_summary"], dict)
            self.assertIsInstance(health["extraction_summary"], dict)
            self.assertIsInstance(health["outputs_written"], dict)
            self.assertIsInstance(health["tensor_name_examples"], dict)

            # Date/time should be parseable ISO, not a placeholder.
            gen = health["generated_at"]
            self.assertIsInstance(gen, str)
            datetime.fromisoformat(gen.replace("Z", "+00:00"))

            dur = health["duration_seconds"]
            self.assertIsInstance(dur, (int, float))
            self.assertGreater(dur, 0)

            run = health["run"]
            self.assertEqual(run.get("model_id"), "test-model")
            self.assertEqual(run.get("run_name"), "test-run")
            self.assertEqual(run.get("model_path"), str(model_dir.resolve()))

            scan_summary = health["scan_summary"]
            self.assertEqual(scan_summary.get("files_scanned"), 1)
            self.assertEqual(scan_summary.get("tensors_observed"), 3)

            extraction = health["extraction_summary"]
            self.assertEqual(extraction.get("extracted_by_rule"), 1)
            self.assertEqual(extraction.get("extracted_by_fallback"), 1)
            self.assertEqual(extraction.get("unmatched_expertish"), 1)

            outputs = health["outputs_written"]
            self.assertEqual(outputs.get("format"), "csv")
            self.assertEqual(outputs.get("tensor_inventory_rows"), 3)
            self.assertEqual(outputs.get("quant_sim_rows"), 0)
            self.assertEqual(outputs.get("unmatched_tensors_rows"), 1)

            examples = health["tensor_name_examples"]
            self.assertIn(t_rule, examples.get("rule_extracted", []))
            self.assertIn(t_fallback, examples.get("fallback_extracted", []))
            self.assertIn(t_unmatched, examples.get("unmatched_expertish", []))

            used = health["config_used"]
            self.assertEqual(used.get("model_path"), str(model_dir.resolve()))
            self.assertEqual(used.get("output", {}).get("format"), "csv")
            self.assertEqual(used.get("scan", {}).get("extensions"), [".npz"])

            index_summary = health.get("index_summary")
            self.assertIsInstance(index_summary, dict)
            self.assertTrue(index_summary.get("active"))
            self.assertEqual(index_summary.get("expected_shards_count"), 2)
            self.assertEqual(index_summary.get("scanned_shards_count"), 1)
            self.assertEqual(index_summary.get("missing_shards_count"), 1)
            self.assertEqual(index_summary.get("missing_tensors_count"), 2)
            self.assertEqual(index_summary.get("extra_tensors_count"), 1)

    def test_collect_data_run_health_counts_observed_tensors_even_with_duplicates_without_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "a_shard1.npz", **{t: arr})
            np.savez(model_dir / "b_shard2.npz", **{t: arr + 1})

            # Poison-pill file: should not be touched when max_files limits the scan.
            (model_dir / "z_poison.npz").write_bytes(b"not a real npz zip file")

            run_dir = tmp_path / "run"
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "test-model",
                        "run_name": "walk-mode",
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
                    "use_safetensors_index_json": False,
                    "strict_index": False,
                    "max_files": 2,
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

            self._run_collect(run_dir, model_dir, self._env(), check=True)

            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            self.assertEqual(health.get("status"), "success")

            scan_summary = health.get("scan_summary", {})
            self.assertEqual(scan_summary.get("files_scanned"), 2)
            # Both files contain a tensor with the same name; we still observed two tensors.
            self.assertEqual(scan_summary.get("tensors_observed"), 2)

            index_summary = health.get("index_summary", {})
            self.assertIsInstance(index_summary, dict)
            self.assertFalse(index_summary.get("active"))
