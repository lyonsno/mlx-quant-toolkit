import csv
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
    _REQUIRED_ARTIFACT_KEYS = {"path", "format", "fallback", "error", "rows"}

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _assert_required_keys_subset(
        self,
        payload: dict,
        required_keys: set[str],
        label: str,
    ) -> None:
        self.assertIsInstance(payload, dict)
        missing = sorted(set(required_keys) - set(payload))
        self.assertFalse(missing, f"{label} missing keys: {missing}")

    def _assert_artifact_entry(self, name: str, artifact: dict) -> None:
        self._assert_required_keys_subset(
            artifact,
            self._REQUIRED_ARTIFACT_KEYS,
            f"artifact {name}",
        )
        if artifact.get("fallback"):
            self.assertIsInstance(artifact.get("error"), str)
            self.assertTrue(artifact.get("error"))

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

    def _resolve_manifest_path(self, run_dir: Path, path_value: str | None) -> Path:
        self.assertIsNotNone(path_value)
        path = Path(path_value)
        if not path.is_absolute():
            path = run_dir / path
        return path

    def _csv_row_count(self, path: Path) -> int:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)

    def _assert_manifest_rows_match_csv(self, run_dir: Path, artifact: dict) -> None:
        if artifact.get("format") != "csv":
            return
        path = self._resolve_manifest_path(run_dir, artifact.get("path"))
        self.assertTrue(path.exists())
        self.assertEqual(self._csv_row_count(path), artifact.get("rows"))

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

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            self._assert_required_keys_subset(
                context,
                {"generated_at", "run", "model_path", "cli_overrides", "scan_plan", "index"},
                "run_context",
            )
            scan_plan = context.get("scan_plan", {})
            index_info = context.get("index", {})
            self._assert_required_keys_subset(
                index_info,
                {"status", "searched", "found", "active", "index_path"},
                "run_context.index",
            )
            if index_info.get("status") == "error":
                self.assertIsInstance(index_info.get("error"), str)
                self.assertTrue(index_info.get("error"))

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            self._assert_required_keys_subset(
                write_manifest,
                {"generated_at", "requested_format", "requested_compression", "artifacts"},
                "write_manifest",
            )
            artifacts = write_manifest.get("artifacts", {})
            for name, artifact in artifacts.items():
                self._assert_artifact_entry(name, artifact)

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
            self._assert_required_keys_subset(
                outputs,
                {
                    "format",
                    "tensor_inventory_rows",
                    "matrix_stats_rows",
                    "quant_sim_rows",
                    "wrote_warnings",
                    "wrote_unmatched_tensors",
                    "wrote_index_report",
                },
                "run_health.outputs_written",
            )
            self.assertEqual(outputs.get("format"), "csv")
            self.assertEqual(outputs.get("tensor_inventory_rows"), 3)
            self.assertEqual(outputs.get("matrix_stats_rows"), 2)
            self.assertEqual(outputs.get("quant_sim_rows"), 0)
            self.assertEqual(outputs.get("unmatched_tensors_rows"), 1)
            self.assertTrue(outputs.get("wrote_unmatched_tensors"))

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

            self.assertEqual(scan_summary.get("files_scanned"), scan_plan.get("scanned_files_count"))
            inv = artifacts.get("tensor_inventory", {})
            stats = artifacts.get("matrix_stats", {})
            quant = artifacts.get("quant_sim", {})
            self.assertEqual(outputs.get("tensor_inventory_rows"), inv.get("rows"))
            self.assertEqual(outputs.get("matrix_stats_rows"), stats.get("rows"))
            self.assertEqual(outputs.get("quant_sim_rows"), quant.get("rows"))
            self.assertEqual(outputs.get("wrote_warnings"), "warnings" in artifacts)
            if outputs.get("wrote_warnings"):
                warnings_meta = artifacts.get("warnings", {})
                warnings_path = self._resolve_manifest_path(run_dir, warnings_meta.get("path"))
                self.assertTrue(warnings_path.exists())
            index_report_path = run_dir / "logs" / "index_report.json"
            self.assertEqual(outputs.get("wrote_index_report"), index_report_path.exists())
            self.assertTrue(outputs.get("wrote_index_report"))
            self.assertTrue(index_report_path.exists())
            self._assert_manifest_rows_match_csv(run_dir, inv)
            self._assert_manifest_rows_match_csv(run_dir, stats)
            self._assert_manifest_rows_match_csv(run_dir, quant)

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
