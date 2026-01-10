import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class AuditabilityArtifactsIntegrationTests(unittest.TestCase):
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

    def _resolve_manifest_path(self, run_dir: Path, path_value: str | None) -> Path:
        self.assertIsNotNone(path_value)
        path = Path(path_value)
        # Allow write manifests to store either absolute paths or run_dir-relative paths.
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

    def _assert_relative_path_example_endswith(self, value: object, suffix: str) -> None:
        if isinstance(value, list) and value:
            candidate = value[0]
        else:
            candidate = value
        self.assertIsInstance(candidate, str)
        self.assertFalse(Path(candidate).is_absolute())
        self.assertTrue(candidate.endswith(suffix))

    def _env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        return env

    def _run_collect(self, run_dir: Path, model_path: Path | None, env: dict, check: bool):
        cmd = [
            sys.executable,
            str(self.repo_root / "scripts" / "collect_data.py"),
            "--run-dir",
            str(run_dir),
        ]
        if model_path is not None:
            cmd += ["--model-path", str(model_path)]
        return subprocess.run(
            cmd,
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _init_run_dir(self, tmp_path: Path, run_name: str) -> Path:
        run_dir = tmp_path / run_name
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "data").mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_id": "test-model",
                    "run_name": run_name,
                    "created_at": "2000-01-01T00:00:00Z",
                    "version": 2,
                },
                indent=2,
            )
        )
        return run_dir

    def _write_config(
        self,
        run_dir: Path,
        configured_model_path: Path,
        *,
        use_index: bool,
        output_format: str,
        compression: str | None,
    ) -> None:
        cfg = {
            "model_path": str(configured_model_path),
            "scan": {
                "extensions": [".npz"],
                "experts_only": True,
                "include_shared_expert": True,
                "inventory_all_tensors": True,
                "use_safetensors_index_json": use_index,
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
            "output": {"format": output_format, "compression": compression},
            "debug": {"dump_unmatched_tensors": False, "print_progress_every_files": 0},
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def test_collect_data_writes_run_context_and_write_manifest_with_cli_override_and_index_active(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "shard1.npz", **{tensor_name: arr})

            # Poison-pill file: should not be touched when the index limits the scan.
            (model_dir / "poison.npz").write_bytes(b"not a real npz zip file")

            index_payload = {
                "weight_map": {tensor_name: "shard1.npz"},
                "metadata": {"format": "npz-test"},
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index_payload, indent=2)
            )

            run_dir = self._init_run_dir(tmp_path, "index-active")
            configured_model_path = tmp_path / "does_not_exist"
            self._write_config(
                run_dir,
                configured_model_path,
                use_index=True,
                output_format="parquet",
                compression="invalid-codec",
            )

            self._run_collect(run_dir, model_dir, self._env(), check=True)

            context_path = run_dir / "logs" / "run_context.json"
            self.assertTrue(context_path.exists())
            context = json.loads(context_path.read_text())
            self._assert_required_keys_subset(
                context,
                {"generated_at", "run", "model_path", "cli_overrides", "scan_plan", "index"},
                "run_context",
            )

            model_path_info = context.get("model_path", {})
            self.assertEqual(model_path_info.get("resolved"), str(model_dir.resolve()))
            self.assertEqual(model_path_info.get("configured"), str(configured_model_path.resolve()))
            self.assertEqual(model_path_info.get("source"), "cli_override")

            cli_overrides = context.get("cli_overrides", {})
            self.assertEqual(cli_overrides.get("model_path"), str(model_dir.resolve()))

            scan_plan = context.get("scan_plan", {})
            self.assertEqual(scan_plan.get("use_safetensors_index_json"), True)
            self.assertEqual(scan_plan.get("extensions"), [".npz"])
            self.assertEqual(scan_plan.get("scanned_files_count"), 1)
            self._assert_relative_path_example_endswith(
                scan_plan.get("scanned_files_example"),
                "shard1.npz",
            )

            index_info = context.get("index", {})
            self._assert_required_keys_subset(
                index_info,
                {"status", "searched", "found", "active", "index_path"},
                "run_context.index",
            )
            if index_info.get("status") == "error":
                self.assertIsInstance(index_info.get("error"), str)
                self.assertTrue(index_info.get("error"))
            self.assertEqual(index_info.get("status"), "active")
            self.assertEqual(index_info.get("searched"), True)
            self.assertEqual(index_info.get("found"), True)
            self.assertEqual(index_info.get("active"), True)
            self.assertEqual(
                Path(index_info.get("index_path")).resolve(),
                (model_dir / "model.safetensors.index.json").resolve(),
            )

            manifest_path = run_dir / "logs" / "write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            self._assert_required_keys_subset(
                write_manifest,
                {"generated_at", "requested_format", "requested_compression", "artifacts"},
                "write_manifest",
            )
            self.assertEqual(write_manifest.get("requested_format"), "parquet")
            self.assertEqual(write_manifest.get("requested_compression"), "invalid-codec")

            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("tensor_inventory", artifacts)
            for name, artifact in artifacts.items():
                self._assert_artifact_entry(name, artifact)

            inv = artifacts["tensor_inventory"]
            self.assertEqual(inv.get("format"), "csv")
            self.assertTrue(inv.get("fallback"))
            self.assertIsInstance(inv.get("error"), str)
            self.assertTrue(inv.get("error"))
            self.assertEqual(inv.get("rows"), 1)
            self.assertTrue(inv.get("path", "").endswith("tensor_inventory.csv"))
            self.assertTrue(self._resolve_manifest_path(run_dir, inv.get("path")).exists())
            self._assert_manifest_rows_match_csv(run_dir, inv)

            stats = artifacts.get("matrix_stats", {})
            self.assertEqual(stats.get("rows"), 1)
            self.assertTrue(self._resolve_manifest_path(run_dir, stats.get("path")).exists())
            self._assert_manifest_rows_match_csv(run_dir, stats)

            quant = artifacts.get("quant_sim", {})
            self.assertEqual(quant.get("rows"), 0)
            self.assertTrue(self._resolve_manifest_path(run_dir, quant.get("path")).exists())
            self._assert_manifest_rows_match_csv(run_dir, quant)

            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            scan_summary = health.get("scan_summary", {})
            outputs = health.get("outputs_written", {})
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
            self.assertEqual(scan_summary.get("files_scanned"), scan_plan.get("scanned_files_count"))
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

    def test_collect_data_run_context_logs_index_status_when_index_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "shard1.npz", **{tensor_name: arr})

            run_dir = self._init_run_dir(tmp_path, "index-missing")
            self._write_config(
                run_dir,
                model_dir,
                use_index=True,
                output_format="csv",
                compression=None,
            )

            self._run_collect(run_dir, None, self._env(), check=True)

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            self._assert_required_keys_subset(
                context,
                {"generated_at", "run", "model_path", "cli_overrides", "scan_plan", "index"},
                "run_context",
            )
            index_info = context.get("index", {})
            self._assert_required_keys_subset(
                index_info,
                {"status", "searched", "found", "active", "index_path"},
                "run_context.index",
            )
            self.assertEqual(index_info.get("status"), "not_found")
            self.assertEqual(index_info.get("searched"), True)
            self.assertEqual(index_info.get("found"), False)
            self.assertEqual(index_info.get("active"), False)
            self.assertIsNone(index_info.get("index_path"))

    def test_collect_data_run_context_logs_index_status_when_index_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "shard1.npz", **{tensor_name: arr})

            run_dir = self._init_run_dir(tmp_path, "index-disabled")
            self._write_config(
                run_dir,
                model_dir,
                use_index=False,
                output_format="csv",
                compression=None,
            )

            self._run_collect(run_dir, None, self._env(), check=True)

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            self._assert_required_keys_subset(
                context,
                {"generated_at", "run", "model_path", "cli_overrides", "scan_plan", "index"},
                "run_context",
            )
            index_info = context.get("index", {})
            self._assert_required_keys_subset(
                index_info,
                {"status", "searched", "found", "active", "index_path"},
                "run_context.index",
            )
            self.assertEqual(index_info.get("status"), "disabled")
            self.assertEqual(index_info.get("searched"), False)
            self.assertEqual(index_info.get("found"), False)
            self.assertEqual(index_info.get("active"), False)
            self.assertIsNone(index_info.get("index_path"))

    def test_collect_data_run_context_logs_index_status_when_index_parse_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)
            np.savez(model_dir / "shard1.npz", **{tensor_name: arr})

            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"format": "npz-test"}}, indent=2)
            )

            run_dir = self._init_run_dir(tmp_path, "index-parse-error")
            self._write_config(
                run_dir,
                model_dir,
                use_index=True,
                output_format="csv",
                compression=None,
            )

            self._run_collect(run_dir, None, self._env(), check=True)

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            self._assert_required_keys_subset(
                context,
                {"generated_at", "run", "model_path", "cli_overrides", "scan_plan", "index"},
                "run_context",
            )
            index_info = context.get("index", {})
            self._assert_required_keys_subset(
                index_info,
                {"status", "searched", "found", "active", "index_path"},
                "run_context.index",
            )
            self.assertEqual(index_info.get("status"), "error")
            self.assertEqual(index_info.get("searched"), True)
            self.assertEqual(index_info.get("found"), True)
            self.assertEqual(index_info.get("active"), False)
            self.assertIsInstance(index_info.get("error"), str)
            self.assertTrue(index_info.get("error"))
            index_path = index_info.get("index_path")
            self.assertIsInstance(index_path, str)
            self.assertEqual(
                Path(index_path).resolve(),
                (model_dir / "model.safetensors.index.json").resolve(),
            )

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            self._assert_required_keys_subset(
                write_manifest,
                {"generated_at", "requested_format", "requested_compression", "artifacts"},
                "write_manifest",
            )
            artifacts = write_manifest.get("artifacts", {})
            for name, artifact in artifacts.items():
                self._assert_artifact_entry(name, artifact)

            self.assertIn("warnings", artifacts)
            warnings_meta = artifacts.get("warnings", {})
            warnings_path = self._resolve_manifest_path(run_dir, warnings_meta.get("path"))
            self.assertTrue(warnings_path.exists())
            self.assertGreater(warnings_meta.get("rows"), 0)

            inv = artifacts.get("tensor_inventory", {})
            stats = artifacts.get("matrix_stats", {})
            quant = artifacts.get("quant_sim", {})
            self._assert_manifest_rows_match_csv(run_dir, inv)
            self._assert_manifest_rows_match_csv(run_dir, stats)
            self._assert_manifest_rows_match_csv(run_dir, quant)

            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            scan_summary = health.get("scan_summary", {})
            outputs = health.get("outputs_written", {})
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
            scan_plan = context.get("scan_plan", {})
            self.assertEqual(scan_summary.get("files_scanned"), scan_plan.get("scanned_files_count"))
            self.assertEqual(outputs.get("tensor_inventory_rows"), inv.get("rows"))
            self.assertEqual(outputs.get("matrix_stats_rows"), stats.get("rows"))
            self.assertEqual(outputs.get("quant_sim_rows"), quant.get("rows"))
            self.assertEqual(outputs.get("wrote_warnings"), "warnings" in artifacts)
            index_report_path = run_dir / "logs" / "index_report.json"
            self.assertEqual(outputs.get("wrote_index_report"), index_report_path.exists())
            self.assertFalse(outputs.get("wrote_index_report"))
            self.assertFalse(index_report_path.exists())
