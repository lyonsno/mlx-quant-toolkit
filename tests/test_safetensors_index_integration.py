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

    def _write_config(
        self,
        run_dir: Path,
        model_dir: Path,
        use_index: bool | None,
        strict_index: bool,
        extensions: list[str] | None = None,
    ) -> None:
        if extensions is None:
            extensions = [".safetensors"]
        cfg = {
            "model_path": str(model_dir),
            "scan": {
                "extensions": extensions,
                "experts_only": True,
                "include_shared_expert": True,
                "inventory_all_tensors": True,
                "max_files": None,
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
        if use_index is not None:
            cfg["scan"]["use_safetensors_index_json"] = use_index
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
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "shard1.safetensors", {t1: arr})
            self._write_safetensors(model_dir / "shard2.safetensors", {t2: arr + 1})
            (model_dir / "extra.safetensors").write_bytes(b"not a safetensors file")

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
            expected_shards = {"shard1.safetensors", "shard2.safetensors"}
            self.assertEqual(report.get("missing_tensors"), [])
            self.assertEqual(report.get("extra_tensors"), [])
            self.assertEqual(report.get("missing_shards"), [])
            self.assertEqual(set(report.get("expected_shards", [])), expected_shards)
            self.assertEqual(set(report.get("scanned_shards", [])), expected_shards)
            self.assertEqual(set(report.get("extra_scanned_shards", [])), set())
            self.assertNotIn("extra.safetensors", report.get("scanned_shards", []))
            self.assertEqual(
                set(report.get("extra_safetensors_files_on_disk", [])),
                {"extra.safetensors"},
            )
            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            scan_plan = context.get("scan_plan", {})
            self.assertEqual(scan_plan.get("scan_mode"), "index")
            index_info = context.get("index", {})
            self.assertEqual(index_info.get("parsed"), True)
            self.assertEqual(index_info.get("used_for_scan"), True)
            self.assertEqual(index_info.get("active"), True)

            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            index_summary = health.get("index_summary", {})
            self.assertEqual(index_summary.get("parsed"), True)
            self.assertEqual(index_summary.get("used_for_scan"), True)
            self.assertEqual(index_summary.get("active"), True)

    def test_collect_data_reports_extra_tensor_in_indexed_shard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            t_extra = "layers.0.experts.2.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "shard1.safetensors", {t1: arr, t_extra: arr + 2})
            self._write_safetensors(model_dir / "shard2.safetensors", {t2: arr + 1})

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: "shard1.safetensors", t2: "shard2.safetensors"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=False)

            self._run_collect(run_dir, self._env(), check=True)

            inv_path = run_dir / "data" / "tensor_inventory.csv"
            with inv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 3)
            by_name = {row["tensor_name"]: row for row in rows}
            self.assertEqual(by_name[t1]["in_index"], "True")
            self.assertEqual(by_name[t1]["index_shard"], "shard1.safetensors")
            self.assertEqual(by_name[t2]["in_index"], "True")
            self.assertEqual(by_name[t2]["index_shard"], "shard2.safetensors")
            self.assertEqual(by_name[t_extra]["in_index"], "False")
            self.assertEqual(by_name[t_extra]["index_shard"], "")

            report_path = run_dir / "logs" / "index_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text())
            expected_shards = {"shard1.safetensors", "shard2.safetensors"}
            self.assertEqual(report.get("missing_tensors"), [])
            self.assertEqual(set(report.get("extra_tensors", [])), {t_extra})
            self.assertEqual(report.get("missing_shards"), [])
            self.assertEqual(set(report.get("expected_shards", [])), expected_shards)
            self.assertEqual(set(report.get("scanned_shards", [])), expected_shards)
            self.assertEqual(set(report.get("extra_scanned_shards", [])), set())
            self.assertEqual(set(report.get("extra_safetensors_files_on_disk", [])), set())

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
            expected_shards = {"shard1.safetensors", "shard2.safetensors"}
            self.assertIn("shard2.safetensors", report.get("missing_shards", []))
            self.assertIn(t2, report.get("missing_tensors", []))
            self.assertEqual(set(report.get("expected_shards", [])), expected_shards)
            self.assertEqual(set(report.get("scanned_shards", [])), {"shard1.safetensors"})
            self.assertEqual(set(report.get("extra_scanned_shards", [])), set())
            self.assertEqual(set(report.get("extra_safetensors_files_on_disk", [])), set())

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

    def test_collect_data_missing_index_falls_back_without_inventory_enrichment(self):
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

    def test_collect_data_strict_index_requires_active_index_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "weights.safetensors", {t1: arr})

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=True)

            result = self._run_collect(run_dir, self._env(), check=False)
            self.assertNotEqual(result.returncode, 0)
            output = ((result.stdout or "") + (result.stderr or "")).lower()
            self.assertIn("strict_index", output)
            self.assertIn("active index", output)

    def test_collect_data_strict_index_requires_active_index_when_invalid(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "weights.safetensors", {t1: arr})
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"format": "pt"}}, indent=2)
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=True, strict_index=True)

            result = self._run_collect(run_dir, self._env(), check=False)
            self.assertNotEqual(result.returncode, 0)
            output = ((result.stdout or "") + (result.stderr or "")).lower()
            self.assertIn("strict_index", output)
            self.assertIn("active index", output)

    def test_collect_data_strict_index_requires_use_index_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "weights.safetensors", {t1: arr})

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=False, strict_index=True)

            result = self._run_collect(run_dir, self._env(), check=False)
            self.assertNotEqual(result.returncode, 0)
            output = ((result.stdout or "") + (result.stderr or "")).lower()
            self.assertIn("strict_index", output)
            self.assertIn("use_safetensors_index_json", output)

    def test_collect_data_file_model_path_strict_index_requires_active_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            shard_ok = model_dir / "shard_ok.npz"
            np.savez(shard_ok, **{t1: arr})

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                shard_ok,
                use_index=True,
                strict_index=True,
                extensions=[".npz"],
            )

            result = self._run_collect(run_dir, self._env(), check=False)
            self.assertNotEqual(result.returncode, 0)
            output = ((result.stdout or "") + (result.stderr or "")).lower()
            self.assertIn("strict_index", output)
            self.assertIn("active index", output)

    def test_collect_data_file_model_path_does_not_expand_index_scan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            shard_ok = model_dir / "shard_ok.safetensors"
            self._write_safetensors(shard_ok, {t1: arr})
            (model_dir / "poison.safetensors").write_bytes(b"not a safetensors file")

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: shard_ok.name, t2: "poison.safetensors"},
                metadata={"format": "pt"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                shard_ok,
                use_index=True,
                strict_index=True,
            )

            result = self._run_collect(run_dir, self._env(), check=False)
            combined = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(
                result.returncode,
                0,
                msg=f"collect_data failed unexpectedly:\n{combined}",
            )
            output = combined.lower()
            self.assertIn("index found", output)
            self.assertIn("model_path is a file", output)

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            scan_plan = context.get("scan_plan", {})
            self.assertEqual(scan_plan.get("scan_mode"), "walk")
            self.assertEqual(scan_plan.get("scanned_files_count"), 1)
            self.assertEqual(
                scan_plan.get("index_discovered_but_ignored_due_to_file_model_path"),
                True,
            )
            scan_example = scan_plan.get("scanned_files_example") or []
            self.assertTrue(scan_example)
            self.assertTrue(scan_example[0].endswith("shard_ok.safetensors"))
            index_info = context.get("index", {})
            self.assertEqual(index_info.get("parsed"), True)
            self.assertEqual(index_info.get("active"), False)
            self.assertEqual(index_info.get("used_for_scan"), False)

            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            index_summary = health.get("index_summary", {})
            self.assertEqual(index_summary.get("parsed"), True)
            self.assertEqual(index_summary.get("active"), False)
            self.assertEqual(index_summary.get("used_for_scan"), False)

    def test_collect_data_file_model_path_ignores_missing_index_shards_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            t2 = "layers.0.experts.1.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            shard_ok = model_dir / "shard_ok.safetensors"
            self._write_safetensors(shard_ok, {t1: arr})

            index_path = model_dir / "model.safetensors.index.json"
            self._write_index(
                index_path,
                {t1: shard_ok.name, t2: "missing.safetensors"},
                metadata={"format": "pt"},
            )

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(
                run_dir,
                shard_ok,
                use_index=True,
                strict_index=True,
            )

            result = self._run_collect(run_dir, self._env(), check=False)
            combined = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(
                result.returncode,
                0,
                msg=f"collect_data failed unexpectedly:\n{combined}",
            )
            output = combined.lower()
            self.assertIn("index found", output)
            self.assertIn("model_path is a file", output)

            context = json.loads((run_dir / "logs" / "run_context.json").read_text())
            scan_plan = context.get("scan_plan", {})
            self.assertEqual(scan_plan.get("scanned_files_count"), 1)
            self.assertEqual(
                scan_plan.get("index_discovered_but_ignored_due_to_file_model_path"),
                True,
            )
    def test_collect_data_without_index_flag_keeps_inventory_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            t1 = "layers.0.experts.0.down_proj.weight"
            arr = np.arange(4, dtype=np.float32).reshape(2, 2)

            self._write_safetensors(model_dir / "weights.safetensors", {t1: arr})

            run_dir = tmp_path / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(run_dir, model_dir, use_index=None, strict_index=False)

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
