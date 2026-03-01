import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CollectReportingSplitContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def test_collect_reporting_module_exists_and_exports_expected_api(self):
        path = self.scripts_dir / "collect_reporting.py"
        self.assertTrue(path.exists(), f"Expected helper module file is missing: {path}")
        mod = _load_module("collect_reporting", path)

        self.assertTrue(hasattr(mod, "build_index_report_data"), "Expected symbol build_index_report_data")
        self.assertTrue(callable(mod.build_index_report_data), "build_index_report_data must be callable")
        self.assertTrue(hasattr(mod, "build_index_summary"), "Expected symbol build_index_summary")
        self.assertTrue(callable(mod.build_index_summary), "build_index_summary must be callable")

        report_sig = inspect.signature(mod.build_index_report_data)
        self.assertEqual(
            list(report_sig.parameters.keys()),
            [
                "expected_shards",
                "scanned_shards",
                "weight_map",
                "observed_tensor_names",
                "extra_safetensors_files_on_disk",
                "index_metadata",
            ],
        )

        summary_sig = inspect.signature(mod.build_index_summary)
        self.assertEqual(
            list(summary_sig.parameters.keys()),
            [
                "index_used_for_scan",
                "index_parsed",
                "index_path",
                "strict_index",
                "expected_shards",
                "scanned_shards",
                "missing_shards_report",
                "extra_scanned_shards",
                "missing_tensors",
                "extra_tensors",
                "extra_on_disk",
                "index_metadata",
            ],
        )

    def test_build_index_report_data_computes_set_diffs_and_metadata(self):
        mod = _load_module("collect_reporting", self.scripts_dir / "collect_reporting.py")

        report = mod.build_index_report_data(
            expected_shards={"a.safetensors", "b.safetensors"},
            scanned_shards={"b.safetensors", "c.safetensors"},
            weight_map={"t0": "a.safetensors", "t1": "b.safetensors"},
            observed_tensor_names={"t1", "t2"},
            extra_safetensors_files_on_disk={"orphan.safetensors"},
            index_metadata={"format": "v1"},
        )

        self.assertEqual(report["expected_shards"], ["a.safetensors", "b.safetensors"])
        self.assertEqual(report["scanned_shards"], ["b.safetensors", "c.safetensors"])
        self.assertEqual(report["missing_shards"], ["a.safetensors"])
        self.assertEqual(report["extra_scanned_shards"], ["c.safetensors"])
        self.assertEqual(report["missing_tensors"], ["t0"])
        self.assertEqual(report["extra_tensors"], ["t2"])
        self.assertEqual(report["extra_safetensors_files_on_disk"], ["orphan.safetensors"])
        self.assertEqual(report["index_metadata"], {"format": "v1"})

    def test_build_index_summary_zeroes_counts_when_index_not_used(self):
        mod = _load_module("collect_reporting", self.scripts_dir / "collect_reporting.py")

        summary = mod.build_index_summary(
            index_used_for_scan=False,
            index_parsed=True,
            index_path=Path("/tmp/model.index.json"),
            strict_index=True,
            expected_shards={"a.safetensors"},
            scanned_shards={"a.safetensors", "x.safetensors"},
            missing_shards_report=["missing.safetensors"],
            extra_scanned_shards=["x.safetensors"],
            missing_tensors=["t_missing"],
            extra_tensors=["t_extra"],
            extra_on_disk=["orphan.safetensors"],
            index_metadata={"foo": "bar"},
        )

        self.assertFalse(summary["active"])
        self.assertTrue(summary["parsed"])
        self.assertFalse(summary["used_for_scan"])
        self.assertEqual(summary["index_path"], "/tmp/model.index.json")
        self.assertTrue(summary["strict_index"])
        self.assertEqual(summary["expected_shards_count"], 0)
        self.assertEqual(summary["scanned_shards_count"], 0)
        self.assertEqual(summary["missing_shards_count"], 0)
        self.assertEqual(summary["extra_scanned_shards_count"], 0)
        self.assertEqual(summary["missing_tensors_count"], 0)
        self.assertEqual(summary["extra_tensors_count"], 0)
        self.assertEqual(summary["extra_safetensors_files_on_disk_count"], 0)
        self.assertEqual(summary["index_metadata"], {"foo": "bar"})

    def test_build_index_summary_counts_inputs_when_index_used_for_scan(self):
        mod = _load_module("collect_reporting", self.scripts_dir / "collect_reporting.py")

        summary = mod.build_index_summary(
            index_used_for_scan=True,
            index_parsed=True,
            index_path=Path("/tmp/model.index.json"),
            strict_index=False,
            expected_shards={"a.safetensors", "b.safetensors"},
            scanned_shards={"a.safetensors", "x.safetensors", "y.safetensors"},
            missing_shards_report=["b.safetensors"],
            extra_scanned_shards=["x.safetensors", "y.safetensors"],
            missing_tensors=["t0", "t1", "t2"],
            extra_tensors=["z0"],
            extra_on_disk=["orphan1.safetensors", "orphan2.safetensors"],
            index_metadata={"format": "v2"},
        )

        self.assertTrue(summary["active"])
        self.assertTrue(summary["parsed"])
        self.assertTrue(summary["used_for_scan"])
        self.assertEqual(summary["index_path"], "/tmp/model.index.json")
        self.assertFalse(summary["strict_index"])
        self.assertEqual(summary["expected_shards_count"], 2)
        self.assertEqual(summary["scanned_shards_count"], 3)
        self.assertEqual(summary["missing_shards_count"], 1)
        self.assertEqual(summary["extra_scanned_shards_count"], 2)
        self.assertEqual(summary["missing_tensors_count"], 3)
        self.assertEqual(summary["extra_tensors_count"], 1)
        self.assertEqual(summary["extra_safetensors_files_on_disk_count"], 2)
        self.assertEqual(summary["index_metadata"], {"format": "v2"})

    def test_collect_data_main_delegates_reporting_assembly_to_helpers(self):
        collect_data = _load_module("collect_data_reporting_wiring", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            tensor_name = "layers.0.experts.0.up_proj.weight"
            shard_path = model_dir / "shard1.npz"
            np.savez(shard_path, **{tensor_name: np.arange(4, dtype=np.float32).reshape(2, 2)})

            index_path = model_dir / "model.safetensors.index.json"
            index_path.write_text("{}")

            run_dir = tmp_path / "run"
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            (run_dir / "data").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model_id": "test-model",
                        "run_name": "reporting-wiring",
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

            calls = {"report": 0, "summary": 0}
            sentinel_report = {
                "expected_shards": ["shard1.npz"],
                "scanned_shards": ["shard1.npz"],
                "missing_shards": [],
                "extra_scanned_shards": [],
                "missing_tensors": [],
                "extra_tensors": [],
                "extra_safetensors_files_on_disk": [],
                "index_metadata": {"format": "stub-v1"},
            }
            sentinel_summary = {
                "active": True,
                "parsed": True,
                "used_for_scan": True,
                "index_path": str(index_path),
                "strict_index": False,
                "expected_shards_count": 1,
                "scanned_shards_count": 1,
                "missing_shards_count": 0,
                "extra_scanned_shards_count": 0,
                "missing_tensors_count": 0,
                "extra_tensors_count": 0,
                "extra_safetensors_files_on_disk_count": 0,
                "index_metadata": {"format": "stub-v1"},
            }

            class StubMeta:
                def find_safetensors_index_json(self, _model_path: Path) -> Path:
                    return index_path

                def parse_safetensors_index(self, _path: Path):
                    return ({tensor_name: "shard1.npz"}, {"format": "stub-v1"})

            def fake_build_index_report_data(*_args, **_kwargs):
                calls["report"] += 1
                return dict(sentinel_report)

            def fake_build_index_summary(*_args, **_kwargs):
                calls["summary"] += 1
                return dict(sentinel_summary)

            old_loaded = collect_data._METADATA_LOADED
            old_module = collect_data._METADATA_MODULE
            old_argv = sys.argv
            old_report = getattr(collect_data, "build_index_report_data", None)
            old_summary = getattr(collect_data, "build_index_summary", None)
            had_report = hasattr(collect_data, "build_index_report_data")
            had_summary = hasattr(collect_data, "build_index_summary")

            try:
                collect_data._METADATA_LOADED = True
                collect_data._METADATA_MODULE = StubMeta()
                collect_data.build_index_report_data = fake_build_index_report_data
                collect_data.build_index_summary = fake_build_index_summary
                sys.argv = ["collect_data.py", "--run-dir", str(run_dir)]
                collect_data.main()
            finally:
                sys.argv = old_argv
                collect_data._METADATA_LOADED = old_loaded
                collect_data._METADATA_MODULE = old_module
                if had_report:
                    collect_data.build_index_report_data = old_report
                else:
                    delattr(collect_data, "build_index_report_data")
                if had_summary:
                    collect_data.build_index_summary = old_summary
                else:
                    delattr(collect_data, "build_index_summary")

            self.assertEqual(calls["report"], 1)
            self.assertEqual(calls["summary"], 1)
            report_json = json.loads((run_dir / "logs" / "index_report.json").read_text())
            self.assertEqual(report_json, sentinel_report)
            health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            self.assertEqual(health.get("index_summary"), sentinel_summary)


if __name__ == "__main__":
    unittest.main()
