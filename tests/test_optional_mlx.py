import csv
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


class OptionalMlxPipelineTests(unittest.TestCase):
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

    def _create_stub_mlx_quantize_fail(self, root: Path) -> Path:
        stub_root = root / "stub_mlx_quantize_fail"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text(
            "def array(x):\n"
            "    return x\n"
            "\n"
            "def quantize(*_args, **_kwargs):\n"
            "    raise RuntimeError('stub quantize fail')\n"
            "\n"
            "def set_default_device(_device):\n"
            "    return None\n"
            "\n"
            "cpu = object()\n"
            "gpu = object()\n"
        )
        return stub_root

    def _run(self, args, env=None):
        return subprocess.run(
            args,
            cwd=self.repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _setup_and_collect(
        self,
        tmp_path: Path,
        stub_factory=None,
        cfg_overrides=None,
        tensor_key: str | None = None,
        arr: np.ndarray | None = None,
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if arr is None:
            arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        if tensor_key is None:
            tensor_key = "layers.0.experts.0.down_proj.weight"
        self._write_npz_with_key(
            model_dir / "weights.npz",
            tensor_key,
            arr,
        )

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

        run_dir = run_root / "model" / "run"
        cfg_path = run_dir / "analysis_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["output"]["format"] = "csv"
        cfg["output"]["compression"] = None
        # Explicit delta_pairs override so test does not depend on init_run.py defaults
        cfg["delta_pairs"] = [
            {"name": "dummy_delta", "a": "scheme_a", "b": "scheme_b"}
        ]
        if cfg_overrides:
            for key, value in cfg_overrides.items():
                if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
        cfg_path.write_text(json.dumps(cfg, indent=2))

        if stub_factory is None:
            stub_factory = self._create_stub_mlx
        stub_root = stub_factory(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"

        result = self._run([
            sys.executable,
            str(self.repo_root / "scripts" / "collect_data.py"),
            "--run-dir",
            str(run_dir),
        ], env=env)

        return run_dir, env, result

    def test_collect_data_without_mlx_warns_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, _, result = self._setup_and_collect(Path(tmp_dir))

            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("mlx is not importable", output)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            quant_path = run_dir / "data" / "quant_sim.csv"
            self.assertTrue(matrix_path.exists())
            self.assertTrue(quant_path.exists())

            header = quant_path.read_text().splitlines()[0]
            self.assertIn("scheme", header)

    def test_collect_data_emits_unmatched_tensors_when_no_proj_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tensor_name = "layers.0.experts.0.weird_proj.weight"
            run_dir, _, _ = self._setup_and_collect(
                Path(tmp_dir),
                tensor_key=tensor_name,
                cfg_overrides={
                    "scan": {"experts_only": True},
                    "debug": {"dump_unmatched_tensors": True},
                },
            )

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())

            with unmatched_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            match = next((row for row in rows if row["tensor_name"] == tensor_name), None)
            self.assertIsNotNone(match)
            self.assertEqual(match["reason"], "no_rule_match_or_proj_infer")

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())

            with matrix_path.open(newline="") as handle:
                matrix_rows = list(csv.DictReader(handle))

            self.assertEqual(len(matrix_rows), 0)

    def test_build_tables_computes_quant_deltas_from_manual_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "output": {"format": "csv", "compression": None},
                "delta_pairs": [{"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"}],
            }
            (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

            stat_cols = [
                "layer",
                "proj",
                "mean",
                "std",
                "mean_abs",
                "rms",
                "max_abs",
                "p50_abs",
                "p99_abs",
                "p999_abs",
                "outlier_max_over_mean",
                "outlier_p99_over_median",
                "outlier_p999_over_median",
            ]
            matrix_rows = [
                {
                    "layer": 0,
                    "proj": "down_proj",
                    "mean": 1.0,
                    "std": 0.1,
                    "mean_abs": 1.0,
                    "rms": 1.0,
                    "max_abs": 1.5,
                    "p50_abs": 1.0,
                    "p99_abs": 1.4,
                    "p999_abs": 1.5,
                    "outlier_max_over_mean": 1.5,
                    "outlier_p99_over_median": 1.4,
                    "outlier_p999_over_median": 1.5,
                }
            ]
            with (data_dir / "matrix_stats.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=stat_cols)
                writer.writeheader()
                writer.writerows(matrix_rows)

            quant_cols = [
                "derived_tensor",
                "layer",
                "proj",
                "expert_id",
                "rows",
                "cols",
                "scheme",
                "w_rel_fro",
                "w_rel_max",
                "scale_mean",
                "scale_max",
                "bias_mean",
                "bias_max",
            ]
            quant_rows = [
                {
                    "derived_tensor": "layers.0.experts.0.down_proj.weight",
                    "layer": 0,
                    "proj": "down_proj",
                    "expert_id": 0,
                    "rows": 2,
                    "cols": 2,
                    "scheme": "scheme_a",
                    "w_rel_fro": 0.15,
                    "w_rel_max": 0.2,
                    "scale_mean": 0.0,
                    "scale_max": 0.0,
                    "bias_mean": 0.0,
                    "bias_max": 0.0,
                },
                {
                    "derived_tensor": "layers.0.experts.0.down_proj.weight",
                    "layer": 0,
                    "proj": "down_proj",
                    "expert_id": 0,
                    "rows": 2,
                    "cols": 2,
                    "scheme": "scheme_b",
                    "w_rel_fro": 0.09,
                    "w_rel_max": 0.12,
                    "scale_mean": 0.0,
                    "scale_max": 0.0,
                    "bias_mean": 0.0,
                    "bias_max": 0.0,
                },
            ]
            with (data_dir / "quant_sim.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=quant_cols)
                writer.writeheader()
                writer.writerows(quant_rows)

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            self._run([
                sys.executable,
                str(self.repo_root / "scripts" / "build_tables.py"),
                "--run-dir",
                str(run_dir),
            ], env=env)

            deltas_path = run_dir / "tables" / "B_quant_deltas.csv"
            with deltas_path.open(newline="") as handle:
                delta_rows = list(csv.DictReader(handle))

            self.assertEqual(len(delta_rows), 1)
            delta_row = delta_rows[0]
            self.assertEqual(delta_row["delta_name"], "delta_ab")
            self.assertAlmostEqual(float(delta_row["delta_w_rel_fro"]), 0.06)
            self.assertAlmostEqual(float(delta_row["delta_w_rel_max"]), 0.08)

            global_path = run_dir / "tables" / "B_quant_global_summary.csv"
            with global_path.open(newline="") as handle:
                global_rows = list(csv.DictReader(handle))

            self.assertEqual(len(global_rows), 2)
            scheme_a = next(row for row in global_rows if row["scheme"] == "scheme_a")
            self.assertAlmostEqual(float(scheme_a["w_rel_fro__median"]), 0.15)

    def test_collect_data_with_mlx_quantize_failure_emits_error_message(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
            run_dir, _, _ = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_quantize_fail,
                cfg_overrides={
                    "mlx": {"enabled": True, "device": "cpu"},
                    "quant_schemes": [
                        {
                            "name": "s1",
                            "mode": "symmetric",
                            "bits": 4,
                            "group_size": 32,
                            "enabled": True,
                        }
                    ],
                },
                arr=arr,
            )

            quant_path = run_dir / "data" / "quant_sim.csv"
            self.assertTrue(quant_path.exists())

            with quant_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(len(rows), arr.shape[0])

            expert_ids = [int(row["expert_id"]) for row in rows]
            self.assertEqual(sorted(expert_ids), list(range(arr.shape[0])))

            for row in rows:
                self.assertIn("stub quantize fail", row.get("error", ""))


if __name__ == "__main__":
    unittest.main()
