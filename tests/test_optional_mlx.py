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

    def _setup_and_collect(self, tmp_path: Path, stub_factory=None, cfg_overrides=None):
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        self._write_npz_with_key(
            model_dir / "weights.npz",
            "layers.0.experts.0.down_proj.weight",
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

    def test_build_tables_after_collect_data_without_mlx(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, env, _ = self._setup_and_collect(Path(tmp_dir))

            self._run([
                sys.executable,
                str(self.repo_root / "scripts" / "build_tables.py"),
                "--run-dir",
                str(run_dir),
            ], env=env)

            self.assertTrue((run_dir / "tables" / "A_weight_global_summary.csv").exists())
            self.assertTrue((run_dir / "tables" / "B_quant_global_summary.csv").exists())
            self.assertTrue((run_dir / "tables" / "B_quant_deltas.csv").exists())

    def test_collect_data_with_mlx_quantize_failure_emits_error_message(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
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
            )

            quant_path = run_dir / "data" / "quant_sim.csv"
            self.assertTrue(quant_path.exists())

            with quant_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertGreaterEqual(len(rows), 2)
            for row in rows:
                self.assertIn("stub quantize fail", row.get("error", ""))


if __name__ == "__main__":
    unittest.main()
