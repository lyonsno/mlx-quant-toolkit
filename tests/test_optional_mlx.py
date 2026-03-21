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
import pandas as pd


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
            "def array(x, dtype=None):\n"
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
            "bfloat16 = 'bfloat16'\n"
            "float16 = 'float16'\n"
            "float32 = 'float32'\n"
        )
        return stub_root

    def _create_stub_mlx_success(self, root: Path) -> Path:
        stub_root = root / "stub_mlx_success"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text(
            "import numpy as np\n"
            "\n"
            "bfloat16 = 'bfloat16'\n"
            "float16 = 'float16'\n"
            "float32 = 'float32'\n"
            "\n"
            "def array(x, dtype=None):\n"
            "    return np.array(x)\n"
            "\n"
            "def quantize(w, *_args, mode=None, **_kwargs):\n"
            "    scales = np.ones((w.shape[0], 1, 1), dtype=np.float32)\n"
            "    if mode == 'affine':\n"
            "        biases = np.zeros((w.shape[0], 1, 1), dtype=np.float32)\n"
            "        return np.array(w), scales, biases\n"
            "    return np.array(w), scales\n"
            "\n"
            "def dequantize(wq, *_args, **_kwargs):\n"
            "    wq_np = np.array(wq, copy=True)\n"
            "    perturb = np.zeros_like(wq_np, dtype=np.float16)\n"
            "    perturb[..., 0, 1] = np.float16(0.1)\n"
            "    return wq_np + perturb\n"
            "\n"
            "def sqrt(x):\n"
            "    return np.sqrt(x)\n"
            "\n"
            "def sum(x, axis=None):\n"
            "    return np.sum(x, axis=axis)\n"
            "\n"
            "def max(x, axis=None):\n"
            "    return np.max(x, axis=axis)\n"
            "\n"
            "def abs(x):\n"
            "    return np.abs(x)\n"
            "\n"
            "def mean(x, axis=None):\n"
            "    return np.mean(x, axis=axis)\n"
            "\n"
            "def eval(*_args):\n"
            "    return None\n"
            "\n"
            "def set_default_device(_device):\n"
            "    return None\n"
            "\n"
            "cpu = object()\n"
            "gpu = object()\n"
        )
        return stub_root

    def _create_stub_mlx_dtype_sensitive(self, root: Path) -> Path:
        stub_root = root / "stub_mlx_dtype_sensitive"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text(
            "import numpy as np\n"
            "\n"
            "bfloat16 = 'bfloat16'\n"
            "float16 = 'float16'\n"
            "float32 = 'float32'\n"
            "\n"
            "def array(x, dtype=None):\n"
            "    return np.asarray(x)\n"
            "\n"
            "def quantize(w, *_args, mode=None, **_kwargs):\n"
            "    scales = np.ones((w.shape[0], 1, 1), dtype=np.float32)\n"
            "    if mode == 'affine':\n"
            "        biases = np.zeros((w.shape[0], 1, 1), dtype=np.float32)\n"
            "        return np.array(w), scales, biases\n"
            "    return np.array(w), scales\n"
            "\n"
            "def dequantize(wq, *_args, dtype=None, **_kwargs):\n"
            "    dtype_text = str(dtype)\n"
            "    if 'bfloat16' in dtype_text:\n"
            "        perturb_value = 0.125\n"
            "    elif 'float16' in dtype_text:\n"
            "        perturb_value = 0.25\n"
            "    elif 'float32' in dtype_text:\n"
            "        perturb_value = 0.5\n"
            "    else:\n"
            "        raise RuntimeError(f'unexpected dtype: {dtype_text}')\n"
            "    wq_np = np.array(wq, copy=True, dtype=np.float32)\n"
            "    perturb = np.zeros_like(wq_np, dtype=np.float32)\n"
            "    perturb[..., 0, 1] = np.float32(perturb_value)\n"
            "    return wq_np + perturb\n"
            "\n"
            "def sqrt(x):\n"
            "    return np.sqrt(x)\n"
            "\n"
            "def sum(x, axis=None):\n"
            "    return np.sum(x, axis=axis)\n"
            "\n"
            "def max(x, axis=None):\n"
            "    return np.max(x, axis=axis)\n"
            "\n"
            "def abs(x):\n"
            "    return np.abs(x)\n"
            "\n"
            "def mean(x, axis=None):\n"
            "    return np.mean(x, axis=axis)\n"
            "\n"
            "def eval(*_args):\n"
            "    return None\n"
            "\n"
            "def set_default_device(_device):\n"
            "    return None\n"
            "\n"
            "cpu = object()\n"
            "gpu = object()\n"
        )
        return stub_root

    def _create_stub_mlx_bf16_api_strict(self, root: Path) -> Path:
        stub_root = root / "stub_mlx_bf16_api_strict"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text(
            "import numpy as np\n"
            "\n"
            "bfloat16 = 'bfloat16'\n"
            "float16 = 'float16'\n"
            "float32 = 'float32'\n"
            "cpu = object()\n"
            "gpu = object()\n"
            "\n"
            "class _ArrayWrapper:\n"
            "    def __init__(self, arr, dtype_token):\n"
            "        self._arr = np.asarray(arr, dtype=np.float32)\n"
            "        self.dtype = dtype_token\n"
            "        self.shape = self._arr.shape\n"
            "\n"
            "    def __array__(self, dtype=None, copy=None):\n"
            "        return np.asarray(self._arr, dtype=dtype)\n"
            "\n"
            "    def __mul__(self, other):\n"
            "        return np.asarray(self._arr, dtype=np.float32) * np.asarray(other, dtype=np.float32)\n"
            "\n"
            "    def __rmul__(self, other):\n"
            "        return np.asarray(other, dtype=np.float32) * np.asarray(self._arr, dtype=np.float32)\n"
            "\n"
            "def array(x, dtype=None):\n"
            "    arr = np.asarray(x)\n"
            "    if dtype is None and 'bfloat16' in str(arr.dtype):\n"
            "        raise ValueError('Invalid type ndarray received in array initialization')\n"
            "    return _ArrayWrapper(arr, dtype or arr.dtype)\n"
            "\n"
            "def quantize(w, *_args, mode=None, **_kwargs):\n"
            "    arr = np.array(w, copy=True)\n"
            "    scales = np.ones((arr.shape[0], 1, 1), dtype=np.float32)\n"
            "    if mode == 'affine':\n"
            "        biases = np.zeros((arr.shape[0], 1, 1), dtype=np.float32)\n"
            "        return arr, scales, biases\n"
            "    return arr, scales\n"
            "\n"
            "def dequantize(wq, *_args, dtype=None, **_kwargs):\n"
            "    dtype_text = str(dtype)\n"
            "    if 'bfloat16' in dtype_text:\n"
            "        perturb_value = 0.125\n"
            "    elif 'float16' in dtype_text:\n"
            "        perturb_value = 0.25\n"
            "    elif 'float32' in dtype_text:\n"
            "        perturb_value = 0.5\n"
            "    else:\n"
            "        raise RuntimeError(f'unexpected dtype: {dtype_text}')\n"
            "    wq_np = np.array(wq, copy=True, dtype=np.float32)\n"
            "    perturb = np.zeros_like(wq_np, dtype=np.float32)\n"
            "    perturb[..., 0, 1] = np.float32(perturb_value)\n"
            "    return wq_np + perturb\n"
            "\n"
            "def sqrt(x):\n"
            "    return np.sqrt(x)\n"
            "\n"
            "def sum(x, axis=None):\n"
            "    return np.sum(x, axis=axis)\n"
            "\n"
            "def max(x, axis=None):\n"
            "    return np.max(x, axis=axis)\n"
            "\n"
            "def abs(x):\n"
            "    return np.abs(x)\n"
            "\n"
            "def mean(x, axis=None):\n"
            "    return np.mean(x, axis=axis)\n"
            "\n"
            "def eval(*_args):\n"
            "    return None\n"
            "\n"
            "def set_default_device(_device):\n"
            "    return None\n"
        )
        return stub_root

    def _run(self, args, env=None, check=True):
        return subprocess.run(
            args,
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _write_manual_build_tables_inputs(
        self,
        run_dir: Path,
        *,
        output_format: str,
        compression,
        include_deltas: bool = False,
    ):
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "output": {"format": output_format, "compression": compression},
            "delta_pairs": (
                [{"name": "delta_ab", "a": "scheme_a", "b": "scheme_b"}]
                if include_deltas
                else []
            ),
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

        matrix_cols = [
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
        with (data_dir / "matrix_stats.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_cols)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "layer": 0,
                        "proj": "down_proj",
                        "mean": 1.0,
                        "std": 0.1,
                        "mean_abs": 1.0,
                        "rms": 1.0,
                        "max_abs": 1.2,
                        "p50_abs": 1.0,
                        "p99_abs": 1.2,
                        "p999_abs": 1.2,
                        "outlier_max_over_mean": 1.2,
                        "outlier_p99_over_median": 1.2,
                        "outlier_p999_over_median": 1.2,
                    },
                    {
                        "layer": 1,
                        "proj": "down_proj",
                        "mean": 2.0,
                        "std": 0.2,
                        "mean_abs": 2.0,
                        "rms": 2.0,
                        "max_abs": 2.2,
                        "p50_abs": 2.0,
                        "p99_abs": 2.2,
                        "p999_abs": 2.2,
                        "outlier_max_over_mean": 1.1,
                        "outlier_p99_over_median": 1.1,
                        "outlier_p999_over_median": 1.1,
                    },
                ]
            )

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
        with (data_dir / "quant_sim.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=quant_cols)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "derived_tensor": "layers.0.experts.0.down_proj.weight",
                        "layer": 0,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.10,
                        "w_rel_max": 0.15,
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
                        "w_rel_fro": 0.20,
                        "w_rel_max": 0.25,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                    },
                    {
                        "derived_tensor": "layers.1.experts.0.down_proj.weight",
                        "layer": 1,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_a",
                        "w_rel_fro": 0.30,
                        "w_rel_max": 0.35,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                    },
                    {
                        "derived_tensor": "layers.1.experts.0.down_proj.weight",
                        "layer": 1,
                        "proj": "down_proj",
                        "expert_id": 0,
                        "rows": 2,
                        "cols": 2,
                        "scheme": "scheme_b",
                        "w_rel_fro": 0.40,
                        "w_rel_max": 0.45,
                        "scale_mean": 0.0,
                        "scale_max": 0.0,
                        "bias_mean": 0.0,
                        "bias_max": 0.0,
                    },
                ]
            )

    def _run_build_tables(self, run_dir: Path):
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "default"
        result = self._run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "build_tables.py"),
                "--run-dir",
                str(run_dir),
            ],
            env=env,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.returncode, 0, f"build_tables failed: {output}")
        return result

    def _expected_table_artifacts(self, include_deltas: bool = False):
        keys = [
            "A_weight_layer_summary",
            "A_weight_block4_summary",
            "A_weight_global_summary",
            "B_quant_layer_summary",
            "B_quant_block4_summary",
            "B_quant_global_summary",
        ]
        if include_deltas:
            keys.append("B_quant_deltas")
        return keys

    def _probe_parquet_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            probe_path = Path(tmp_dir) / "probe.parquet"
            try:
                pd.DataFrame({"x": [1]}).to_parquet(probe_path, index=False, compression=None)
            except Exception as exc:
                self.skipTest(f"parquet engine unavailable: {exc}")

    def _load_module(self, module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _setup_and_collect(
        self,
        tmp_path: Path,
        stub_factory=None,
        cfg_overrides=None,
        tensor_key: str | None = None,
        arr: np.ndarray | None = None,
        *,
        check: bool = True,
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
        ], env=env, check=check)

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

            with quant_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])

            self.assertIn("scheme", fieldnames)
            self.assertIn("w_rel_spectral", fieldnames)
            self.assertIn("w_gram_cos_drift_sampled_rms", fieldnames)
            self.assertNotIn("w_gram_cos_drift_sampled_max", fieldnames)

    def test_collect_data_without_mlx_allows_duplicate_quant_scheme_names_and_still_warns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, _, result = self._setup_and_collect(
                Path(tmp_dir),
                cfg_overrides={
                    "mlx": {"enabled": True, "device": "cpu"},
                    "quant_schemes": [
                        {
                            "name": "dup",
                            "mode": "symmetric",
                            "bits": 4,
                            "group_size": 32,
                            "enabled": True,
                        },
                        {
                            "name": "dup",
                            "mode": "symmetric",
                            "bits": 8,
                            "group_size": 64,
                            "enabled": True,
                        },
                    ],
                },
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")
            self.assertIn("mlx is not importable", output)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            quant_path = run_dir / "data" / "quant_sim.csv"
            self.assertTrue(matrix_path.exists())
            self.assertTrue(quant_path.exists())

    def test_collect_data_with_stub_mlx_success_emits_renamed_quant_metric_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
            run_dir, _, result = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_success,
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

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            quant_path = run_dir / "data" / "quant_sim.csv"
            self.assertTrue(quant_path.exists())

            with quant_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)

            self.assertIn("w_rel_spectral", fieldnames)
            self.assertIn("w_gram_cos_drift_sampled_rms", fieldnames)
            self.assertNotIn("w_gram_cos_drift_sampled_max", fieldnames)
            self.assertEqual(len(rows), arr.shape[0])
            for row in rows:
                self.assertNotEqual(row.get("w_rel_spectral"), "")
                self.assertNotEqual(row.get("w_gram_cos_drift_sampled_rms"), "")
                self.assertEqual(row.get("error"), "")

    def test_collect_data_with_stub_mlx_success_honors_quant_compute_dtype_from_analysis_config(self):
        def run_case(tmp_path: Path, stats_overrides=None) -> float:
            run_dir, _, result = self._setup_and_collect(
                tmp_path,
                stub_factory=self._create_stub_mlx_dtype_sensitive,
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
                    "stats": stats_overrides or {},
                },
                arr=np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            with (run_dir / "data" / "quant_sim.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("error"), "")
            return float(rows[0]["w_rel_max"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bf16_value = run_case(tmp_path / "bf16")
            fp16_value = run_case(
                tmp_path / "fp16",
                {"quant_compute_dtype": "fp16"},
            )
            fp32_value = run_case(
                tmp_path / "fp32",
                {"quant_compute_dtype": "fp32"},
            )

            self.assertAlmostEqual(bf16_value, 0.125, places=6)
            self.assertAlmostEqual(fp16_value, 0.25, places=6)
            self.assertAlmostEqual(fp32_value, 0.5, places=6)

    def test_collect_data_with_stub_mlx_success_missing_quant_compute_dtype_preserves_legacy_fp16_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
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
            cfg["delta_pairs"] = [{"name": "dummy_delta", "a": "scheme_a", "b": "scheme_b"}]
            cfg["mlx"] = {"enabled": True, "device": "cpu"}
            cfg["quant_schemes"] = [
                {
                    "name": "s1",
                    "mode": "symmetric",
                    "bits": 4,
                    "group_size": 32,
                    "enabled": True,
                }
            ]
            del cfg["stats"]["quant_compute_dtype"]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            stub_root = self._create_stub_mlx_dtype_sensitive(tmp_path)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONWARNINGS"] = "default"

            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "collect_data.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            with (run_dir / "data" / "quant_sim.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("error"), "")
            self.assertAlmostEqual(float(rows[0]["w_rel_max"]), 0.25, places=6)

    def test_collect_data_with_stub_mlx_success_applies_quant_rel_den_floor_from_analysis_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, _, result = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_dtype_sensitive,
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
                    "stats": {
                        "quant_compute_dtype": "fp16",
                        "quant_rel_den_floor": 1.0,
                    },
                },
                arr=np.array([[[1e-4, 0.0], [0.0, 0.0]]], dtype=np.float32),
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            with (run_dir / "data" / "quant_sim.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("error"), "")
            self.assertAlmostEqual(float(rows[0]["w_rel_fro"]), 0.25, places=6)
            self.assertAlmostEqual(float(rows[0]["w_rel_max"]), 0.25, places=6)
            self.assertAlmostEqual(float(rows[0]["w_rel_spectral"]), 0.25, delta=1e-3)

    def test_collect_data_with_stub_mlx_success_missing_rel_den_floor_preserves_legacy_low_norm_inflation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, _, result = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_dtype_sensitive,
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
                    "stats": {
                        "quant_compute_dtype": "fp16",
                    },
                },
                arr=np.array([[[1e-4, 0.0], [0.0, 0.0]]], dtype=np.float32),
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            with (run_dir / "data" / "quant_sim.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("error"), "")
            self.assertAlmostEqual(float(rows[0]["w_rel_fro"]), 2500.0, places=6)
            self.assertAlmostEqual(float(rows[0]["w_rel_max"]), 2500.0, places=6)
            self.assertAlmostEqual(float(rows[0]["w_rel_spectral"]), 2500.0, delta=1.0)

    def test_collect_data_with_stub_mlx_rejects_non_finite_quant_rel_den_floor_from_analysis_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, _, result = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_dtype_sensitive,
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
                    "stats": {
                        "quant_compute_dtype": "fp16",
                        "quant_rel_den_floor": float("nan"),
                    },
                },
                arr=np.array([[[1e-4, 0.0], [0.0, 0.0]]], dtype=np.float32),
                check=False,
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("quant_rel_den_floor", output)
            self.assertIn("finite", output)

    def test_collect_data_with_strict_stub_mlx_succeeds_with_default_bf16_init_run_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, _, result = self._setup_and_collect(
                Path(tmp_dir),
                stub_factory=self._create_stub_mlx_bf16_api_strict,
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
                arr=np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
                check=False,
            )

            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"collect_data failed unexpectedly:\n{output}")

            with (run_dir / "data" / "quant_sim.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("error"), "")
            self.assertAlmostEqual(float(rows[0]["w_rel_max"]), 0.125, places=6)

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

    def test_build_tables_handles_legitimate_zero_row_collect_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tensor_name = "layers.0.experts.0.weird_proj.weight"
            run_dir, env, _ = self._setup_and_collect(
                Path(tmp_dir),
                tensor_key=tensor_name,
                cfg_overrides={
                    "scan": {"experts_only": True},
                    "debug": {"dump_unmatched_tensors": True},
                },
            )

            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(
                result.returncode,
                0,
                f"build_tables should succeed for zero-row runs. Output:\n{output}",
            )

            a_global = run_dir / "tables" / "A_weight_global_summary.csv"
            b_global = run_dir / "tables" / "B_quant_global_summary.csv"
            self.assertTrue(a_global.exists())
            self.assertTrue(b_global.exists())

            with a_global.open(newline="") as handle:
                a_reader = csv.DictReader(handle)
                a_rows = list(a_reader)
            self.assertEqual(len(a_rows), 0)
            self.assertIsNotNone(a_reader.fieldnames)
            self.assertIn("proj", a_reader.fieldnames)
            self.assertIn("mean__median", a_reader.fieldnames)

            with b_global.open(newline="") as handle:
                b_reader = csv.DictReader(handle)
                b_rows = list(b_reader)
            self.assertEqual(len(b_rows), 0)
            self.assertIsNotNone(b_reader.fieldnames)
            self.assertIn("proj", b_reader.fieldnames)
            self.assertIn("scheme", b_reader.fieldnames)
            self.assertIn("w_rel_fro__median", b_reader.fieldnames)

    def test_build_tables_handles_header_only_empty_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            cfg = {
                "output": {"format": "csv", "compression": None},
                "delta_pairs": [],
            }
            (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

            matrix_cols = [
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
            with (data_dir / "matrix_stats.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=matrix_cols)
                writer.writeheader()

            quant_cols = [
                "layer",
                "proj",
                "scheme",
                "w_rel_fro",
                "w_rel_max",
                "scale_mean",
                "scale_max",
                "bias_mean",
                "bias_max",
            ]
            with (data_dir / "quant_sim.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=quant_cols)
                writer.writeheader()

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "default"
            result = self._run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "build_tables.py"),
                    "--run-dir",
                    str(run_dir),
                ],
                env=env,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_tables failed: {output}")

            a_layer = run_dir / "tables" / "A_weight_layer_summary.csv"
            b_layer = run_dir / "tables" / "B_quant_layer_summary.csv"
            self.assertTrue(a_layer.exists())
            self.assertTrue(b_layer.exists())

            with a_layer.open(newline="") as handle:
                a_reader = csv.DictReader(handle)
                a_rows = list(a_reader)
            self.assertEqual(len(a_rows), 0)
            self.assertIsNotNone(a_reader.fieldnames)
            self.assertIn("layer", a_reader.fieldnames)
            self.assertIn("proj", a_reader.fieldnames)
            self.assertIn("mean__median", a_reader.fieldnames)

            with b_layer.open(newline="") as handle:
                b_reader = csv.DictReader(handle)
                b_rows = list(b_reader)
            self.assertEqual(len(b_rows), 0)
            self.assertIsNotNone(b_reader.fieldnames)
            self.assertIn("layer", b_reader.fieldnames)
            self.assertIn("proj", b_reader.fieldnames)
            self.assertIn("scheme", b_reader.fieldnames)
            self.assertIn("w_rel_fro__median", b_reader.fieldnames)

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

    def test_build_tables_parquet_fallback_writes_tables_manifest_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression="invalid-codec",
            )
            self._run_build_tables(run_dir)

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            self.assertIn("generated_at", write_manifest)
            self.assertEqual(write_manifest.get("requested_format"), "parquet")
            self.assertEqual(write_manifest.get("requested_compression"), "invalid-codec")

            artifacts = write_manifest.get("artifacts", {})
            expected = self._expected_table_artifacts()
            self.assertEqual(sorted(artifacts), sorted(expected))

            expected_rows = {
                "A_weight_layer_summary": 2,
                "A_weight_block4_summary": 1,
                "A_weight_global_summary": 1,
                "B_quant_layer_summary": 4,
                "B_quant_block4_summary": 2,
                "B_quant_global_summary": 2,
            }
            for name in expected:
                meta = artifacts[name]
                self.assertEqual(meta.get("format"), "csv")
                self.assertTrue(meta.get("fallback"))
                self.assertIsInstance(meta.get("error"), str)
                self.assertTrue(meta.get("error"))
                rel_path = meta.get("path")
                self.assertIsInstance(rel_path, str)
                self.assertEqual(rel_path, f"tables/{name}.csv")
                csv_path = run_dir / rel_path
                self.assertTrue(csv_path.exists())
                with csv_path.open(newline="") as handle:
                    row_count = sum(1 for _ in csv.DictReader(handle))
                self.assertEqual(row_count, expected_rows[name])
                self.assertEqual(meta.get("rows"), expected_rows[name])

            artifact_paths = [artifacts[name]["path"] for name in expected]
            self.assertEqual(len(artifact_paths), len(set(artifact_paths)))

    def test_build_tables_parquet_fallback_manifest_rows_and_aggregates_match_csv_tables_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression="invalid-codec",
            )
            self._run_build_tables(run_dir)

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            artifacts = write_manifest.get("artifacts", {})
            expected = self._expected_table_artifacts()
            self.assertEqual(sorted(artifacts), sorted(expected))

            with (run_dir / "tables" / "A_weight_global_summary.csv").open(newline="") as handle:
                a_global_rows = list(csv.DictReader(handle))
            self.assertEqual(len(a_global_rows), 1)
            self.assertEqual(a_global_rows[0]["proj"], "down_proj")
            self.assertAlmostEqual(float(a_global_rows[0]["mean__median"]), 1.5)
            self.assertAlmostEqual(float(a_global_rows[0]["max_abs__p99"]), 2.19, places=2)
            self.assertEqual(
                artifacts["A_weight_global_summary"]["rows"],
                len(a_global_rows),
            )
            self.assertEqual(
                artifacts["A_weight_global_summary"]["path"],
                "tables/A_weight_global_summary.csv",
            )

            with (run_dir / "tables" / "B_quant_global_summary.csv").open(newline="") as handle:
                b_global_rows = list(csv.DictReader(handle))
            self.assertEqual(len(b_global_rows), 2)
            per_scheme = {row["scheme"]: row for row in b_global_rows}
            self.assertEqual(sorted(per_scheme), ["scheme_a", "scheme_b"])
            self.assertAlmostEqual(float(per_scheme["scheme_a"]["w_rel_fro__median"]), 0.2)
            self.assertAlmostEqual(float(per_scheme["scheme_b"]["w_rel_fro__median"]), 0.3)
            self.assertAlmostEqual(float(per_scheme["scheme_b"]["w_rel_max__max"]), 0.45)
            self.assertEqual(
                artifacts["B_quant_global_summary"]["rows"],
                len(b_global_rows),
            )
            self.assertEqual(
                artifacts["B_quant_global_summary"]["path"],
                "tables/B_quant_global_summary.csv",
            )

    def test_build_tables_parquet_fallback_manifest_drives_table_artifact_discovery(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression="invalid-codec",
            )
            self._run_build_tables(run_dir)

            expected = self._expected_table_artifacts()
            # Poison-pill: if discovery ignores manifest metadata and uses legacy extension scan,
            # these stale parquet files would be incorrectly selected for plotting.
            for name in expected:
                (run_dir / "tables" / f"{name}.parquet").write_text("not real parquet")

            table_artifacts = self._load_module(
                "table_artifacts",
                self.repo_root / "scripts" / "table_artifacts.py",
            )
            discovered = table_artifacts.discover_table_artifacts(run_dir, expected)

            self.assertEqual(sorted(discovered), sorted(expected))
            for name in expected:
                entry = discovered[name]
                self.assertEqual(entry["source"], "manifest")
                self.assertEqual(entry["path"], f"tables/{name}.csv")
                self.assertEqual(entry["format"], "csv")
                self.assertTrue(entry["fallback"])
                self.assertIsInstance(entry["error"], str)
                self.assertTrue(entry["error"])
                self.assertGreater(entry["rows"], 0)

    def test_build_tables_parquet_success_tables_manifest_marks_non_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()

            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression=None,
            )
            self._run_build_tables(run_dir)

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            artifacts = write_manifest.get("artifacts", {})
            expected = self._expected_table_artifacts()
            self.assertEqual(sorted(artifacts), sorted(expected))

            expected_rows = {
                "A_weight_layer_summary": 2,
                "A_weight_block4_summary": 1,
                "A_weight_global_summary": 1,
                "B_quant_layer_summary": 4,
                "B_quant_block4_summary": 2,
                "B_quant_global_summary": 2,
            }
            for name in expected:
                meta = artifacts[name]
                self.assertEqual(meta.get("format"), "parquet")
                self.assertFalse(meta.get("fallback"))
                self.assertEqual(meta.get("error"), "")
                rel_path = meta.get("path")
                self.assertIsInstance(rel_path, str)
                self.assertEqual(rel_path, f"tables/{name}.parquet")
                parquet_path = run_dir / rel_path
                self.assertTrue(parquet_path.exists())
                table_rows = pd.read_parquet(parquet_path)
                self.assertEqual(len(table_rows), expected_rows[name])
                self.assertEqual(meta.get("rows"), expected_rows[name])

            artifact_paths = [artifacts[name]["path"] for name in expected]
            self.assertEqual(len(artifact_paths), len(set(artifact_paths)))

    def test_build_tables_manifest_artifact_discovery_preserves_plot_axis_sequence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="csv",
                compression=None,
            )

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            with matrix_path.open(newline="") as handle:
                matrix_reader = csv.DictReader(handle)
                matrix_rows = list(matrix_reader)
                matrix_cols = list(matrix_reader.fieldnames or [])
            with matrix_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=matrix_cols)
                writer.writeheader()
                writer.writerows([matrix_rows[1], matrix_rows[0]])

            quant_path = run_dir / "data" / "quant_sim.csv"
            with quant_path.open(newline="") as handle:
                quant_reader = csv.DictReader(handle)
                quant_rows = list(quant_reader)
                quant_cols = list(quant_reader.fieldnames or [])
            with quant_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=quant_cols)
                writer.writeheader()
                writer.writerows([quant_rows[3], quant_rows[1], quant_rows[0], quant_rows[2]])

            self._run_build_tables(run_dir)

            expected = self._expected_table_artifacts()
            # Poison-pill: if discovery ignores manifest metadata, legacy scan would prefer parquet.
            for name in expected:
                (run_dir / "tables" / f"{name}.parquet").write_text("not real parquet")

            table_artifacts = self._load_module(
                "table_artifacts",
                self.repo_root / "scripts" / "table_artifacts.py",
            )
            discovered = table_artifacts.discover_table_artifacts(run_dir, expected)
            self.assertEqual(sorted(discovered), sorted(expected))

            a_entry = discovered["A_weight_layer_summary"]
            self.assertEqual(a_entry["source"], "manifest")
            self.assertEqual(a_entry["path"], "tables/A_weight_layer_summary.csv")
            a_path = run_dir / a_entry["path"]
            with a_path.open(newline="") as handle:
                a_rows = list(csv.DictReader(handle))
            self.assertEqual(a_entry["rows"], len(a_rows))
            self.assertEqual([int(row["layer"]) for row in a_rows], [0, 1])

            b_entry = discovered["B_quant_layer_summary"]
            self.assertEqual(b_entry["source"], "manifest")
            self.assertEqual(b_entry["path"], "tables/B_quant_layer_summary.csv")
            b_path = run_dir / b_entry["path"]
            with b_path.open(newline="") as handle:
                b_rows = list(csv.DictReader(handle))
            self.assertEqual(b_entry["rows"], len(b_rows))
            self.assertEqual(
                [(int(row["layer"]), row["scheme"]) for row in b_rows],
                [(0, "scheme_a"), (0, "scheme_b"), (1, "scheme_a"), (1, "scheme_b")],
            )

    def test_build_tables_writes_tables_manifest_without_touching_collect_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="csv",
                compression=None,
            )

            collect_manifest_path = run_dir / "logs" / "write_manifest.json"
            collect_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            collect_manifest = {"stage": "collect", "sentinel": 123}
            collect_manifest_path.write_text(json.dumps(collect_manifest, indent=2))

            self._run_build_tables(run_dir)

            self.assertEqual(
                json.loads(collect_manifest_path.read_text()),
                collect_manifest,
            )
            tables_manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(tables_manifest_path.exists())
            tables_manifest = json.loads(tables_manifest_path.read_text())
            self.assertEqual(tables_manifest.get("requested_format"), "csv")
            self.assertIsNone(tables_manifest.get("requested_compression"))

    def test_build_tables_parquet_fallback_tables_manifest_includes_delta_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression="invalid-codec",
                include_deltas=True,
            )
            self._run_build_tables(run_dir)

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            artifacts = write_manifest.get("artifacts", {})
            expected = self._expected_table_artifacts(include_deltas=True)
            self.assertEqual(sorted(artifacts), sorted(expected))

            deltas_meta = artifacts["B_quant_deltas"]
            self.assertEqual(deltas_meta.get("format"), "csv")
            self.assertTrue(deltas_meta.get("fallback"))
            self.assertIsInstance(deltas_meta.get("error"), str)
            self.assertTrue(deltas_meta.get("error"))
            self.assertEqual(deltas_meta.get("path"), "tables/B_quant_deltas.csv")
            deltas_path = run_dir / "tables" / "B_quant_deltas.csv"
            self.assertTrue(deltas_path.exists())
            with deltas_path.open(newline="") as handle:
                delta_rows = list(csv.DictReader(handle))
            self.assertEqual(len(delta_rows), 2)
            self.assertEqual(deltas_meta.get("rows"), len(delta_rows))

    def test_build_tables_parquet_success_tables_manifest_includes_delta_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._probe_parquet_engine()
            run_dir = Path(tmp_dir) / "run"
            self._write_manual_build_tables_inputs(
                run_dir,
                output_format="parquet",
                compression=None,
                include_deltas=True,
            )
            self._run_build_tables(run_dir)

            manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertTrue(manifest_path.exists())
            write_manifest = json.loads(manifest_path.read_text())
            artifacts = write_manifest.get("artifacts", {})
            expected = self._expected_table_artifacts(include_deltas=True)
            self.assertEqual(sorted(artifacts), sorted(expected))

            deltas_meta = artifacts["B_quant_deltas"]
            self.assertEqual(deltas_meta.get("format"), "parquet")
            self.assertFalse(deltas_meta.get("fallback"))
            self.assertEqual(deltas_meta.get("error"), "")
            self.assertEqual(deltas_meta.get("path"), "tables/B_quant_deltas.parquet")
            deltas_path = run_dir / "tables" / "B_quant_deltas.parquet"
            self.assertTrue(deltas_path.exists())
            delta_rows = pd.read_parquet(deltas_path)
            self.assertEqual(len(delta_rows), 2)
            self.assertEqual(deltas_meta.get("rows"), len(delta_rows))

    def test_tables_manifest_discovery_for_legacy_runs_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            expected = self._expected_table_artifacts()
            for name in expected:
                (tables_dir / f"{name}.csv").write_text("col_a\n1\n")

            tables_manifest_path = run_dir / "logs" / "tables_write_manifest.json"
            self.assertFalse(tables_manifest_path.exists())

            table_artifacts = self._load_module(
                "table_artifacts",
                self.repo_root / "scripts" / "table_artifacts.py",
            )
            discovered = table_artifacts.discover_table_artifacts(run_dir, expected)
            self.assertEqual(sorted(discovered), sorted(expected))
            for name in expected:
                entry = discovered[name]
                self.assertEqual(entry["path"], f"tables/{name}.csv")
                self.assertEqual(entry["format"], "csv")
                self.assertEqual(entry["source"], "legacy_scan")

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
                self.assertIn("w_rel_spectral", list(reader.fieldnames or []))
                self.assertIn("w_gram_cos_drift_sampled_rms", list(reader.fieldnames or []))
                self.assertNotIn("w_gram_cos_drift_sampled_max", list(reader.fieldnames or []))
                rows = list(reader)

            self.assertEqual(len(rows), arr.shape[0])

            expert_ids = [int(row["expert_id"]) for row in rows]
            self.assertEqual(sorted(expert_ids), list(range(arr.shape[0])))

            for row in rows:
                self.assertEqual(row.get("w_rel_spectral"), "")
                self.assertEqual(row.get("w_gram_cos_drift_sampled_rms"), "")
                self.assertIn("stub quantize fail", row.get("error", ""))


if __name__ == "__main__":
    unittest.main()
