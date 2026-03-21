import importlib.util
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


def _normalized_offdiag_gram_drift(matrix: np.ndarray, quantized: np.ndarray, eps: float) -> float:
    axis_count = matrix.shape[0]
    if axis_count <= 1:
        return 0.0
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + eps
    quantized_norms = np.linalg.norm(quantized, axis=1, keepdims=True) + eps
    matrix_unit = matrix / matrix_norms
    quantized_unit = quantized / quantized_norms
    diff = (matrix_unit @ matrix_unit.T) - (quantized_unit @ quantized_unit.T)
    np.fill_diagonal(diff, 0.0)
    return float(np.linalg.norm(diff, ord="fro") / np.sqrt(axis_count * (axis_count - 1)))


def _max_offdiag_cos_drift(matrix: np.ndarray, quantized: np.ndarray, eps: float) -> float:
    axis_count = matrix.shape[0]
    if axis_count <= 1:
        return 0.0
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + eps
    quantized_norms = np.linalg.norm(quantized, axis=1, keepdims=True) + eps
    matrix_unit = matrix / matrix_norms
    quantized_unit = quantized / quantized_norms
    diff = np.abs((matrix_unit @ matrix_unit.T) - (quantized_unit @ quantized_unit.T))
    np.fill_diagonal(diff, 0.0)
    return float(np.max(diff))


class CollectQuantMetricContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def _stub_mx(self, perturb_value: float = 0.1):
        class StubMx:
            cpu = object()
            gpu = object()
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"

            def set_default_device(self, _device):
                return None

            def array(self, x, dtype=None):
                return np.array(x)

            def quantize(self, w_mx, *_args, mode, **_kwargs):
                scales = np.ones((w_mx.shape[0], 1, 1), dtype=np.float32)
                if mode == "affine":
                    biases = np.zeros((w_mx.shape[0], 1, 1), dtype=np.float32)
                    return np.array(w_mx), scales, biases
                return np.array(w_mx), scales

            def dequantize(self, wq, *_args, **_kwargs):
                wq_np = np.array(wq, copy=True)
                perturb = np.zeros_like(wq_np, dtype=np.float16)
                perturb[..., 0, 1] = np.float16(perturb_value)
                return wq_np + perturb

            def sqrt(self, x):
                return np.sqrt(x)

            def sum(self, x, axis=None):
                return np.sum(x, axis=axis)

            def max(self, x, axis=None):
                return np.max(x, axis=axis)

            def abs(self, x):
                return np.abs(x)

            def mean(self, x, axis=None):
                return np.mean(x, axis=axis)

            def eval(self, *_args):
                return None

        return StubMx()

    def _stub_mx_dtype_sensitive(self):
        class StubMx:
            cpu = object()
            gpu = object()
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"

            def __init__(self):
                self.last_array_dtype = None
                self.last_array_dtype_arg = None
                self.last_dequantize_dtype = None

            def set_default_device(self, _device):
                return None

            def array(self, x, dtype=None):
                arr = np.asarray(x)
                self.last_array_dtype = str(arr.dtype)
                self.last_array_dtype_arg = str(dtype)
                return arr

            def quantize(self, w_mx, *_args, mode, **_kwargs):
                scales = np.ones((w_mx.shape[0], 1, 1), dtype=np.float32)
                if mode == "affine":
                    biases = np.zeros((w_mx.shape[0], 1, 1), dtype=np.float32)
                    return np.array(w_mx), scales, biases
                return np.array(w_mx), scales

            def dequantize(self, wq, *_args, dtype=None, **_kwargs):
                self.last_dequantize_dtype = str(dtype)
                dtype_text = str(dtype)
                if "bfloat16" in dtype_text:
                    perturb_value = 0.125
                elif "float16" in dtype_text:
                    perturb_value = 0.25
                elif "float32" in dtype_text:
                    perturb_value = 0.5
                else:
                    raise AssertionError(f"Unexpected compute dtype: {dtype_text}")
                wq_np = np.array(wq, copy=True, dtype=np.float32)
                perturb = np.zeros_like(wq_np, dtype=np.float32)
                perturb[..., 0, 1] = np.float32(perturb_value)
                return wq_np + perturb

            def sqrt(self, x):
                return np.sqrt(x)

            def sum(self, x, axis=None):
                return np.sum(x, axis=axis)

            def max(self, x, axis=None):
                return np.max(x, axis=axis)

            def abs(self, x):
                return np.abs(x)

            def mean(self, x, axis=None):
                return np.mean(x, axis=axis)

            def eval(self, *_args):
                return None

        return StubMx()

    def _stub_mx_rejects_numpy_bf16_without_dtype(self):
        class _ArrayWrapper:
            def __init__(self, arr, dtype_token):
                self._arr = np.asarray(arr, dtype=np.float32)
                self.dtype = dtype_token
                self.shape = self._arr.shape

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._arr, dtype=dtype)

            def __mul__(self, other):
                return np.asarray(self._arr, dtype=np.float32) * np.asarray(other, dtype=np.float32)

            def __rmul__(self, other):
                return np.asarray(other, dtype=np.float32) * np.asarray(self._arr, dtype=np.float32)

        class StubMx:
            cpu = object()
            gpu = object()
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"

            def __init__(self):
                self.last_array_dtype_arg = None

            def set_default_device(self, _device):
                return None

            def array(self, x, dtype=None):
                arr = np.asarray(x)
                if dtype is None and "bfloat16" in str(arr.dtype):
                    raise ValueError("Invalid type ndarray received in array initialization")
                self.last_array_dtype_arg = str(dtype)
                return _ArrayWrapper(arr, dtype or arr.dtype)

            def quantize(self, w_mx, *_args, mode, **_kwargs):
                arr = np.array(w_mx, copy=True)
                scales = np.ones((arr.shape[0], 1, 1), dtype=np.float32)
                if mode == "affine":
                    biases = np.zeros((arr.shape[0], 1, 1), dtype=np.float32)
                    return arr, scales, biases
                return arr, scales

            def dequantize(self, wq, *_args, dtype=None, **_kwargs):
                dtype_text = str(dtype)
                if "bfloat16" in dtype_text:
                    perturb_value = 0.125
                elif "float16" in dtype_text:
                    perturb_value = 0.25
                elif "float32" in dtype_text:
                    perturb_value = 0.5
                else:
                    raise AssertionError(f"Unexpected compute dtype: {dtype_text}")
                wq_np = np.array(wq, copy=True, dtype=np.float32)
                perturb = np.zeros_like(wq_np, dtype=np.float32)
                perturb[..., 0, 1] = np.float32(perturb_value)
                return wq_np + perturb

            def sqrt(self, x):
                return np.sqrt(x)

            def sum(self, x, axis=None):
                return np.sum(x, axis=axis)

            def max(self, x, axis=None):
                return np.max(x, axis=axis)

            def abs(self, x):
                return np.abs(x)

            def mean(self, x, axis=None):
                return np.mean(x, axis=axis)

            def eval(self, *_args):
                return None

        return StubMx()

    def _stub_mx_vectorized_rel_metrics_sensitive(self):
        class _Tensor:
            __array_priority__ = 1000

            def __init__(self, arr, tag):
                self._arr = np.asarray(arr, dtype=np.float32)
                self.tag = tag
                self.shape = self._arr.shape
                self.ndim = self._arr.ndim
                self.dtype = np.float32

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._arr, dtype=dtype)

            def __sub__(self, other):
                other_arr = np.asarray(other, dtype=np.float32)
                other_tag = getattr(other, "tag", None)
                tag = "diff" if self.tag == "w_hat" and other_tag == "w_mx" else "sub"
                return _Tensor(self._arr - other_arr, tag)

            def __mul__(self, other):
                other_arr = np.asarray(other, dtype=np.float32)
                other_tag = getattr(other, "tag", None)
                if self.tag == "diff" and other_tag == "diff":
                    tag = "diff_sq"
                elif self.tag == "w_mx" and other_tag == "w_mx":
                    tag = "w_sq"
                else:
                    tag = "mul"
                return _Tensor(self._arr * other_arr, tag)

            def __getitem__(self, item):
                return np.asarray(self._arr[item], dtype=np.float32)

        class StubMx:
            cpu = object()
            gpu = object()
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"

            def set_default_device(self, _device):
                return None

            def array(self, x, dtype=None):
                return _Tensor(np.asarray(x, dtype=np.float32), "w_mx")

            def quantize(self, w_mx, *_args, mode, **_kwargs):
                arr = np.asarray(w_mx, dtype=np.float32)
                scales = np.ones((arr.shape[0], 1, 1), dtype=np.float32)
                if mode == "affine":
                    biases = np.zeros((arr.shape[0], 1, 1), dtype=np.float32)
                    return arr, scales, biases
                return arr, scales

            def dequantize(self, wq, *_args, dtype=None, **_kwargs):
                arr = np.asarray(wq, dtype=np.float32)
                perturbed = arr.copy()
                perturbed[..., 0, 1] += np.float32(0.1)
                return _Tensor(perturbed, "w_hat")

            def sqrt(self, x):
                return np.sqrt(x)

            def sum(self, x, axis=None):
                if getattr(x, "tag", None) == "diff_sq" and axis == (1, 2):
                    return np.array([9.0, 16.0], dtype=np.float32)
                if getattr(x, "tag", None) == "w_sq" and axis == (1, 2):
                    return np.array([1.0, 4.0], dtype=np.float32)
                return np.sum(np.asarray(x), axis=axis)

            def max(self, x, axis=None):
                if getattr(x, "tag", None) == "abs_diff" and axis == (1, 2):
                    return np.array([5.0, 6.0], dtype=np.float32)
                if getattr(x, "tag", None) == "abs_w" and axis == (1, 2):
                    return np.array([1.0, 2.0], dtype=np.float32)
                return np.max(np.asarray(x), axis=axis)

            def abs(self, x):
                tag = getattr(x, "tag", None)
                if tag == "diff":
                    return _Tensor(np.abs(np.asarray(x)), "abs_diff")
                if tag == "w_mx":
                    return _Tensor(np.abs(np.asarray(x)), "abs_w")
                return np.abs(np.asarray(x))

            def mean(self, x, axis=None):
                return np.mean(np.asarray(x), axis=axis)

            def eval(self, *_args):
                return None

        return StubMx()

    def test_mlx_quant_sim_emits_extended_scalar_metrics_with_backward_compatible_defaults(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        # Backward-compat contract: the new sampled-Gram metric must work when the new
        # sample-size config key is absent. On a 2x2 matrix, any reasonable default cap
        # should still evaluate the exact pairwise geometry.
        cfg_stats = {"eps": 1e-12, "sample_seed": 1337}

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            cfg_stats,
            device="cpu",
            load_mlx=self._stub_mx,
        )

        self.assertEqual(warns, [])
        self.assertEqual(len(df), 1)
        self.assertIn("w_rel_spectral", df.columns)
        self.assertIn("w_gram_cos_drift_sampled_rms", df.columns)
        self.assertNotIn("w_gram_cos_drift_sampled_max", df.columns)

        row = df.iloc[0]
        w = bank.astype(np.float16, copy=False)[0].astype(np.float32)
        w_hat = w.copy()
        w_hat[0, 1] = np.float32(w_hat[0, 1] + np.float16(0.1))

        expected_rel_spectral = float(
            np.linalg.norm(w_hat - w, ord=2) / (np.linalg.norm(w, ord=2) + float(cfg_stats["eps"]))
        )
        expected_row_gram = _normalized_offdiag_gram_drift(w, w_hat, float(cfg_stats["eps"]))
        expected_col_gram = _normalized_offdiag_gram_drift(w.T, w_hat.T, float(cfg_stats["eps"]))
        expected_gram_rms = max(expected_row_gram, expected_col_gram)

        self.assertAlmostEqual(float(row["w_rel_spectral"]), expected_rel_spectral, places=6)
        self.assertAlmostEqual(float(row["w_gram_cos_drift_sampled_rms"]), expected_gram_rms, places=6)

    def test_mlx_quant_sim_missing_quant_compute_dtype_preserves_legacy_fp16_behavior(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        cfg_stats = {"eps": 1e-12, "sample_seed": 1337}
        stub_mx = self._stub_mx_dtype_sensitive()

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            cfg_stats,
            device="cpu",
            load_mlx=lambda: stub_mx,
        )

        self.assertEqual(warns, [])
        row = df.iloc[0]
        expected_rel_fro = 0.25 / np.sqrt(2.0)
        self.assertIn("float32", str(stub_mx.last_array_dtype))
        self.assertIn("float16", str(stub_mx.last_array_dtype_arg))
        self.assertIn("float16", str(stub_mx.last_dequantize_dtype))
        self.assertAlmostEqual(float(row["w_rel_fro"]), expected_rel_fro, places=6)
        self.assertAlmostEqual(float(row["w_rel_max"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["w_rel_spectral"]), 0.25, places=6)

    def test_mlx_quant_sim_explicit_bf16_compute_dtype(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        cfg_stats = {"eps": 1e-12, "sample_seed": 1337, "quant_compute_dtype": "bf16"}
        stub_mx = self._stub_mx_dtype_sensitive()

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            cfg_stats,
            device="cpu",
            load_mlx=lambda: stub_mx,
        )

        self.assertEqual(warns, [])
        row = df.iloc[0]
        expected_rel_fro = 0.125 / np.sqrt(2.0)
        self.assertIn("float32", str(stub_mx.last_array_dtype))
        self.assertIn("bfloat16", str(stub_mx.last_array_dtype_arg))
        self.assertIn("bfloat16", str(stub_mx.last_dequantize_dtype))
        self.assertAlmostEqual(float(row["w_rel_fro"]), expected_rel_fro, places=6)
        self.assertAlmostEqual(float(row["w_rel_max"]), 0.125, places=6)
        self.assertAlmostEqual(float(row["w_rel_spectral"]), 0.125, places=6)

    def test_mlx_quant_sim_explicit_bf16_uses_mlx_dtype_token_not_numpy_bf16_array_init(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        stub_mx = self._stub_mx_rejects_numpy_bf16_without_dtype()

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {"eps": 1e-12, "sample_seed": 1337, "quant_compute_dtype": "bf16"},
            device="cpu",
            load_mlx=lambda: stub_mx,
        )

        self.assertEqual(warns, [])
        self.assertIn("bfloat16", str(stub_mx.last_array_dtype_arg))
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df.iloc[0]["w_rel_max"]), 0.125, places=6)

    def test_mlx_quant_sim_fp16_override_changes_dtype_and_metrics_from_explicit_bf16(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        default_stub_mx = self._stub_mx_dtype_sensitive()
        default_df, default_warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {"eps": 1e-12, "sample_seed": 1337, "quant_compute_dtype": "bf16"},
            device="cpu",
            load_mlx=lambda: default_stub_mx,
        )

        fp16_stub_mx = self._stub_mx_dtype_sensitive()
        fp16_df, fp16_warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {
                "eps": 1e-12,
                "sample_seed": 1337,
                "quant_compute_dtype": "fp16",
            },
            device="cpu",
            load_mlx=lambda: fp16_stub_mx,
        )

        self.assertEqual(default_warns, [])
        self.assertEqual(fp16_warns, [])
        default_row = default_df.iloc[0]
        fp16_row = fp16_df.iloc[0]
        self.assertIn("float32", str(default_stub_mx.last_array_dtype))
        self.assertIn("bfloat16", str(default_stub_mx.last_array_dtype_arg))
        self.assertIn("bfloat16", str(default_stub_mx.last_dequantize_dtype))
        self.assertIn("float32", str(fp16_stub_mx.last_array_dtype))
        self.assertIn("float16", str(fp16_stub_mx.last_array_dtype_arg))
        self.assertIn("float16", str(fp16_stub_mx.last_dequantize_dtype))
        self.assertLess(float(default_row["w_rel_max"]), float(fp16_row["w_rel_max"]))
        self.assertLess(float(default_row["w_rel_fro"]), float(fp16_row["w_rel_fro"]))
        self.assertLess(float(default_row["w_rel_spectral"]), float(fp16_row["w_rel_spectral"]))

    def test_mlx_quant_sim_keeps_rel_fro_and_rel_max_on_vectorized_mlx_path(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        )
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {"eps": 1e-12, "sample_seed": 1337, "quant_compute_dtype": "fp16"},
            device="cpu",
            load_mlx=self._stub_mx_vectorized_rel_metrics_sensitive,
        )

        self.assertEqual(warns, [])
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df.iloc[0]["w_rel_fro"]), 3.0, places=6)
        self.assertAlmostEqual(float(df.iloc[1]["w_rel_fro"]), 2.0, places=6)
        self.assertAlmostEqual(float(df.iloc[0]["w_rel_max"]), 5.0, places=6)
        self.assertAlmostEqual(float(df.iloc[1]["w_rel_max"]), 3.0, places=6)

    def test_mlx_quant_sim_applies_explicit_rel_den_floor_to_low_norm_tensors(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1e-4, 0.0], [0.0, 0.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {
                "eps": 1e-12,
                "sample_seed": 1337,
                "quant_compute_dtype": "fp16",
                "quant_rel_den_floor": 1.0,
            },
            device="cpu",
            load_mlx=self._stub_mx_dtype_sensitive,
        )

        self.assertEqual(warns, [])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertAlmostEqual(float(row["w_rel_fro"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["w_rel_max"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["w_rel_spectral"]), 0.25, delta=1e-3)

    def test_mlx_quant_sim_missing_rel_den_floor_preserves_legacy_low_norm_inflation(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1e-4, 0.0], [0.0, 0.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            {
                "eps": 1e-12,
                "sample_seed": 1337,
                "quant_compute_dtype": "fp16",
            },
            device="cpu",
            load_mlx=self._stub_mx_dtype_sensitive,
        )

        self.assertEqual(warns, [])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertAlmostEqual(float(row["w_rel_fro"]), 2500.0, places=6)
        self.assertAlmostEqual(float(row["w_rel_max"]), 2500.0, places=6)
        self.assertAlmostEqual(float(row["w_rel_spectral"]), 2500.0, delta=1.0)

    def test_mlx_quant_sim_rejects_invalid_quant_rel_den_floor(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        for bad_value in (0, -1.0):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_rel_den_floor.*positive",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, "quant_rel_den_floor": bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )

        for bad_value in ("oops", True):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_rel_den_floor.*number",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, "quant_rel_den_floor": bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )

        for bad_value in (float("nan"), float("inf")):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_rel_den_floor.*finite",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, "quant_rel_den_floor": bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )

    def test_mlx_quant_sim_rejects_unsupported_quant_compute_dtype(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        with self.assertRaisesRegex(
            ValueError,
            r"quant_compute_dtype.*bf16.*fp16",
        ):
            mod._mlx_quant_sim(
                bank,
                schemes,
                {"eps": 1e-12, "quant_compute_dtype": "int8"},
                device="cpu",
                load_mlx=self._stub_mx,
            )

    def test_spectral_norm_estimate_large_matrix_does_not_collapse_when_ones_is_nullspace(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        matrix = np.zeros((65, 65), dtype=np.float32)
        matrix[:2, :2] = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)

        estimate = mod._spectral_norm_estimate(matrix, eps=1e-12, power_iters=12)
        exact = float(np.linalg.norm(matrix, ord=2))

        self.assertAlmostEqual(exact, 2.0, places=6)
        self.assertGreater(estimate, 1.9, "Large-matrix spectral estimate should not collapse to zero")
        self.assertAlmostEqual(estimate, exact, places=4)

    def test_mlx_quant_sim_gram_metric_name_pins_rms_summary_not_pairwise_max(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
        cfg_stats = {"eps": 1e-12, "sample_seed": 1337}

        df, warns = mod._mlx_quant_sim(
            bank,
            schemes,
            cfg_stats,
            device="cpu",
            load_mlx=lambda: self._stub_mx(0.3),
        )

        self.assertEqual(warns, [])
        row = df.iloc[0]
        w = bank.astype(np.float16, copy=False)[0].astype(np.float32)
        w_hat = w.copy()
        w_hat[0, 1] = np.float32(w_hat[0, 1] + np.float16(0.3))

        expected_row_rms = _normalized_offdiag_gram_drift(w, w_hat, float(cfg_stats["eps"]))
        expected_col_rms = _normalized_offdiag_gram_drift(w.T, w_hat.T, float(cfg_stats["eps"]))
        expected_rms = max(expected_row_rms, expected_col_rms)

        expected_row_max = _max_offdiag_cos_drift(w, w_hat, float(cfg_stats["eps"]))
        expected_col_max = _max_offdiag_cos_drift(w.T, w_hat.T, float(cfg_stats["eps"]))
        expected_pairwise_max = max(expected_row_max, expected_col_max)

        self.assertNotAlmostEqual(expected_rms, expected_pairwise_max, places=6)
        self.assertAlmostEqual(float(row["w_gram_cos_drift_sampled_rms"]), expected_rms, places=6)

    def test_mlx_quant_sim_rejects_non_positive_quant_spectral_power_iters(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        for bad_value in (0, -3):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_spectral_power_iters.*positive",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, "quant_spectral_power_iters": bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )

    def test_mlx_quant_sim_rejects_non_positive_quant_gram_sample_k(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        for bad_value in (0, -8):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_gram_sample_k.*positive",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, "quant_gram_sample_k": bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )

    def test_mlx_quant_sim_rejects_non_integer_hidden_quant_knobs(self):
        mod = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        bank = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        schemes = [{"name": "q4", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]

        bad_cases = [
            ("quant_spectral_power_iters", True),
            ("quant_spectral_power_iters", 1.9),
            ("quant_gram_sample_k", True),
            ("quant_gram_sample_k", 1.9),
        ]

        for key, bad_value in bad_cases:
            with self.subTest(key=key, bad_value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{key}.*integer",
                ):
                    mod._mlx_quant_sim(
                        bank,
                        schemes,
                        {"eps": 1e-12, key: bad_value},
                        device="cpu",
                        load_mlx=self._stub_mx,
                    )


if __name__ == "__main__":
    unittest.main()
