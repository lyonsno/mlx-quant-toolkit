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

            def set_default_device(self, _device):
                return None

            def array(self, x):
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
