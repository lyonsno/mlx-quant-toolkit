import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_STUB_DIR = None


def _ensure_stub_mlx():
    global _STUB_DIR
    if _STUB_DIR is not None:
        return
    tmp = tempfile.TemporaryDirectory()
    stub_root = Path(tmp.name) / "mlx"
    stub_root.mkdir(parents=True, exist_ok=True)
    (stub_root / "__init__.py").write_text("")
    (stub_root / "core.py").write_text("raise ImportError('stub mlx not available')\n")
    sys.path.insert(0, tmp.name)
    _STUB_DIR = tmp


def _load_collect_data():
    _ensure_stub_mlx()
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "collect_data.py"
    spec = importlib.util.spec_from_file_location("collect_data", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Unable to load collect_data module")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WeightStatsTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_per_expert_weight_stats_small_bank(self):
        bank = np.array(
            [
                [[-1.0, 0.0], [2.0, 3.0]],
                [[-1.0, -2.0], [4.0, 0.0]],
            ],
            dtype=np.float32,
        )
        eps = 1e-6
        cfg_stats = {
            "eps": eps,
            "percentiles_abs": [50.0, 99.0, 99.9],
            "sample_per_matrix": 10,
            "sample_seed": 123,
            "group_outlier_percentile": 50.0,
            "group_sizes_lastdim": [2],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.collect_data._per_expert_weight_stats(
                bank, cfg_stats, Path(tmp)
            )

        expected_mean = np.array([1.0, 0.25])
        expected_std = np.array([math.sqrt(2.5), math.sqrt(5.1875)])
        expected_mean_abs = np.array([1.5, 1.75])
        expected_rms = np.array([math.sqrt(3.5), math.sqrt(5.25)])
        expected_max_abs = np.array([3.0, 4.0])
        expected_p50 = np.array([1.5, 1.5])
        expected_p99 = np.array([2.0 + 0.97, 2.0 + 0.97 * 2.0])
        expected_p999 = np.array([2.0 + 0.997, 2.0 + 0.997 * 2.0])
        expected_outlier_max_over_mean = expected_max_abs / (expected_mean_abs + eps)
        expected_outlier_p99_over_median = expected_p99 / (expected_p50 + eps)
        expected_outlier_p999_over_median = expected_p999 / (expected_p50 + eps)
        ratio0_low = np.float32(1.2)
        ratio0_high = np.float32(2.0)
        ratio1_low = np.float32(4.0) / np.float32(3.0)
        ratio1_high = np.float32(2.0)
        expected_g2_p50 = np.array(
            [
                np.float32(ratio0_low + np.float32(0.5) * (ratio0_high - ratio0_low)),
                np.float32(ratio1_low + np.float32(0.5) * (ratio1_high - ratio1_low)),
            ],
            dtype=np.float32,
        )
        expected_g2_max = np.array([2.0, 2.0])

        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["std"], expected_std, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["mean_abs"], expected_mean_abs, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["rms"], expected_rms, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["max_abs"], expected_max_abs, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["p50_abs"], expected_p50, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["p99_abs"], expected_p99, rtol=0, atol=1e-6)
        np.testing.assert_allclose(stats["p999_abs"], expected_p999, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            stats["outlier_max_over_mean"],
            expected_outlier_max_over_mean,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            stats["outlier_p99_over_median"],
            expected_outlier_p99_over_median,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            stats["outlier_p999_over_median"],
            expected_outlier_p999_over_median,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            stats["g2_p50_outlier"], expected_g2_p50, rtol=0, atol=3e-6
        )
        np.testing.assert_allclose(
            stats["g2_max_outlier"], expected_g2_max, rtol=0, atol=5e-6
        )

    def test_per_expert_weight_stats_casts_to_float32(self):
        bank = np.array([[[16777217.0, 16777218.0]]], dtype=np.float64)
        cfg_stats = {
            "eps": 1e-6,
            "percentiles_abs": [50.0],
            "sample_per_matrix": 2,
            "sample_seed": 7,
            "group_outlier_percentile": 50.0,
            "group_sizes_lastdim": [2],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.collect_data._per_expert_weight_stats(
                bank, cfg_stats, Path(tmp)
            )

        expected_mean = np.array([bank.astype(np.float32).mean()])
        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=0, atol=0)
        self.assertNotEqual(float(stats["mean"][0]), float(bank.mean()))


if __name__ == "__main__":
    unittest.main()
