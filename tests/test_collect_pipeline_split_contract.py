import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CollectPipelineSplitContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def test_collect_pipeline_module_exists_and_exports_expected_api(self):
        path = self.scripts_dir / "collect_pipeline.py"
        self.assertTrue(path.exists(), f"Expected helper module file is missing: {path}")
        mod = _load_module("collect_pipeline", path)

        self.assertTrue(hasattr(mod, "record_example"), "Expected symbol record_example")
        self.assertTrue(callable(mod.record_example), "record_example must be callable")
        self.assertTrue(hasattr(mod, "process_one_bank"), "Expected symbol process_one_bank")
        self.assertTrue(callable(mod.process_one_bank), "process_one_bank must be callable")
        self.assertTrue(hasattr(mod, "process_extracted_banks"), "Expected symbol process_extracted_banks")
        self.assertTrue(
            callable(mod.process_extracted_banks),
            "process_extracted_banks must be callable",
        )

        record_sig = inspect.signature(mod.record_example)
        self.assertEqual(list(record_sig.parameters.keys()), ["dst", "value", "limit"])
        self.assertEqual(record_sig.parameters["limit"].default, 25)

        process_sig = inspect.signature(mod.process_one_bank)
        expected_process_params = [
            "bank_obj",
            "bank_erc",
            "layer_idx",
            "cfg_stats",
            "cache_idx_dir",
            "matrix_rows",
            "quant_rows",
            "mlx_enabled",
            "schemes",
            "mlx_device",
            "per_expert_weight_stats",
            "mlx_quant_sim",
            "warn_log",
        ]
        self.assertEqual(list(process_sig.parameters.keys()), expected_process_params)

        extracted_sig = inspect.signature(mod.process_extracted_banks)
        expected_extracted_params = [
            "extracted",
            "cfg_stats",
            "cache_idx_dir",
            "matrix_rows",
            "quant_rows",
            "mlx_enabled",
            "schemes",
            "mlx_device",
            "per_expert_weight_stats",
            "mlx_quant_sim",
            "warn_log",
            "process_one_bank_fn",
        ]
        self.assertEqual(list(extracted_sig.parameters.keys()), expected_extracted_params)

    def test_record_example_deduplicates_and_honors_limit(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        dst: list[str] = []
        for value in ["a", "b", "a", "c", "d"]:
            mod.record_example(dst, value, limit=3)

        self.assertEqual(dst, ["a", "b", "c"])

    def test_process_one_bank_shared_expert_ids_and_flags(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        bank_obj = SimpleNamespace(
            source_file="f.npz",
            source_tensor="layers.0.shared_expert.w2.weight",
            derived_tensor="layers.0.shared_expert.w2.weight::down_proj",
            proj="down_proj",
            is_shared_expert=True,
            layer_base=0,
            expert_single_id=None,
        )
        bank_erc = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )

        def fake_stats(bank, _cfg_stats, _cache_dir):
            e_count = bank.shape[0]
            return {
                "mean": np.arange(e_count, dtype=np.float32),
                "max_abs": np.arange(e_count, dtype=np.float32) + 10.0,
            }

        matrix_rows: list[dict] = []
        quant_rows: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod.process_one_bank(
                bank_obj=bank_obj,
                bank_erc=bank_erc,
                layer_idx=0,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=False,
                schemes=[],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=lambda *_args, **_kwargs: (None, []),
            )

        self.assertEqual(len(matrix_rows), 2)
        self.assertEqual(len(quant_rows), 0)
        for i, row in enumerate(matrix_rows):
            self.assertEqual(row["file"], "f.npz")
            self.assertEqual(row["source_tensor"], "layers.0.shared_expert.w2.weight")
            self.assertEqual(
                row["derived_tensor"],
                "layers.0.shared_expert.w2.weight::down_proj",
            )
            self.assertEqual(row["proj"], "down_proj")
            self.assertEqual(row["expert_id"], -1)
            self.assertFalse(row["is_routed_expert"])
            self.assertTrue(row["is_shared_expert"])
            self.assertEqual(row["layer"], 0)
            self.assertEqual(row["block4"], 0)
            self.assertEqual(row["rows"], 2)
            self.assertEqual(row["cols"], 2)
            self.assertEqual(row["dtype"], "float32")
            self.assertEqual(row["mean"], float(i))
            self.assertEqual(row["max_abs"], float(i + 10.0))

    def test_process_one_bank_layer_fallback_uses_layer_base_then_unknown(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        bank_erc = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

        def fake_stats(bank, _cfg_stats, _cache_dir):
            e_count = bank.shape[0]
            return {"mean": np.arange(e_count, dtype=np.float32)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Case 1: layer_idx is None, so layer should fall back to bank_obj.layer_base.
            bank_obj_with_base = SimpleNamespace(
                source_file="f.npz",
                source_tensor="t0",
                derived_tensor="t0::down_proj",
                proj="down_proj",
                is_shared_expert=False,
                layer_base=9,
                expert_single_id=None,
            )
            matrix_rows_1: list[dict] = []
            mod.process_one_bank(
                bank_obj=bank_obj_with_base,
                bank_erc=bank_erc,
                layer_idx=None,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows_1,
                quant_rows=[],
                mlx_enabled=False,
                schemes=[],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=lambda *_args, **_kwargs: (None, []),
            )
            self.assertEqual(matrix_rows_1[0]["layer"], 9)
            self.assertEqual(matrix_rows_1[0]["block4"], 2)

            # Case 2: layer_idx is None and layer_base is None, so layer should be unknown (-1).
            bank_obj_unknown = SimpleNamespace(
                source_file="f.npz",
                source_tensor="t1",
                derived_tensor="t1::down_proj",
                proj="down_proj",
                is_shared_expert=False,
                layer_base=None,
                expert_single_id=None,
            )
            matrix_rows_2: list[dict] = []
            mod.process_one_bank(
                bank_obj=bank_obj_unknown,
                bank_erc=bank_erc,
                layer_idx=None,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows_2,
                quant_rows=[],
                mlx_enabled=False,
                schemes=[],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=lambda *_args, **_kwargs: (None, []),
            )
            self.assertEqual(matrix_rows_2[0]["layer"], -1)
            self.assertIsNone(matrix_rows_2[0]["block4"])

    def test_process_one_bank_quant_rows_map_qdf_fields(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        bank_obj = SimpleNamespace(
            source_file="f.npz",
            source_tensor="layers.5.experts.0.w2.weight",
            derived_tensor="layers.5.experts.0.w2.weight::down_proj",
            proj="down_proj",
            is_shared_expert=False,
            layer_base=5,
            expert_single_id=None,
        )
        bank_erc = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )

        def fake_stats(bank, _cfg_stats, _cache_dir):
            e_count = bank.shape[0]
            return {"mean": np.arange(e_count, dtype=np.float32)}

        def fake_quant(_bank, _schemes, _cfg_stats, _device):
            qdf = pd.DataFrame(
                [
                    {
                        "scheme": "q4",
                        "mode": "symmetric",
                        "bits": 4,
                        "group_size": 32,
                        "expert_id_in_bank": 0,
                        "w_rel_fro": 0.1,
                        "w_rel_max": 0.2,
                        "scale_mean": 1.1,
                        "scale_max": 1.2,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": None,
                    },
                    {
                        "scheme": "q4",
                        "mode": "symmetric",
                        "bits": 4,
                        "group_size": 32,
                        "expert_id_in_bank": 1,
                        "w_rel_fro": 0.3,
                        "w_rel_max": 0.4,
                        "scale_mean": 1.3,
                        "scale_max": 1.4,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": "stub error",
                    },
                ]
            )
            return qdf, ["stub quant warn"]

        matrix_rows: list[dict] = []
        quant_rows: list[dict] = []
        warn_log: list[str] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod.process_one_bank(
                bank_obj=bank_obj,
                bank_erc=bank_erc,
                layer_idx=5,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=True,
                schemes=[{"name": "q4", "enabled": True}],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=fake_quant,
                warn_log=warn_log,
            )

        self.assertEqual(len(matrix_rows), 2)
        self.assertEqual(len(quant_rows), 2)
        self.assertEqual(warn_log, ["stub quant warn"])

        first = quant_rows[0]
        self.assertEqual(first["file"], "f.npz")
        self.assertEqual(first["source_tensor"], "layers.5.experts.0.w2.weight")
        self.assertEqual(first["derived_tensor"], "layers.5.experts.0.w2.weight::down_proj")
        self.assertEqual(first["proj"], "down_proj")
        self.assertEqual(first["layer"], 5)
        self.assertEqual(first["block4"], 1)
        self.assertEqual(first["expert_id"], 0)
        self.assertFalse(first["is_shared_expert"])
        self.assertEqual(first["rows"], 2)
        self.assertEqual(first["cols"], 2)
        self.assertEqual(first["scheme"], "q4")
        self.assertEqual(first["mode"], "symmetric")
        self.assertEqual(first["bits"], 4)
        self.assertEqual(first["group_size"], 32)
        self.assertEqual(first["w_rel_fro"], 0.1)
        self.assertEqual(first["w_rel_max"], 0.2)
        self.assertEqual(first["scale_mean"], 1.1)
        self.assertEqual(first["scale_max"], 1.2)
        self.assertIsNone(first["bias_mean"])
        self.assertIsNone(first["bias_max"])
        self.assertIsNone(first["error"])

        second = quant_rows[1]
        self.assertEqual(second["expert_id"], 1)
        self.assertEqual(second["error"], "stub error")

    def test_process_one_bank_quant_rows_use_minus_one_for_shared_expert(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        bank_obj = SimpleNamespace(
            source_file="f.npz",
            source_tensor="layers.2.shared_expert.w2.weight",
            derived_tensor="layers.2.shared_expert.w2.weight::down_proj",
            proj="down_proj",
            is_shared_expert=True,
            layer_base=2,
            expert_single_id=None,
        )
        bank_erc = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )

        def fake_stats(bank, _cfg_stats, _cache_dir):
            e_count = bank.shape[0]
            return {"mean": np.arange(e_count, dtype=np.float32)}

        def fake_quant(_bank, _schemes, _cfg_stats, _device):
            qdf = pd.DataFrame(
                [
                    {
                        "scheme": "q4",
                        "mode": "symmetric",
                        "bits": 4,
                        "group_size": 32,
                        "expert_id_in_bank": 0,
                        "w_rel_fro": 0.1,
                        "w_rel_max": 0.2,
                        "scale_mean": 1.1,
                        "scale_max": 1.2,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": None,
                    },
                    {
                        "scheme": "q4",
                        "mode": "symmetric",
                        "bits": 4,
                        "group_size": 32,
                        "expert_id_in_bank": 1,
                        "w_rel_fro": 0.3,
                        "w_rel_max": 0.4,
                        "scale_mean": 1.3,
                        "scale_max": 1.4,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": None,
                    },
                ]
            )
            return qdf, []

        with tempfile.TemporaryDirectory() as tmp_dir:
            matrix_rows: list[dict] = []
            quant_rows: list[dict] = []
            mod.process_one_bank(
                bank_obj=bank_obj,
                bank_erc=bank_erc,
                layer_idx=2,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=True,
                schemes=[{"name": "q4", "enabled": True}],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=fake_quant,
                warn_log=[],
            )

        self.assertEqual(len(quant_rows), 2)
        self.assertEqual(quant_rows[0]["expert_id"], -1)
        self.assertEqual(quant_rows[1]["expert_id"], -1)
        self.assertTrue(quant_rows[0]["is_shared_expert"])
        self.assertTrue(quant_rows[1]["is_shared_expert"])

    def test_process_one_bank_quant_rows_use_expert_single_id_for_singleton_bank(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        bank_obj = SimpleNamespace(
            source_file="f.npz",
            source_tensor="layers.3.experts.17.w2.weight",
            derived_tensor="layers.3.experts.17.w2.weight::down_proj",
            proj="down_proj",
            is_shared_expert=False,
            layer_base=3,
            expert_single_id=17,
        )
        bank_erc = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

        def fake_stats(bank, _cfg_stats, _cache_dir):
            e_count = bank.shape[0]
            return {"mean": np.arange(e_count, dtype=np.float32)}

        def fake_quant(_bank, _schemes, _cfg_stats, _device):
            qdf = pd.DataFrame(
                [
                    {
                        "scheme": "q4",
                        "mode": "symmetric",
                        "bits": 4,
                        "group_size": 32,
                        "expert_id_in_bank": 0,
                        "w_rel_fro": 0.1,
                        "w_rel_max": 0.2,
                        "scale_mean": 1.1,
                        "scale_max": 1.2,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": None,
                    }
                ]
            )
            return qdf, []

        with tempfile.TemporaryDirectory() as tmp_dir:
            matrix_rows: list[dict] = []
            quant_rows: list[dict] = []
            mod.process_one_bank(
                bank_obj=bank_obj,
                bank_erc=bank_erc,
                layer_idx=3,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=True,
                schemes=[{"name": "q4", "enabled": True}],
                mlx_device="cpu",
                per_expert_weight_stats=fake_stats,
                mlx_quant_sim=fake_quant,
                warn_log=[],
            )

        self.assertEqual(len(quant_rows), 1)
        self.assertEqual(quant_rows[0]["expert_id"], 17)
        self.assertFalse(quant_rows[0]["is_shared_expert"])

    def test_process_extracted_banks_layer_progression_and_unsupported_warning(self):
        mod = _load_module("collect_pipeline", self.scripts_dir / "collect_pipeline.py")

        extracted = [
            SimpleNamespace(
                bank=np.zeros((2, 1, 2, 2), dtype=np.float32),
                layer_base=7,
                derived_tensor="layers.7.experts.w2",
                source_file="f.npz",
                source_tensor="t0",
                proj="down_proj",
                is_shared_expert=False,
                expert_single_id=None,
            ),
            SimpleNamespace(
                bank=np.zeros((1, 2, 2), dtype=np.float32),
                layer_base=3,
                derived_tensor="layers.3.experts.w2",
                source_file="f.npz",
                source_tensor="t1",
                proj="down_proj",
                is_shared_expert=False,
                expert_single_id=None,
            ),
            SimpleNamespace(
                bank=np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
                layer_base=0,
                derived_tensor="bad.tensor",
                source_file="f.npz",
                source_tensor="tb",
                proj="down_proj",
                is_shared_expert=False,
                expert_single_id=None,
            ),
        ]

        calls: list[tuple[int | None, tuple[int, ...]]] = []

        def fake_process_one_bank_fn(**kwargs):
            bank_erc = kwargs["bank_erc"]
            calls.append((kwargs["layer_idx"], tuple(bank_erc.shape)))

        warn_log: list[str] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod.process_extracted_banks(
                extracted=extracted,
                cfg_stats={"eps": 1e-12},
                cache_idx_dir=Path(tmp_dir),
                matrix_rows=[],
                quant_rows=[],
                mlx_enabled=False,
                schemes=[],
                mlx_device="cpu",
                per_expert_weight_stats=lambda *_args, **_kwargs: {"mean": np.array([0.0], dtype=np.float32)},
                mlx_quant_sim=lambda *_args, **_kwargs: (pd.DataFrame(), []),
                warn_log=warn_log,
                process_one_bank_fn=fake_process_one_bank_fn,
            )

        self.assertEqual(
            calls,
            [
                (7, (1, 2, 2)),
                (8, (1, 2, 2)),
                (3, (1, 2, 2)),
            ],
        )
        self.assertEqual(len(warn_log), 1)
        self.assertIn("unsupported canonical ndim=5", warn_log[0])
        self.assertIn("bad.tensor", warn_log[0])


if __name__ == "__main__":
    unittest.main()
