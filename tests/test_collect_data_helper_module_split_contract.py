import importlib.util
import json
import io
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == path.resolve():
            return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CollectDataHelperModuleSplitContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def _write_npz_with_key(self, path: Path, key: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{key}.npy", buf.getvalue())

    def test_helper_modules_exist_and_export_expected_api(self):
        expected_exports = {
            "collect_extract.py": [
                "Rule",
                "PackedSplitError",
                "ExtractedBank",
                "_compile_rules",
                "_parse_int_from_regex",
                "_is_shared_expert",
                "_infer_proj",
                "_suggest_proj",
                "_record_proj_issue",
                "_split_along_axis",
                "_canonicalize_layout",
                "_apply_rules",
                "_fallback_extract",
            ],
            "collect_stats.py": [
                "_get_sample_indices",
                "_per_expert_weight_stats",
            ],
            "collect_io.py": [
                "_safe_read_json_dict",
                "_write_json",
                "_write_df",
                "_iter_weight_files",
                "_iter_tensors_from_file",
            ],
            "collect_quant.py": [
                "QUANT_SIM_COLUMNS",
                "_mlx_quant_sim",
            ],
        }

        for filename, symbols in expected_exports.items():
            path = self.scripts_dir / filename
            self.assertTrue(path.exists(), f"Expected helper module file is missing: {path}")
            mod = _load_module(path.stem, path)
            for symbol in symbols:
                self.assertTrue(
                    hasattr(mod, symbol),
                    f"Expected symbol {symbol} in helper module {filename}",
                )

    def test_collect_data_reexports_selected_helpers_from_split_modules(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")
        collect_extract = _load_module("collect_extract", self.scripts_dir / "collect_extract.py")
        collect_stats = _load_module("collect_stats", self.scripts_dir / "collect_stats.py")
        collect_io = _load_module("collect_io", self.scripts_dir / "collect_io.py")
        collect_quant = _load_module("collect_quant", self.scripts_dir / "collect_quant.py")

        self.assertIs(collect_data._split_along_axis, collect_extract._split_along_axis)
        self.assertIs(collect_data._canonicalize_layout, collect_extract._canonicalize_layout)
        self.assertIs(collect_data._infer_proj, collect_extract._infer_proj)
        self.assertIs(collect_data._apply_rules, collect_extract._apply_rules)
        self.assertIs(collect_data.Rule, collect_extract.Rule)
        self.assertIs(collect_data._per_expert_weight_stats, collect_stats._per_expert_weight_stats)
        self.assertIs(collect_data._iter_tensors_from_file, collect_io._iter_tensors_from_file)
        self.assertEqual(collect_data.QUANT_SIM_COLUMNS, collect_quant.QUANT_SIM_COLUMNS)
        self.assertIn("w_rel_spectral", collect_data.QUANT_SIM_COLUMNS)
        self.assertIn("w_gram_cos_drift_sampled_rms", collect_data.QUANT_SIM_COLUMNS)
        self.assertNotIn("w_gram_cos_drift_sampled_max", collect_data.QUANT_SIM_COLUMNS)
        self.assertTrue(callable(collect_data._mlx_quant_sim))
        self.assertTrue(callable(collect_quant._mlx_quant_sim))

    def test_split_helper_behavior_is_locked_in_via_extract_module(self):
        collect_extract = _load_module("collect_extract", self.scripts_dir / "collect_extract.py")

        x = np.arange(12, dtype=np.int32).reshape(3, 4)
        parts = collect_extract._split_along_axis(x, axis=1, splits=[1, 3])

        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(parts[0], x[:, :1])
        np.testing.assert_array_equal(parts[1], x[:, 1:])

    def test_collect_io_safe_read_json_dict_handles_missing_invalid_and_valid(self):
        collect_io = _load_module("collect_io", self.scripts_dir / "collect_io.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing = tmp_path / "missing.json"
            self.assertEqual(collect_io._safe_read_json_dict(missing), {})

            invalid = tmp_path / "invalid.json"
            invalid.write_text("{not json")
            self.assertEqual(collect_io._safe_read_json_dict(invalid), {})

            valid = tmp_path / "valid.json"
            valid.write_text(json.dumps({"alpha": 1, "beta": "x"}))
            self.assertEqual(
                collect_io._safe_read_json_dict(valid),
                {"alpha": 1, "beta": "x"},
            )

    def test_collect_io_write_json_and_write_df_csv_contract(self):
        collect_io = _load_module("collect_io", self.scripts_dir / "collect_io.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            json_path = tmp_path / "nested" / "out.json"
            collect_io._write_json({"b": 2, "a": 1}, json_path)
            self.assertTrue(json_path.exists())
            raw = json_path.read_text()
            self.assertLess(raw.find('"a"'), raw.find('"b"'))

            df = pd.DataFrame([{"k": "v", "n": 7}])
            parquet_target = tmp_path / "data" / "sample.parquet"
            meta = collect_io._write_df(df, parquet_target, fmt="csv", compression=None)
            self.assertEqual(meta["format"], "csv")
            self.assertFalse(meta["fallback"])
            self.assertEqual(meta["error"], "")
            self.assertEqual(meta["path"], parquet_target.with_suffix(".csv"))
            written = pd.read_csv(parquet_target.with_suffix(".csv"))
            self.assertEqual(written.to_dict(orient="records"), [{"k": "v", "n": 7}])

    def test_collect_stats_get_sample_indices_is_deterministic_and_cached(self):
        collect_stats = _load_module("collect_stats", self.scripts_dir / "collect_stats.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            idx1 = collect_stats._get_sample_indices(cache_dir, total=8, k=5, seed=123)
            self.assertEqual(idx1.dtype, np.int64)
            cache_path = cache_dir / "idx_N8_k5_seed123.npy"
            self.assertTrue(cache_path.exists())

            # Prove second call is a true cache hit by injecting a sentinel array that
            # could not come from recomputation (duplicates are impossible with replace=False).
            sentinel = np.array([0, 0, 0, 0, 0], dtype=np.int64)
            np.save(cache_path, sentinel)

            idx2 = collect_stats._get_sample_indices(cache_dir, total=8, k=5, seed=123)
            np.testing.assert_array_equal(idx2, sentinel)

    def test_collect_data_quant_entrypoint_preserves_mx_monkeypatch_contract(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        class StubMx:
            cpu = object()
            gpu = object()

            def set_default_device(self, _device):
                return None

            def array(self, x):
                return x

            def quantize(self, *_args, **_kwargs):
                raise RuntimeError("contract quantize fail")

        original_mx = collect_data.mx
        collect_data.mx = StubMx()
        try:
            bank = np.zeros((2, 4, 4), dtype=np.float32)
            schemes = [{"name": "s1", "mode": "symmetric", "bits": 4, "group_size": 32, "enabled": True}]
            cfg_stats = {"eps": 1e-12}
            df, _warns = collect_data._mlx_quant_sim(bank, schemes, cfg_stats, device="cpu")
        finally:
            collect_data.mx = original_mx

        self.assertGreaterEqual(len(df), 2)
        for err in df["error"].tolist():
            self.assertIn("contract quantize fail", str(err))

    def test_collect_data_fails_fast_when_quant_rows_would_write_reduced_public_schema(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            )

            run_root = tmp_path / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            init_run = _load_module("init_run", self.scripts_dir / "init_run.py")
            init_run.init_run(run_root, "model", "run", str(model_dir))

            run_dir = run_root / "model" / "run"
            cfg_path = run_dir / "analysis_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["output"]["format"] = "csv"
            cfg["output"]["compression"] = None
            cfg["mlx"]["enabled"] = True
            cfg["mlx"]["device"] = "cpu"
            cfg["quant_schemes"] = [
                {
                    "name": "s1",
                    "mode": "symmetric",
                    "bits": 4,
                    "group_size": 32,
                    "enabled": True,
                }
            ]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            original_quant = collect_data._mlx_quant_sim
            original_load_mlx = collect_data._load_mlx

            def fake_quant(_bank, schemes, _cfg_stats, _device):
                rows = []
                for scheme in schemes:
                    rows.append(
                        {
                            "scheme": scheme["name"],
                            "mode": scheme["mode"],
                            "bits": int(scheme.get("bits", 4)),
                            "group_size": int(scheme.get("group_size", 32)),
                            "expert_id_in_bank": 0,
                            "w_rel_fro": 0.1,
                            "w_rel_max": 0.2,
                            "w_gram_cos_drift_sampled_rms": 0.06,
                            "scale_mean": 1.1,
                            "scale_max": 1.2,
                            "bias_mean": None,
                            "bias_max": None,
                            "error": None,
                        }
                    )
                return pd.DataFrame(rows), []

            collect_data._mlx_quant_sim = fake_quant
            collect_data._load_mlx = lambda: object()
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    r"quant_sim is missing required public columns: .*w_rel_spectral",
                ):
                    collect_data._main_impl(
                        Namespace(run_dir=str(run_dir), model_path=None),
                        {
                            "run_dir": run_dir,
                            "manifest": {},
                            "configured_model_path": None,
                            "model_path": None,
                            "cli_overrides": {},
                            "index_status": "not_initialized",
                            "index_searched": False,
                            "index_found": False,
                            "index_active": False,
                            "index_path": None,
                            "index_path_found": None,
                            "index_error": None,
                        },
                    )
            finally:
                collect_data._mlx_quant_sim = original_quant
                collect_data._load_mlx = original_load_mlx

    def test_collect_data_rejects_silent_zero_row_quant_output_when_quantization_was_expected(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            )

            run_root = tmp_path / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            init_run = _load_module("init_run", self.scripts_dir / "init_run.py")
            init_run.init_run(run_root, "model", "run", str(model_dir))

            run_dir = run_root / "model" / "run"
            cfg_path = run_dir / "analysis_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["output"]["format"] = "csv"
            cfg["output"]["compression"] = None
            cfg["mlx"]["enabled"] = True
            cfg["mlx"]["device"] = "cpu"
            cfg["quant_schemes"] = [
                {
                    "name": "s1",
                    "mode": "symmetric",
                    "bits": 4,
                    "group_size": 32,
                    "enabled": True,
                }
            ]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            original_quant = collect_data._mlx_quant_sim
            original_load_mlx = collect_data._load_mlx

            def fake_quant(_bank, _schemes, _cfg_stats, _device):
                return pd.DataFrame(columns=collect_data.QUANT_SIM_COLUMNS), []

            collect_data._mlx_quant_sim = fake_quant
            collect_data._load_mlx = lambda: object()
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    r"layers\.0\.experts\.0\.down_proj\.weight.*expected 1 quant rows.*got 0",
                ):
                    collect_data._main_impl(
                        Namespace(run_dir=str(run_dir), model_path=None),
                        {
                            "run_dir": run_dir,
                            "manifest": {},
                            "configured_model_path": None,
                            "model_path": None,
                            "cli_overrides": {},
                            "index_status": "not_initialized",
                            "index_searched": False,
                            "index_found": False,
                            "index_active": False,
                            "index_path": None,
                            "index_path_found": None,
                            "index_error": None,
                        },
                    )
            finally:
                collect_data._mlx_quant_sim = original_quant
                collect_data._load_mlx = original_load_mlx

    def test_collect_data_rejects_duplicate_enabled_quant_scheme_names_before_runtime(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            )

            run_root = tmp_path / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            init_run = _load_module("init_run", self.scripts_dir / "init_run.py")
            init_run.init_run(run_root, "model", "run", str(model_dir))

            run_dir = run_root / "model" / "run"
            cfg_path = run_dir / "analysis_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["output"]["format"] = "csv"
            cfg["output"]["compression"] = None
            cfg["mlx"]["enabled"] = True
            cfg["mlx"]["device"] = "cpu"
            cfg["quant_schemes"] = [
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
            ]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            original_iter_weight_files = collect_data._iter_weight_files
            original_load_mlx = collect_data._load_mlx
            collect_data._load_mlx = lambda: object()
            collect_data._iter_weight_files = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("_iter_weight_files should not be reached for duplicate quant scheme names")
            )
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    r"duplicate enabled quant_schemes names: dup",
                ):
                    collect_data._main_impl(
                        Namespace(run_dir=str(run_dir), model_path=None),
                        {
                            "run_dir": run_dir,
                            "manifest": {},
                            "configured_model_path": None,
                            "model_path": None,
                            "cli_overrides": {},
                            "index_status": "not_initialized",
                            "index_searched": False,
                            "index_found": False,
                            "index_active": False,
                            "index_path": None,
                            "index_path_found": None,
                            "index_error": None,
                        },
                    )
            finally:
                collect_data._iter_weight_files = original_iter_weight_files
                collect_data._load_mlx = original_load_mlx

    def test_collect_data_allows_duplicate_quant_scheme_names_when_mlx_is_disabled(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            )

            run_root = tmp_path / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            init_run = _load_module("init_run", self.scripts_dir / "init_run.py")
            init_run.init_run(run_root, "model", "run", str(model_dir))

            run_dir = run_root / "model" / "run"
            cfg_path = run_dir / "analysis_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["output"]["format"] = "csv"
            cfg["output"]["compression"] = None
            cfg["mlx"]["enabled"] = False
            cfg["quant_schemes"] = [
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
            ]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            original_load_mlx = collect_data._load_mlx
            collect_data._load_mlx = lambda: (_ for _ in ()).throw(
                AssertionError("_load_mlx should not be reached when mlx.enabled=false")
            )
            try:
                collect_data._main_impl(
                    Namespace(run_dir=str(run_dir), model_path=None),
                    {
                        "run_dir": run_dir,
                        "manifest": {},
                        "configured_model_path": None,
                        "model_path": None,
                        "cli_overrides": {},
                        "index_status": "not_initialized",
                        "index_searched": False,
                        "index_found": False,
                        "index_active": False,
                        "index_path": None,
                        "index_path_found": None,
                        "index_error": None,
                    },
                )
            finally:
                collect_data._load_mlx = original_load_mlx

            self.assertTrue((run_dir / "data" / "matrix_stats.csv").exists())
            self.assertTrue((run_dir / "data" / "quant_sim.csv").exists())

    def test_collect_data_duplicate_quant_scheme_names_beat_later_strict_index_error(self):
        collect_data = _load_module("collect_data", self.scripts_dir / "collect_data.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._write_npz_with_key(
                model_dir / "weights.npz",
                "layers.0.experts.0.down_proj.weight",
                np.arange(16, dtype=np.float32).reshape(1, 4, 4),
            )

            run_root = tmp_path / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            init_run = _load_module("init_run", self.scripts_dir / "init_run.py")
            init_run.init_run(run_root, "model", "run", str(model_dir))

            run_dir = run_root / "model" / "run"
            cfg_path = run_dir / "analysis_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["output"]["format"] = "csv"
            cfg["output"]["compression"] = None
            cfg["mlx"]["enabled"] = True
            cfg["mlx"]["device"] = "cpu"
            cfg["scan"]["strict_index"] = True
            cfg["scan"]["use_safetensors_index_json"] = True
            cfg["quant_schemes"] = [
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
            ]
            cfg_path.write_text(json.dumps(cfg, indent=2))

            original_iter_weight_files = collect_data._iter_weight_files
            original_load_mlx = collect_data._load_mlx
            collect_data._load_mlx = lambda: object()
            collect_data._iter_weight_files = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "_iter_weight_files should not be reached before duplicate quant scheme validation"
                )
            )
            try:
                with self.assertRaisesRegex(
                    ValueError,
                    r"duplicate enabled quant_schemes names: dup",
                ):
                    collect_data._main_impl(
                        Namespace(run_dir=str(run_dir), model_path=None),
                        {
                            "run_dir": run_dir,
                            "manifest": {},
                            "configured_model_path": None,
                            "model_path": None,
                            "cli_overrides": {},
                            "index_status": "not_initialized",
                            "index_searched": False,
                            "index_found": False,
                            "index_active": False,
                            "index_path": None,
                            "index_path_found": None,
                            "index_error": None,
                        },
                    )
            finally:
                collect_data._iter_weight_files = original_iter_weight_files
                collect_data._load_mlx = original_load_mlx


if __name__ == "__main__":
    unittest.main()
