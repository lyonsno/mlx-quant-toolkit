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
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load collect_data module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProjInferenceUnitTests(unittest.TestCase):
    def setUp(self):
        self.collect_data = _load_collect_data()

    def test_iter_weight_files_accepts_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            weights_path = tmp_path / "weights.npz"
            np.savez(weights_path, arr=np.zeros((1,), dtype=np.float32))

            files = list(
                self.collect_data._iter_weight_files(weights_path, exts=[".npz"])
            )

            self.assertEqual(files, [weights_path])

    def test_iter_weight_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            np.savez(tmp_path / "a.npz", arr=np.zeros((1,), dtype=np.float32))
            (tmp_path / "b.safetensors").write_bytes(b"")

            files = list(
                self.collect_data._iter_weight_files(tmp_path, exts=[".npz"])
            )

            self.assertEqual(set(files), {tmp_path / "a.npz"})

    def test_infer_proj_respects_aliases(self):
        alias_map = {
            "down_proj": ["w2"],
            "gate_proj": ["w1", ".gate."],
            "up_proj": ["w3"],
        }
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.w1.weight", alias_map
        )
        self.assertEqual(proj, "gate_proj")

    def test_infer_proj_does_not_match_substrings(self):
        alias_map = {
            "down_proj": ["w2"],
            "gate_proj": ["w1", ".gate."],
            "up_proj": ["w3"],
        }
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.w13.weight", alias_map
        )
        self.assertIsNone(proj)

    def test_infer_proj_keeps_sentinel_contains(self):
        alias_map = {"gate_proj": [".gate."]}
        proj = self.collect_data._infer_proj(
            "layers.0.experts.0.ffn.gate.weight", alias_map
        )
        self.assertEqual(proj, "gate_proj")

    def test_is_shared_expert_matches_explicit_alias_keyword(self):
        keywords = ["share_expert"]
        self.assertTrue(
            self.collect_data._is_shared_expert(
                "model.layers.4.share_expert.down_proj.weight",
                keywords,
            )
        )

    def test_is_shared_expert_requires_all_keywords_even_with_extra_token(self):
        keywords = ["shared", "expert", "router"]
        self.assertFalse(
            self.collect_data._is_shared_expert(
                "model.layers.4.router.down_proj.weight",
                keywords,
            )
        )
        self.assertTrue(
            self.collect_data._is_shared_expert(
                "model.layers.4.shared.expert.router.down_proj.weight",
                keywords,
            )
        )

    def test_is_shared_expert_returns_false_for_empty_keyword_list(self):
        self.assertFalse(
            self.collect_data._is_shared_expert(
                "model.layers.4.shared.expert.down_proj.weight",
                [],
            )
        )

    def test_is_shared_expert_returns_false_for_whitespace_only_keywords(self):
        self.assertFalse(
            self.collect_data._is_shared_expert(
                "model.layers.4.shared.expert.down_proj.weight",
                ["", "   "],
            )
        )

    def test_is_shared_expert_ignores_blank_keywords_around_real_tokens(self):
        self.assertTrue(
            self.collect_data._is_shared_expert(
                "model.layers.4.shared.down_proj.weight",
                [" shared ", "   "],
            )
        )

    def test_record_proj_issue_coalesces_counts_and_keeps_first_example(self):
        acc = {}
        self.collect_data._record_proj_issue(
            acc,
            context="packed_split",
            rule_name="rule_a",
            raw_proj="w999",
            resolved_proj="w999",
            action="kept_raw",
            source_file="a.npz",
            source_tensor="layers.0.experts.0.w999.weight",
            derived_tensor="layers.0.experts.0.w999.weight::split[rows]::w999",
            suggested_proj="down_proj",
            suggested_match="down_projj",
        )
        self.collect_data._record_proj_issue(
            acc,
            context="packed_split",
            rule_name="rule_a",
            raw_proj="w999",
            resolved_proj="w999",
            action="kept_raw",
            source_file="b.npz",
            source_tensor="layers.1.experts.0.w999.weight",
            derived_tensor="layers.1.experts.0.w999.weight::split[rows]::w999",
            suggested_proj="down_proj",
            suggested_match="down_projj",
        )

        self.assertEqual(len(acc), 1)
        row = next(iter(acc.values()))
        self.assertEqual(row["count"], 2)
        self.assertEqual(row["example_file"], "a.npz")
        self.assertEqual(row["example_source_tensor"], "layers.0.experts.0.w999.weight")
        self.assertEqual(
            row["example_derived_tensor"],
            "layers.0.experts.0.w999.weight::split[rows]::w999",
        )

    def test_suggest_proj_returns_canonical_for_close_typo(self):
        alias_map = {
            "down_proj": ["down_proj", "w2"],
            "gate_proj": ["gate_proj", "w1"],
            "up_proj": ["up_proj", "w3"],
        }
        suggested_proj, suggested_match = self.collect_data._suggest_proj(
            "down_projj",
            alias_map,
        )
        self.assertEqual(suggested_proj, "down_proj")
        self.assertTrue(suggested_match)

    def test_suggest_proj_returns_empty_when_no_close_match(self):
        alias_map = {
            "down_proj": ["down_proj", "w2"],
            "gate_proj": ["gate_proj", "w1"],
            "up_proj": ["up_proj", "w3"],
        }
        suggested_proj, suggested_match = self.collect_data._suggest_proj(
            "completely_unrelated_token",
            alias_map,
        )
        self.assertEqual(suggested_proj, "")
        self.assertEqual(suggested_match, "")


class ProjGroupNormalizationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def _write_npz(self, path: Path, tensors: dict[str, np.ndarray]) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for key, arr in tensors.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{key}.npy", buf.getvalue())

    def _create_stub_mlx(self, root: Path) -> Path:
        stub_root = root / "stub_mlx"
        pkg_dir = stub_root / "mlx"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "core.py").write_text("raise ImportError('stub mlx not available')\n")
        return stub_root

    def _write_config(
        self,
        run_dir: Path,
        model_dir: Path,
        rule: dict,
        proj_aliases: dict[str, list[str]],
        proj_group_strict: bool,
        rules: list[dict] | None = None,
        dump_unmatched_tensors: bool = True,
        expert_regex: str = r"(?:^|\\.)experts\\.(\\d+)(?:\\.|$)",
        shared_expert_keywords: list[str] | None = None,
        include_shared_expert: bool = True,
    ) -> None:
        if rules is None:
            rules = [rule]
        if shared_expert_keywords is None:
            shared_expert_keywords = ["shared", "expert"]
        cfg = {
            "model_path": str(model_dir),
            "scan": {
                "extensions": [".npz"],
                "experts_only": True,
                "include_shared_expert": include_shared_expert,
                "inventory_all_tensors": True,
                "max_files": None,
            },
            "parsing": {
                "layer_regex": r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)",
                "expert_regex": expert_regex,
                "proj_aliases": proj_aliases,
                "shared_expert_keywords": shared_expert_keywords,
                "strict_packed_split": True,
                "proj_group_strict": proj_group_strict,
            },
            "extract_rules": rules,
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
            "debug": {
                "dump_unmatched_tensors": dump_unmatched_tensors,
                "print_progress_every_files": 0,
            },
        }
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def _run_collect(self, run_dir: Path, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ],
            cwd=self.repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def _setup_run(
        self,
        tmp_path: Path,
        tensors: dict[str, np.ndarray],
        rule: dict,
        proj_aliases: dict[str, list[str]],
        proj_group_strict: bool,
        rules: list[dict] | None = None,
        dump_unmatched_tensors: bool = True,
        expert_regex: str = r"(?:^|\\.)experts\\.(\\d+)(?:\\.|$)",
        shared_expert_keywords: list[str] | None = None,
        include_shared_expert: bool = True,
    ) -> tuple[Path, dict]:
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._write_npz(model_dir / "weights.npz", tensors)

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(
            run_dir,
            model_dir,
            rule,
            proj_aliases,
            proj_group_strict,
            rules=rules,
            dump_unmatched_tensors=dump_unmatched_tensors,
            expert_regex=expert_regex,
            shared_expert_keywords=shared_expert_keywords,
            include_shared_expert=include_shared_expert,
        )

        stub_root = self._create_stub_mlx(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"
        return run_dir, env

    def test_experts_only_includes_moe_and_shared_expert_alias_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            shared_proj = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.down_proj.weight": expert_bank,
                "model.layers.0.moe.gate_proj.weight": expert_bank,
                "model.layers.0.moe.up_proj.weight": expert_bank,
                "model.layers.0.share_expert.down_proj.weight": shared_proj,
            }
            moe_rule = {
                "name": "moe_3d_bank",
                "match": r".*moe.*\.(down_proj|gate_proj|up_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                # shared_expert_keywords is conjunctive; configure the explicit
                # alias token directly for this compatibility path.
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertGreater(len(rows), 0)
            self.assertTrue(
                any(
                    row["source_tensor"] == "model.layers.0.moe.down_proj.weight"
                    for row in rows
                )
            )
            self.assertTrue(
                any(
                    row["source_tensor"]
                    == "model.layers.0.share_expert.down_proj.weight"
                    for row in rows
                )
            )

    def test_experts_only_excludes_shared_expert_alias_tensors_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            shared_proj = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.down_proj.weight": expert_bank,
                "model.layers.0.share_expert.down_proj.weight": shared_proj,
            }
            moe_rule = {
                "name": "moe_3d_bank",
                "match": r".*moe.*\.(down_proj|gate_proj|up_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                shared_expert_keywords=["share_expert"],
                include_shared_expert=False,
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(
                    row["source_tensor"] == "model.layers.0.moe.down_proj.weight"
                    for row in rows
                )
            )

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_includes_alias_named_moe_projections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            tensors = {
                "model.layers.0.moe.w2.weight": expert_bank,
            }
            moe_rule = {
                "name": "moe_3d_bank_aliases",
                "match": r".*moe.*\.(w1|w2|w3)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertGreater(len(rows), 0)
            self.assertTrue(
                any(
                    row["source_tensor"] == "model.layers.0.moe.w2.weight"
                    and row["proj"] == "down_proj"
                    for row in rows
                )
            )

    def test_experts_only_includes_moe_expert_id_proj_tensors_before_rule_extraction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.7.w2.weight": expert_matrix,
            }
            moe_rule = {
                "name": "moe_single_expert_alias",
                "match": r".*moe\.(\d+)\.(w1|w2|w3)\.weight$",
                "ndim": 2,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": None,
                    "rows_axis": 0,
                    "cols_axis": 1,
                },
                "expert_group": 1,
                "proj_group": 2,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                len(rows),
                1,
                "A direct .moe.<expert_id>.<proj> tensor should survive experts_only and reach rule extraction.",
            )
            self.assertEqual(rows[0]["source_tensor"], "model.layers.0.moe.7.w2.weight")
            self.assertEqual(rows[0]["proj"], "down_proj")
            self.assertEqual(int(rows[0]["expert_id"]), 7)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_includes_start_of_string_moe_expert_id_proj_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_matrix = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "moe.7.w2.weight": expert_matrix,
            }
            moe_rule = {
                "name": "moe_single_expert_alias",
                "match": r".*moe\.(\d+)\.(w1|w2|w3)\.weight$",
                "ndim": 2,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": None,
                    "rows_axis": 0,
                    "cols_axis": 1,
                },
                "expert_group": 1,
                "proj_group": 2,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                len(rows),
                1,
                "A start-of-string moe.<expert_id>.<proj> tensor should survive experts_only and reach rule extraction.",
            )
            self.assertEqual(rows[0]["source_tensor"], "moe.7.w2.weight")
            self.assertEqual(rows[0]["proj"], "down_proj")
            self.assertEqual(int(rows[0]["expert_id"]), 7)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_keeps_non_strict_raw_moe_proj_group_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            tensor_name = "model.layers.0.moe.down_projj.weight"
            tensors = {
                tensor_name: expert_bank,
            }
            moe_rule = {
                "name": "moe_unmapped_proj_group",
                "match": r".*moe.*\.(down_projj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["source_tensor"] == tensor_name for row in rows))
            self.assertTrue(all(row["proj"] == "down_projj" for row in rows))

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_keeps_non_strict_raw_moe_proj_group_tokens_with_empty_alias_map(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            tensor_name = "model.layers.0.moe.down_projj.weight"
            tensors = {
                tensor_name: expert_bank,
            }
            moe_rule = {
                "name": "moe_unmapped_proj_group_empty_aliases",
                "match": r".*moe.*\.(down_projj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases={},
                proj_group_strict=False,
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["source_tensor"] == tensor_name for row in rows))
            self.assertTrue(all(row["proj"] == "down_projj" for row in rows))

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_keeps_dotted_sentinel_alias_named_moe_tensors_out_of_narrow_gate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            tensor_name = "model.layers.0.moe.gate.weight"
            tensors = {
                tensor_name: expert_bank,
            }
            placeholder_rule = {
                "name": "unused_placeholder_rule",
                "match": r"$^",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2", ".down."],
                "gate_proj": ["gate_proj", "w1", ".gate."],
                "up_proj": ["up_proj", "w3", ".up."],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                placeholder_rule,
                proj_aliases,
                proj_group_strict=False,
                rules=[],
                expert_regex=r"(?:^|\.)moe\.(\d+)(?:\.|$)",
                shared_expert_keywords=["share_expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                rows,
                [],
                "Dotted sentinel aliases should not widen the narrow .moe.<proj> experts_only gate.",
            )

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_keeps_valid_moe_tensor_while_excluding_router_poison_pill(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            router_weights = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.down_proj.weight": expert_bank,
                "model.layers.0.moe.router.w1.weight": router_weights,
            }
            moe_rule = {
                "name": "moe_3d_bank",
                "match": r".*moe.*\.(down_proj|gate_proj|up_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                shared_expert_keywords=["shared", "expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(
                    row["source_tensor"] == "model.layers.0.moe.down_proj.weight"
                    for row in rows
                )
            )

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(
                unmatched_rows,
                [],
                "Router poison-pill tensors should stay excluded even when a valid expert tensor is present in the same run.",
            )

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_trusts_explicit_rule_matches_for_router_like_moe_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expert_bank = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            tensor_name = "model.layers.0.moe.router.w1.weight"
            tensors = {
                tensor_name: expert_bank,
            }
            explicit_router_rule = {
                "name": "explicit_router_moe_bank",
                "match": r".*moe\.router\.(w1)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                explicit_router_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                shared_expert_keywords=["shared", "expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                len(rows),
                2,
                "Explicit enabled rules should be trusted to admit tensors even when the heuristic gate would normally exclude the name.",
            )
            self.assertTrue(all(row["source_tensor"] == tensor_name for row in rows))
            self.assertTrue(all(row["proj"] == "gate_proj" for row in rows))

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(unmatched_rows, [])

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_excludes_non_expert_moe_router_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            router_weights = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.router.weight": router_weights,
            }
            moe_rule = {
                "name": "moe_3d_bank",
                "match": r".*moe.*\.(down_proj|gate_proj|up_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                shared_expert_keywords=["shared", "expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(unmatched_rows),
                0,
                "Router-only .moe. tensors should be excluded before expertish unmatched accounting.",
            )

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_experts_only_excludes_non_expert_moe_tensors_even_with_alias_like_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            router_weights = np.arange(12, dtype=np.float32).reshape(4, 3)
            tensors = {
                "model.layers.0.moe.router.w1.weight": router_weights,
            }
            moe_rule = {
                "name": "moe_3d_bank",
                "match": r".*moe.*\.(down_proj|gate_proj|up_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                moe_rule,
                proj_aliases,
                proj_group_strict=False,
                expert_regex=r"(?:^|\\.)moe\\.(\\d+)(?:\\.|$)",
                shared_expert_keywords=["shared", "expert"],
            )

            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(unmatched_rows),
                0,
                "Alias-like router names should still be excluded before expertish unmatched accounting.",
            )

            run_health = json.loads((run_dir / "logs" / "run_health.json").read_text())
            extraction_summary = run_health.get("extraction_summary", {})
            self.assertEqual(int(extraction_summary.get("unmatched_expertish", -1)), 0)

    def test_proj_group_canonicalizes_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensors = {
                "layers.0.experts.0.w1.weight": arr,
                "layers.0.experts.0.w2.weight": arr,
                "layers.0.experts.0.w3.weight": arr,
            }
            rule = {
                "name": "proj_group_w123",
                "match": r".*experts.*\.(w1|w2|w3)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2"],
                "gate_proj": ["w1"],
                "up_proj": ["w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path, tensors, rule, proj_aliases, proj_group_strict=False
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 6)
            allowed = {"gate_proj", "down_proj", "up_proj"}
            self.assertTrue(all(row["proj"] in allowed for row in rows))
            for row in rows:
                source = row["source_tensor"]
                if source.endswith(".w1.weight"):
                    self.assertEqual(row["proj"], "gate_proj")
                elif source.endswith(".w2.weight"):
                    self.assertEqual(row["proj"], "down_proj")
                elif source.endswith(".w3.weight"):
                    self.assertEqual(row["proj"], "up_proj")
                else:
                    self.fail(f"Unexpected tensor name: {source}")

    def test_proj_group_strict_unmatched_when_alias_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            # Include a known alias token (w1) so fallback inference could "rescue" this tensor
            # unless strict proj_group drops are explicitly marked to skip fallback.
            tensor_name = "layers.0.experts.0.w1.w999.weight"
            tensors = {tensor_name: arr}
            rule = {
                "name": "proj_group_w999",
                "match": r".*experts.*\.(w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2"],
                "gate_proj": ["w1"],
                "up_proj": ["w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path, tensors, rule, proj_aliases, proj_group_strict=True
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            match = next(
                (row for row in unmatched_rows if row["tensor_name"] == tensor_name),
                None,
            )
            self.assertIsNotNone(match)
            self.assertEqual(match["reason"], "proj_group_strict_unmapped")

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            self.assertIn("proj_canonicalization_report", artifacts)

            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))
            strict_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if row.get("warning", "").startswith(
                    "[proj] strict proj_group dropped tensors due to unmapped proj tokens:"
                )
            ]
            kept_raw_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if "[proj] unmapped proj tokens kept raw" in row.get("warning", "")
            ]

            report_meta = artifacts["proj_canonicalization_report"]
            report_rel_path = Path(report_meta["path"])
            self.assertIn("logs", report_rel_path.parts)
            self.assertTrue(
                report_rel_path.name.startswith("proj_canonicalization_report")
            )
            report_path = (
                report_rel_path
                if report_rel_path.is_absolute()
                else run_dir / report_rel_path
            )
            self.assertTrue(report_path.exists())
            with report_path.open(newline="") as handle:
                report_rows = list(csv.DictReader(handle))
            self.assertEqual(int(report_meta["rows"]), len(report_rows))

            dropped_rows = [
                row
                for row in report_rows
                if row.get("context") == "proj_group"
                and row.get("raw_proj") == "w999"
                and row.get("action") == "dropped_strict"
            ]
            self.assertTrue(dropped_rows)
            dropped_occurrences = sum(int(row.get("count", 0)) for row in dropped_rows)
            dropped_unique = len({row.get("raw_proj") for row in dropped_rows})
            expected_strict_warning = (
                "[proj] strict proj_group dropped tensors due to unmapped proj tokens: "
                f"occurrences={dropped_occurrences} "
                f"(unique={dropped_unique}). "
                f"See {report_meta['path']}"
            )
            self.assertEqual(len(strict_warnings), 1)
            self.assertEqual(strict_warnings[0], expected_strict_warning)
            self.assertEqual(len(kept_raw_warnings), 0)

    def test_proj_group_strict_with_empty_alias_map_sets_config_reason_and_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensor_names = [
                "layers.0.experts.0.w1.w999.weight",
                "layers.0.experts.1.w1.w998.weight",
            ]
            tensors = {name: arr for name in tensor_names}
            rule = {
                "name": "proj_group_w998_w999",
                "match": r".*experts.*\.(w998|w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                rule,
                proj_aliases={},
                proj_group_strict=True,
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            unmatched_path = run_dir / "data" / "unmatched_tensors.csv"
            self.assertTrue(unmatched_path.exists())
            with unmatched_path.open(newline="") as handle:
                unmatched_rows = list(csv.DictReader(handle))
            for tensor_name in tensor_names:
                match = next(
                    (row for row in unmatched_rows if row["tensor_name"] == tensor_name),
                    None,
                )
                self.assertIsNotNone(match)
                self.assertEqual(match["reason"], "proj_group_strict_no_alias_map")

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            self.assertNotIn("proj_canonicalization_report", artifacts)

            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))

            self.assertIn("unmatched_tensors", artifacts)
            config_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if row.get("warning", "").startswith(
                    "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty;"
                )
            ]
            kept_raw_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if "[proj] unmapped proj tokens kept raw" in row.get("warning", "")
            ]
            strict_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if row.get("warning", "").startswith(
                    "[proj] strict proj_group dropped tensors due to unmapped proj tokens:"
                )
            ]
            expected_config_warning = (
                "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty; "
                "strict proj_group drops occurred (occurrences=2). "
                f"See {artifacts['unmatched_tensors']['path']} for details."
            )
            self.assertEqual(len(config_warnings), 1)
            self.assertEqual(config_warnings[0], expected_config_warning)
            self.assertEqual(len(kept_raw_warnings), 0)
            self.assertEqual(len(strict_warnings), 0)

    def test_proj_group_strict_with_empty_alias_map_and_unmatched_dump_disabled_warns_how_to_enable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensor_names = [
                "layers.0.experts.0.w1.w999.weight",
                "layers.0.experts.1.w1.w998.weight",
            ]
            tensors = {name: arr for name in tensor_names}
            rule = {
                "name": "proj_group_w998_w999",
                "match": r".*experts.*\.(w998|w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                rule,
                proj_aliases={},
                proj_group_strict=True,
                dump_unmatched_tensors=False,
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 0)

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            self.assertNotIn("proj_canonicalization_report", artifacts)
            self.assertNotIn("unmatched_tensors", artifacts)

            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))

            config_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if row.get("warning", "").startswith(
                    "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty;"
                )
            ]
            expected_config_warning = (
                "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty; "
                "strict proj_group drops occurred (occurrences=2). "
                "Enable debug.dump_unmatched_tensors=true to write unmatched_tensors.*."
            )
            self.assertEqual(len(config_warnings), 1)
            self.assertEqual(config_warnings[0], expected_config_warning)

    def test_proj_group_non_strict_unmapped_token_writes_report_and_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensor_name = "layers.0.experts.0.w999.weight"
            tensors = {tensor_name: arr}
            rule = {
                "name": "proj_group_w999",
                "match": r".*experts.*\.(w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2"],
                "gate_proj": ["w1"],
                "up_proj": ["w3"],
            }
            run_dir, env = self._setup_run(
                tmp_path, tensors, rule, proj_aliases, proj_group_strict=False
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row.get("proj") == "w999" for row in rows))

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))
            proj_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if "[proj] unmapped proj tokens kept raw" in row.get("warning", "")
            ]

            self.assertIn("proj_canonicalization_report", artifacts)
            report_meta = artifacts["proj_canonicalization_report"]
            report_rel_path = Path(report_meta["path"])
            self.assertIn("logs", report_rel_path.parts)
            self.assertTrue(
                report_rel_path.name.startswith("proj_canonicalization_report")
            )
            report_path = (
                report_rel_path
                if report_rel_path.is_absolute()
                else run_dir / report_rel_path
            )
            self.assertTrue(report_path.exists())
            with report_path.open(newline="") as handle:
                report_rows = list(csv.DictReader(handle))
            self.assertEqual(int(report_meta["rows"]), len(report_rows))
            proj_group_rows = [
                row
                for row in report_rows
                if row.get("context") == "proj_group"
                and row.get("raw_proj") == "w999"
                and row.get("action") == "kept_raw"
            ]
            self.assertTrue(proj_group_rows)
            self.assertTrue(all(row.get("resolved_proj") == "w999" for row in proj_group_rows))
            self.assertGreater(report_meta.get("rows", 0), 0)
            kept_raw_rows = [row for row in report_rows if row.get("action") == "kept_raw"]
            packed_split_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "packed_split"
            )
            proj_group_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "proj_group"
            )
            unique_raw = len({row.get("raw_proj") for row in kept_raw_rows})
            total_occurrences = packed_split_occurrences + proj_group_occurrences
            expected_warning = (
                "[proj] unmapped proj tokens kept raw: "
                f"packed_split={packed_split_occurrences}, "
                f"proj_group={proj_group_occurrences} "
                f"(unique={unique_raw}, occurrences={total_occurrences}). "
                f"See {report_meta['path']}"
            )
            self.assertEqual(len(proj_warnings), 1)
            self.assertEqual(proj_warnings[0], expected_warning)
            self.assertEqual(packed_split_occurrences, 0)
            self.assertEqual(proj_group_occurrences, 1)

    def test_single_proj_warning_line_has_context_breakdown_across_contexts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            packed_arr = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
            grouped_arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            tensors = {
                "layers.0.experts.0.gate_proj.weight": packed_arr,
                "layers.0.experts.0.w999.weight": grouped_arr,
            }
            packed_rule = {
                "name": "packed_split_unmapped",
                "match": r".*experts.*\.(gate_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "packed_split": {
                    "axis": "rows",
                    "splits": [3, 3],
                    "projs": ["w1", "down_projj"],
                },
            }
            proj_group_rule = {
                "name": "proj_group_unmapped",
                "match": r".*experts.*\.(w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2", "down_proj"],
                "gate_proj": ["w1", "gate_proj"],
                "up_proj": ["w3", "up_proj"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                proj_group_rule,
                proj_aliases,
                proj_group_strict=False,
                rules=[packed_rule, proj_group_rule],
            )
            self._run_collect(run_dir, env)

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            self.assertIn("proj_canonicalization_report", artifacts)

            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))
            proj_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if "[proj] unmapped proj tokens kept raw" in row.get("warning", "")
            ]

            report_meta = artifacts["proj_canonicalization_report"]
            report_rel_path = Path(report_meta["path"])
            self.assertIn("logs", report_rel_path.parts)
            self.assertTrue(
                report_rel_path.name.startswith("proj_canonicalization_report")
            )
            report_path = (
                report_rel_path
                if report_rel_path.is_absolute()
                else run_dir / report_rel_path
            )
            self.assertTrue(report_path.exists())
            with report_path.open(newline="") as handle:
                report_rows = list(csv.DictReader(handle))
            self.assertEqual(int(report_meta["rows"]), len(report_rows))
            kept_raw_rows = [row for row in report_rows if row.get("action") == "kept_raw"]
            packed_split_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "packed_split"
            )
            proj_group_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "proj_group"
            )
            unique_raw = len({row.get("raw_proj") for row in kept_raw_rows})
            total_occurrences = packed_split_occurrences + proj_group_occurrences
            expected_warning = (
                "[proj] unmapped proj tokens kept raw: "
                f"packed_split={packed_split_occurrences}, "
                f"proj_group={proj_group_occurrences} "
                f"(unique={unique_raw}, occurrences={total_occurrences}). "
                f"See {report_meta['path']}"
            )
            self.assertEqual(len(proj_warnings), 1)
            self.assertEqual(proj_warnings[0], expected_warning)
            self.assertEqual(packed_split_occurrences, 1)
            self.assertEqual(proj_group_occurrences, 1)

    def test_strict_proj_group_drops_have_separate_warning_and_do_not_change_kept_raw_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            packed_arr = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
            grouped_arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            strict_tensor_name = "layers.0.experts.0.w1.w999.weight"
            tensors = {
                "layers.0.experts.0.gate_proj.weight": packed_arr,
                strict_tensor_name: grouped_arr,
            }
            packed_rule = {
                "name": "packed_split_unmapped",
                "match": r".*experts.*\.(gate_proj)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "packed_split": {
                    "axis": "rows",
                    "splits": [3, 3],
                    "projs": ["w1", "down_projj"],
                },
            }
            proj_group_rule = {
                "name": "proj_group_strict_drop",
                "match": r".*experts.*\.(w999)\.weight$",
                "ndim": 3,
                "layout": {
                    "layer_axis": None,
                    "expert_axis": 0,
                    "rows_axis": 1,
                    "cols_axis": 2,
                },
                "proj_group": 1,
            }
            proj_aliases = {
                "down_proj": ["w2", "down_proj"],
                "gate_proj": ["w1", "gate_proj"],
                "up_proj": ["w3", "up_proj"],
            }
            run_dir, env = self._setup_run(
                tmp_path,
                tensors,
                proj_group_rule,
                proj_aliases,
                proj_group_strict=True,
                rules=[packed_rule, proj_group_rule],
            )
            self._run_collect(run_dir, env)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                matrix_rows = list(csv.DictReader(handle))
            self.assertFalse(
                any(row.get("source_tensor") == strict_tensor_name for row in matrix_rows)
            )

            write_manifest = json.loads((run_dir / "logs" / "write_manifest.json").read_text())
            artifacts = write_manifest.get("artifacts", {})
            self.assertIn("warnings", artifacts)
            self.assertIn("proj_canonicalization_report", artifacts)

            warnings_meta = artifacts["warnings"]
            warnings_rel_path = Path(warnings_meta["path"])
            self.assertIn("logs", warnings_rel_path.parts)
            self.assertTrue(warnings_rel_path.name.startswith("warnings"))
            warnings_path = (
                warnings_rel_path
                if warnings_rel_path.is_absolute()
                else run_dir / warnings_rel_path
            )
            self.assertTrue(warnings_path.exists())
            with warnings_path.open(newline="") as handle:
                warning_rows = list(csv.DictReader(handle))
            kept_raw_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if "[proj] unmapped proj tokens kept raw" in row.get("warning", "")
            ]
            strict_warnings = [
                row.get("warning", "")
                for row in warning_rows
                if row.get("warning", "").startswith(
                    "[proj] strict proj_group dropped tensors due to unmapped proj tokens:"
                )
            ]

            report_meta = artifacts["proj_canonicalization_report"]
            report_rel_path = Path(report_meta["path"])
            self.assertIn("logs", report_rel_path.parts)
            self.assertTrue(
                report_rel_path.name.startswith("proj_canonicalization_report")
            )
            report_path = (
                report_rel_path
                if report_rel_path.is_absolute()
                else run_dir / report_rel_path
            )
            self.assertTrue(report_path.exists())
            with report_path.open(newline="") as handle:
                report_rows = list(csv.DictReader(handle))
            self.assertEqual(int(report_meta["rows"]), len(report_rows))

            kept_raw_rows = [row for row in report_rows if row.get("action") == "kept_raw"]
            packed_split_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "packed_split"
            )
            proj_group_occurrences = sum(
                int(row.get("count", 0))
                for row in kept_raw_rows
                if row.get("context") == "proj_group"
            )
            unique_raw = len({row.get("raw_proj") for row in kept_raw_rows})
            total_occurrences = packed_split_occurrences + proj_group_occurrences
            expected_kept_raw_warning = (
                "[proj] unmapped proj tokens kept raw: "
                f"packed_split={packed_split_occurrences}, "
                f"proj_group={proj_group_occurrences} "
                f"(unique={unique_raw}, occurrences={total_occurrences}). "
                f"See {report_meta['path']}"
            )

            dropped_rows = [
                row
                for row in report_rows
                if row.get("action") == "dropped_strict" and row.get("context") == "proj_group"
            ]
            dropped_occurrences = sum(int(row.get("count", 0)) for row in dropped_rows)
            dropped_unique = len({row.get("raw_proj") for row in dropped_rows})
            expected_strict_warning = (
                "[proj] strict proj_group dropped tensors due to unmapped proj tokens: "
                f"occurrences={dropped_occurrences} "
                f"(unique={dropped_unique}). "
                f"See {report_meta['path']}"
            )

            self.assertEqual(len(kept_raw_warnings), 1)
            self.assertEqual(kept_raw_warnings[0], expected_kept_raw_warning)
            self.assertEqual(len(strict_warnings), 1)
            self.assertEqual(strict_warnings[0], expected_strict_warning)
            self.assertEqual(packed_split_occurrences, 1)
            self.assertEqual(proj_group_occurrences, 0)
            self.assertGreater(dropped_occurrences, 0)


if __name__ == "__main__":
    unittest.main()
