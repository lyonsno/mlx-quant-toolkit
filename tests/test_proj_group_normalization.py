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
    ) -> None:
        if rules is None:
            rules = [rule]
        cfg = {
            "model_path": str(model_dir),
            "scan": {
                "extensions": [".npz"],
                "experts_only": True,
                "include_shared_expert": True,
                "inventory_all_tensors": True,
                "max_files": None,
            },
            "parsing": {
                "layer_regex": r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)",
                "expert_regex": r"(?:^|\\.)experts\\.(\\d+)(?:\\.|$)",
                "proj_aliases": proj_aliases,
                "shared_expert_keywords": ["shared", "expert"],
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
            "debug": {"dump_unmatched_tensors": True, "print_progress_every_files": 0},
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
        )

        stub_root = self._create_stub_mlx(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"
        return run_dir, env

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
