import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


class PackedSplitStrictnessTests(unittest.TestCase):
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

    def _run_collect(self, run_dir: Path, env: dict, check: bool):
        return subprocess.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "collect_data.py"),
                "--run-dir",
                str(run_dir),
            ],
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _write_config(
        self,
        run_dir: Path,
        model_dir: Path,
        strict_packed_split: bool,
        proj_aliases: dict[str, list[str]] | None = None,
        packed_split_projs: list[str] | None = None,
    ) -> None:
        if proj_aliases is None:
            proj_aliases = {
                "down_proj": ["down_proj"],
                "gate_proj": ["gate_proj"],
                "up_proj": ["up_proj"],
            }
        if packed_split_projs is None:
            packed_split_projs = ["gate_proj", "down_proj"]

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
                "layer_regex": r"(?:^|\.)layers\.(\d+)(?:\.|$)",
                "expert_regex": r"(?:^|\.)experts\.(\d+)(?:\.|$)",
                "proj_aliases": proj_aliases,
                "shared_expert_keywords": ["shared", "expert"],
                "strict_packed_split": strict_packed_split,
            },
            "extract_rules": [
                {
                    "name": "packed_split_test",
                    "match": r".*experts.*\.(gate_proj)\.weight$",
                    "ndim": 3,
                    "layout": {"layer_axis": None, "expert_axis": 0, "rows_axis": 1, "cols_axis": 2},
                    "packed_split": {
                        "axis": "rows",
                        "splits": [3, 3],
                        "projs": packed_split_projs,
                    },
                }
            ],
            "mlx": {"enabled": False, "device": "cpu"},
            "stats": {
                "eps": 1e-12,
                "sample_per_matrix": 8,
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

    def _setup_run(
        self,
        tmp_path: Path,
        strict_packed_split: bool,
        arr: np.ndarray = None,
        proj_aliases: dict[str, list[str]] | None = None,
        packed_split_projs: list[str] | None = None,
    ) -> tuple[Path, dict]:
        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if arr is None:
            arr = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        self._write_npz_with_key(
            model_dir / "weights.npz",
            "layers.0.experts.0.gate_proj.weight",
            arr,
        )

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(
            run_dir,
            model_dir,
            strict_packed_split,
            proj_aliases=proj_aliases,
            packed_split_projs=packed_split_projs,
        )

        stub_root = self._create_stub_mlx(tmp_path)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stub_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "default"
        return run_dir, env

    def test_packed_split_mismatch_fails_when_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, env = self._setup_run(Path(tmp_dir), strict_packed_split=True)
            result = self._run_collect(run_dir, env, check=False)
            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("PackedSplitError", output)

    def test_packed_split_mismatch_warns_and_falls_back_when_non_strict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir, env = self._setup_run(Path(tmp_dir), strict_packed_split=False)
            result = self._run_collect(run_dir, env, check=True)
            output = (result.stdout or "") + (result.stderr or "")

            warnings_path = run_dir / "logs" / "warnings.csv"
            self.assertTrue(warnings_path.exists())
            warnings_text = warnings_path.read_text()
            self.assertIn("packed_split failed", warnings_text)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            self.assertIn("matrix_stats rows", output)
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row.get("proj") == "gate_proj" for row in rows))
            for row in rows:
                self.assertEqual(int(row["rows"]), 4)
                self.assertEqual(int(row["cols"]), 4)

    def test_packed_split_success_produces_expected_projs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
            run_dir, env = self._setup_run(Path(tmp_dir), strict_packed_split=True, arr=arr)
            self._run_collect(run_dir, env, check=True)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())

            with matrix_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            projs = {"gate_proj", "down_proj"}
            proj_rows = [row for row in rows if row.get("proj") in projs]

            self.assertTrue(proj_rows)
            self.assertEqual({row["proj"] for row in proj_rows}, projs)
            for row in proj_rows:
                self.assertEqual(int(row["rows"]), 3)
                self.assertEqual(int(row["cols"]), 4)

            warnings_path = run_dir / "logs" / "warnings.csv"
            if warnings_path.exists():
                warnings_text = warnings_path.read_text()
                self.assertNotIn("packed_split failed", warnings_text)

    def test_packed_split_projs_are_canonicalized_via_proj_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                Path(tmp_dir),
                strict_packed_split=True,
                arr=arr,
                proj_aliases=proj_aliases,
                packed_split_projs=["w1", "w2"],
            )
            self._run_collect(run_dir, env, check=True)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            split_rows = [
                row
                for row in rows
                if row.get("source_tensor") == "layers.0.experts.0.gate_proj.weight"
                and "::split[rows]::" in row.get("derived_tensor", "")
            ]
            self.assertEqual(len(split_rows), 4)
            self.assertEqual({row["proj"] for row in split_rows}, {"gate_proj", "down_proj"})
            self.assertFalse(any(row["proj"] in {"w1", "w2"} for row in split_rows))
            self.assertEqual(
                Counter(row["proj"] for row in split_rows),
                Counter({"gate_proj": 2, "down_proj": 2}),
            )
            for row in split_rows:
                self.assertEqual(int(row["rows"]), 3)
                self.assertEqual(int(row["cols"]), 4)
            derived_suffixes = [
                row["derived_tensor"].split("::split[rows]::", 1)[1]
                for row in split_rows
            ]
            self.assertEqual(set(derived_suffixes), {"gate_proj", "down_proj"})
            self.assertEqual(
                Counter(derived_suffixes),
                Counter({"gate_proj": 2, "down_proj": 2}),
            )
            self.assertFalse(
                any(
                    "::split[rows]::w1" in row["derived_tensor"]
                    or "::split[rows]::w2" in row["derived_tensor"]
                    for row in split_rows
                )
            )

    def test_packed_split_canonical_keys_are_case_normalized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            arr = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
            proj_aliases = {
                "down_proj": ["down_proj", "w2"],
                "gate_proj": ["gate_proj", "w1"],
                "up_proj": ["up_proj", "w3"],
            }
            run_dir, env = self._setup_run(
                Path(tmp_dir),
                strict_packed_split=True,
                arr=arr,
                proj_aliases=proj_aliases,
                packed_split_projs=["GATE_PROJ", "Down_Proj"],
            )
            self._run_collect(run_dir, env, check=True)

            matrix_path = run_dir / "data" / "matrix_stats.csv"
            self.assertTrue(matrix_path.exists())
            with matrix_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            split_rows = [
                row
                for row in rows
                if row.get("source_tensor") == "layers.0.experts.0.gate_proj.weight"
                and "::split[rows]::" in row.get("derived_tensor", "")
            ]
            self.assertEqual(len(split_rows), 4)
            self.assertEqual({row["proj"] for row in split_rows}, {"gate_proj", "down_proj"})
            self.assertEqual(
                Counter(row["proj"] for row in split_rows),
                Counter({"gate_proj": 2, "down_proj": 2}),
            )
            derived_suffixes = [
                row["derived_tensor"].split("::split[rows]::", 1)[1]
                for row in split_rows
            ]
            self.assertEqual(set(derived_suffixes), {"gate_proj", "down_proj"})
            self.assertEqual(
                Counter(derived_suffixes),
                Counter({"gate_proj": 2, "down_proj": 2}),
            )


if __name__ == "__main__":
    unittest.main()
