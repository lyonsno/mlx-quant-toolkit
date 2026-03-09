import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPlotsContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.scripts_dir = self.repo_root / "scripts"

    def _run(self, args, env=None, check=True):
        return subprocess.run(
            args,
            cwd=self.repo_root,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def _run_build_plots(self, run_dir: Path, env=None):
        run_env = os.environ.copy()
        run_env["PYTHONWARNINGS"] = "default"
        if env is not None:
            run_env.update(env)
        return self._run(
            [
                sys.executable,
                str(self.scripts_dir / "build_plots.py"),
                "--run-dir",
                str(run_dir),
            ],
            env=run_env,
            check=False,
        )

    def _build_plots_path(self) -> Path:
        return self.scripts_dir / "build_plots.py"

    def _assert_build_plots_entrypoint_exists(self):
        path = self._build_plots_path()
        self.assertTrue(path.exists(), f"Expected plotting entrypoint missing: {path}")

    class _RecordingPyplot:
        def __init__(self):
            self.plot_calls = []
            self.bar_calls = []
            self.saved_paths = []

        def figure(self, *_args, **_kwargs):
            return None

        def plot(self, x, y, **kwargs):
            self.plot_calls.append((list(x), list(y), dict(kwargs)))
            return None

        def bar(self, *_args, **_kwargs):
            if _args:
                x = list(_args[0]) if len(_args) > 0 else []
                y = list(_args[1]) if len(_args) > 1 else []
            else:
                x = []
                y = []
            self.bar_calls.append((x, y, dict(_kwargs)))
            return None

        def xticks(self, *_args, **_kwargs):
            return None

        def xlabel(self, *_args, **_kwargs):
            return None

        def ylabel(self, *_args, **_kwargs):
            return None

        def title(self, *_args, **_kwargs):
            return None

        def legend(self, *_args, **_kwargs):
            return None

        def tight_layout(self, *_args, **_kwargs):
            return None

        def savefig(self, path, *_args, **_kwargs):
            self.saved_paths.append(Path(path))
            return None

        def close(self, *_args, **_kwargs):
            return None

    def _write_csv(self, path: Path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _write_minimal_run_config(self, run_dir: Path, plots=None):
        cfg = {"output": {"format": "csv", "compression": None}}
        if plots is not None:
            cfg["plots"] = plots
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=2))

    def _write_manifest(self, run_dir: Path, artifacts: dict[str, dict]):
        manifest = {
            "generated_at": "2026-03-06T00:00:00Z",
            "requested_format": "csv",
            "requested_compression": None,
            "artifacts": artifacts,
        }
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "tables_write_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _read_json(self, path: Path):
        return json.loads(path.read_text())

    def _assert_manifest_written_pngs_exist(self, run_dir: Path, manifest_artifacts: dict):
        # plots_write_manifest.json is invocation-scoped: it records outputs from
        # this build_plots call and does not attempt to enumerate pre-existing PNGs.
        for entry in manifest_artifacts.values():
            if entry.get("status") != "written":
                continue
            rel_path = str(entry.get("path", "")).strip()
            self.assertTrue(rel_path, f"Written manifest entry missing path: {entry}")
            self.assertTrue(rel_path.startswith("plots/"), f"Written manifest path should stay under plots/: {rel_path}")
            self.assertTrue(rel_path.endswith(".png"), f"Written manifest path should be a PNG: {rel_path}")
            abs_path = run_dir / rel_path
            self.assertTrue(abs_path.exists(), f"Manifest path missing on disk: {abs_path}")
            self.assertGreater(abs_path.stat().st_size, 0, f"Manifest PNG should be non-empty: {abs_path}")

    def _assert_plot_manifest_selection(self, run_dir: Path, requested_artifact_keys, expected_artifact_keys):
        manifest_path = run_dir / "logs" / "plots_write_manifest.json"
        self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
        manifest = self._read_json(manifest_path)
        self.assertEqual(manifest.get("requested_artifact_keys"), list(requested_artifact_keys))
        artifacts = manifest.get("artifacts", {})
        self.assertEqual(sorted(artifacts), sorted(expected_artifact_keys))
        self._assert_manifest_written_pngs_exist(run_dir, artifacts)
        return artifacts

    def _fake_matplotlib_env(self, base_dir: Path) -> dict[str, str]:
        fake_site = base_dir / "fake_site"
        fake_matplotlib = fake_site / "matplotlib"
        fake_matplotlib.mkdir(parents=True, exist_ok=True)
        (fake_matplotlib / "__init__.py").write_text(
            "def use(_backend):\n"
            "    return None\n"
        )
        (fake_matplotlib / "pyplot.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "def figure(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def bar(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def plot(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def xticks(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def xlabel(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def ylabel(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def title(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def legend(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def tight_layout(*_args, **_kwargs):\n"
            "    return None\n"
            "\n"
            "def savefig(path, *_args, **_kwargs):\n"
            "    out = Path(path)\n"
            "    out.parent.mkdir(parents=True, exist_ok=True)\n"
            "    out.write_bytes(b'\\x89PNG\\r\\n\\x1a\\nFAKEPNG')\n"
            "\n"
            "def close(*_args, **_kwargs):\n"
            "    return None\n"
        )
        return {"PYTHONPATH": str(fake_site)}

    def test_build_plots_module_exists_and_exports_api(self):
        self._assert_build_plots_entrypoint_exists()
        module_path = self._build_plots_path()
        mod = _load_module("build_plots", module_path)
        self.assertTrue(hasattr(mod, "build_plots"), "Expected symbol build_plots")
        self.assertTrue(callable(mod.build_plots))

    def test_plot_a_weight_layer_handles_numeric_proj_labels_without_empty_series(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots", self._build_plots_path())
        pyplot = self._RecordingPyplot()

        frame = mod.pd.DataFrame(
            {
                "layer": [0, 1],
                "proj": [0, 0],
                "mean_abs__median": [1.0, 2.0],
            }
        )

        ok = mod._plot_a_weight_layer(frame, Path("/tmp/unused_plot.png"), pyplot)

        self.assertTrue(ok)
        self.assertEqual(len(pyplot.plot_calls), 1)
        xs, ys, kwargs = pyplot.plot_calls[0]
        self.assertEqual(xs, [0, 1])
        self.assertEqual(ys, [1.0, 2.0])
        self.assertEqual(kwargs.get("label"), "0")

    def test_plot_a_weight_global_groups_by_proj_median(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots", self._build_plots_path())
        pyplot = self._RecordingPyplot()

        frame = mod.pd.DataFrame(
            {
                "proj": ["b_proj", "a_proj", "a_proj", "b_proj", "a_proj", "a_proj"],
                "mean_abs__median": [2.0, 1.0, 3.0, 6.0, 100.0, "bad"],
            }
        )

        ok = mod._plot_a_weight_global(frame, Path("/tmp/unused_global_plot.png"), pyplot)

        self.assertTrue(ok)
        self.assertEqual(len(pyplot.bar_calls), 1)
        xs, ys, _kwargs = pyplot.bar_calls[0]
        self.assertEqual(xs, ["a_proj", "b_proj"])
        self.assertEqual(ys, [3.0, 4.0])

    def test_plot_b_quant_global_preserves_proj_and_scheme_pairs(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots", self._build_plots_path())
        pyplot = self._RecordingPyplot()

        frame = mod.pd.DataFrame(
            {
                "proj": ["a_proj", "b_proj", "b_proj"],
                "scheme": ["scheme_a", "scheme_a", "scheme_b"],
                "w_rel_fro__median": [0.1, 0.3, 0.5],
            }
        )

        ok = mod._plot_b_quant_global(frame, Path("/tmp/unused_quant_plot.png"), pyplot)

        self.assertTrue(ok)
        self.assertEqual(len(pyplot.bar_calls), 1)
        xs, ys, _kwargs = pyplot.bar_calls[0]
        self.assertEqual(xs, ["a_proj / scheme_a", "b_proj / scheme_a", "b_proj / scheme_b"])
        self.assertEqual(ys, [0.1, 0.3, 0.5])

    def test_plot_b_quant_global_aggregates_duplicate_proj_scheme_rows_by_median(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots", self._build_plots_path())
        pyplot = self._RecordingPyplot()

        frame = mod.pd.DataFrame(
            {
                "proj": ["a_proj", "a_proj", "b_proj", "b_proj", "b_proj", "b_proj"],
                "scheme": ["scheme_a", "scheme_a", "scheme_a", "scheme_b", "scheme_b", "scheme_b"],
                "w_rel_fro__median": [0.1, 0.9, 0.4, 0.2, 0.3, 0.8],
            }
        )

        ok = mod._plot_b_quant_global(frame, Path("/tmp/unused_quant_dupe_plot.png"), pyplot)

        self.assertTrue(ok)
        self.assertEqual(len(pyplot.bar_calls), 1)
        xs, ys, _kwargs = pyplot.bar_calls[0]
        self.assertEqual(xs, ["a_proj / scheme_a", "b_proj / scheme_a", "b_proj / scheme_b"])
        self.assertEqual(ys, [0.5, 0.4, 0.3])

    def test_build_plots_generates_expected_pngs_from_manifest_tables(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "custom_a_global.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[
                    {"proj": "a_proj", "mean_abs__median": 1.0},
                    {"proj": "b_proj", "mean_abs__median": 2.0},
                ],
            )
            self._write_csv(
                tables_dir / "custom_a_layer.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )
            self._write_csv(
                tables_dir / "custom_b_global.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[
                    {"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.10},
                    {"proj": "a_proj", "scheme": "scheme_b", "w_rel_fro__median": 0.20},
                ],
            )

            self._write_manifest(
                run_dir,
                artifacts={
                    "A_weight_global_summary": {
                        "path": "tables/custom_a_global.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 2,
                    },
                    "A_weight_layer_summary": {
                        "path": "tables/custom_a_layer.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 2,
                    },
                    "B_quant_global_summary": {
                        "path": "tables/custom_b_global.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 2,
                    },
                },
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_plots = [
                run_dir / "plots" / "global" / "A_weight_global_summary__mean_abs__median.png",
                run_dir / "plots" / "layer" / "A_weight_layer_summary__mean_abs__median.png",
                run_dir / "plots" / "global" / "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            ]
            for path in expected_plots:
                self.assertTrue(path.exists(), f"Expected plot output missing: {path}")
                self.assertGreater(path.stat().st_size, 0, f"Expected non-empty image file: {path}")
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            expected_pngs = sorted(path.relative_to(run_dir).as_posix() for path in expected_plots)
            self.assertEqual(actual_pngs, expected_pngs)

    def test_build_plots_writes_plots_write_manifest_for_generated_outputs(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[{"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[{"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1}],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"],
            )
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                    "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
                ],
                "Baseline plotting slice intentionally pins an exact artifact set for deterministic contracts.",
            )
            expected_entries = {
                "A_weight_global_summary__mean_abs__median": {
                    "path": "plots/global/A_weight_global_summary__mean_abs__median.png",
                    "source_artifact": "A_weight_global_summary",
                },
                "A_weight_layer_summary__mean_abs__median": {
                    "path": "plots/layer/A_weight_layer_summary__mean_abs__median.png",
                    "source_artifact": "A_weight_layer_summary",
                },
                "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme": {
                    "path": "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
                    "source_artifact": "B_quant_global_summary",
                },
            }
            for artifact_key, expected in expected_entries.items():
                entry = artifacts[artifact_key]
                self.assertEqual(entry.get("path"), expected["path"])
                self.assertEqual(entry.get("source_artifact"), expected["source_artifact"])
                self.assertEqual(entry.get("format"), "png")
                self.assertEqual(entry.get("status"), "written")
                self.assertEqual(entry.get("error"), "")
                plot_path = run_dir / entry["path"]
                self.assertTrue(plot_path.exists(), f"Manifest path missing on disk: {plot_path}")
                self.assertGreater(plot_path.stat().st_size, 0, f"Expected non-empty plot file: {plot_path}")

            self._assert_manifest_written_pngs_exist(run_dir, artifacts)
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            expected_pngs = sorted(v["path"] for v in expected_entries.values())
            self.assertEqual(actual_pngs, expected_pngs)

    def test_build_plots_respects_configured_artifact_selection_from_manifest_inputs(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "custom_selected_a_global.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")
            self._write_manifest(
                run_dir,
                artifacts={
                    "A_weight_global_summary": {
                        "path": "tables/custom_selected_a_global.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                    "A_weight_layer_summary": {
                        "path": "tables/A_weight_layer_summary.parquet",
                        "format": "parquet",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                    "B_quant_global_summary": {
                        "path": "tables/B_quant_global_summary.parquet",
                        "format": "parquet",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                },
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/A_weight_global_summary__mean_abs__median.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["A_weight_global_summary"],
                expected_artifact_keys=["A_weight_global_summary__mean_abs__median"],
            )
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(entry.get("path"), "plots/global/A_weight_global_summary__mean_abs__median.png")
            self.assertEqual(entry.get("error"), "")

    def test_build_plots_respects_configured_artifact_selection_during_legacy_scan(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/A_weight_global_summary__mean_abs__median.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["A_weight_global_summary"],
                expected_artifact_keys=["A_weight_global_summary__mean_abs__median"],
            )
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(entry.get("path"), "plots/global/A_weight_global_summary__mean_abs__median.png")
            self.assertEqual(entry.get("error"), "")

    def test_build_plots_supports_multi_key_artifact_selection(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_layer_summary", "A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/A_weight_global_summary__mean_abs__median.png",
                "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["A_weight_layer_summary", "A_weight_global_summary"],
                expected_artifact_keys=[
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )
            for artifact_key, source_artifact, expected_path in [
                (
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_global_summary",
                    "plots/global/A_weight_global_summary__mean_abs__median.png",
                ),
                (
                    "A_weight_layer_summary__mean_abs__median",
                    "A_weight_layer_summary",
                    "plots/layer/A_weight_layer_summary__mean_abs__median.png",
                ),
            ]:
                entry = artifacts[artifact_key]
                self.assertEqual(entry.get("status"), "written")
                self.assertEqual(entry.get("source_artifact"), source_artifact)
                self.assertEqual(entry.get("path"), expected_path)
                self.assertEqual(entry.get("error"), "")

    def test_build_plots_respects_quant_artifact_selection_during_legacy_scan(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["B_quant_global_summary"]},
            )

            (tables_dir / "A_weight_global_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[
                    {"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1},
                    {"proj": "b_proj", "scheme": "scheme_b", "w_rel_fro__median": 0.2},
                ],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["B_quant_global_summary"],
                expected_artifact_keys=["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"],
            )
            entry = artifacts["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("source_artifact"), "B_quant_global_summary")
            self.assertEqual(
                entry.get("path"),
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            )
            self.assertEqual(entry.get("error"), "")

    def test_build_plots_respects_quant_artifact_selection_from_manifest_inputs(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["B_quant_global_summary"]},
            )

            self._write_csv(
                tables_dir / "custom_selected_b_global.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[
                    {"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1},
                    {"proj": "b_proj", "scheme": "scheme_b", "w_rel_fro__median": 0.2},
                ],
            )
            # Decoy canonical candidate: if discovery regresses away from manifest
            # precedence, plotting this malformed table would fail the run.
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "wrong_metric"],
                rows=[
                    {"proj": "decoy_proj", "scheme": "decoy_scheme", "wrong_metric": 9.9},
                ],
            )
            (tables_dir / "A_weight_global_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")
            self._write_manifest(
                run_dir,
                artifacts={
                    "A_weight_global_summary": {
                        "path": "tables/A_weight_global_summary.parquet",
                        "format": "parquet",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                    "A_weight_layer_summary": {
                        "path": "tables/A_weight_layer_summary.parquet",
                        "format": "parquet",
                        "fallback": False,
                        "error": "",
                        "rows": 1,
                    },
                    "B_quant_global_summary": {
                        "path": "tables/custom_selected_b_global.csv",
                        "format": "csv",
                        "fallback": False,
                        "error": "",
                        "rows": 2,
                    },
                },
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["B_quant_global_summary"],
                expected_artifact_keys=["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"],
            )
            entry = artifacts["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("source_artifact"), "B_quant_global_summary")
            self.assertEqual(
                entry.get("path"),
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            )
            self.assertEqual(entry.get("error"), "")

    def test_build_plots_falls_back_to_legacy_for_selected_key_when_manifest_is_partial_or_stale(self):
        self._assert_build_plots_entrypoint_exists()
        cases = [
            (
                "global_selected_key_omitted",
                "A_weight_global_summary",
                "omitted",
            ),
            (
                "global_selected_key_blank_path",
                "A_weight_global_summary",
                "blank",
            ),
            (
                "layer_selected_key_omitted",
                "A_weight_layer_summary",
                "omitted",
            ),
            (
                "layer_selected_key_blank_path",
                "A_weight_layer_summary",
                "blank",
            ),
            (
                "quant_selected_key_omitted",
                "B_quant_global_summary",
                "omitted",
            ),
            (
                "quant_selected_key_blank_path",
                "B_quant_global_summary",
                "blank",
            ),
        ]

        plot_artifact_key_map = {
            "A_weight_global_summary": "A_weight_global_summary__mean_abs__median",
            "A_weight_layer_summary": "A_weight_layer_summary__mean_abs__median",
            "B_quant_global_summary": "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
        }
        plot_path_map = {
            "A_weight_global_summary": "plots/global/A_weight_global_summary__mean_abs__median.png",
            "A_weight_layer_summary": "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            "B_quant_global_summary": "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
        }

        for label, selected_key, selected_manifest_mode in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    run_dir = Path(tmp_dir) / "run"
                    fake_env = self._fake_matplotlib_env(Path(tmp_dir))
                    tables_dir = run_dir / "tables"
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    self._write_minimal_run_config(
                        run_dir,
                        plots={"artifact_keys": [selected_key]},
                    )

                    if selected_key == "A_weight_global_summary":
                        self._write_csv(
                            tables_dir / "A_weight_global_summary.csv",
                            fieldnames=["proj", "mean_abs__median"],
                            rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
                        )
                    elif selected_key == "A_weight_layer_summary":
                        self._write_csv(
                            tables_dir / "A_weight_layer_summary.csv",
                            fieldnames=["layer", "proj", "mean_abs__median"],
                            rows=[
                                {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                                {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                            ],
                        )
                    else:
                        self._write_csv(
                            tables_dir / "B_quant_global_summary.csv",
                            fieldnames=["proj", "scheme", "w_rel_fro__median"],
                            rows=[
                                {"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1},
                                {"proj": "b_proj", "scheme": "scheme_b", "w_rel_fro__median": 0.2},
                            ],
                        )

                    for other_key in ("A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"):
                        if other_key == selected_key:
                            continue
                        (tables_dir / f"{other_key}.parquet").write_bytes(b"not-a-real-parquet-file")

                    stale_manifest_key = (
                        "A_weight_layer_summary"
                        if selected_key != "A_weight_layer_summary"
                        else "A_weight_global_summary"
                    )
                    manifest_artifacts = {
                        stale_manifest_key: {
                            "path": f"tables/{stale_manifest_key}.parquet",
                            "format": "parquet",
                            "fallback": False,
                            "error": "",
                            "rows": 1,
                        }
                    }

                    if selected_manifest_mode == "blank":
                        manifest_artifacts[selected_key] = {
                            "path": "   ",
                            "format": "csv",
                            "fallback": False,
                            "error": "",
                            "rows": 1,
                        }

                    self._write_manifest(run_dir, artifacts=manifest_artifacts)

                    result = self._run_build_plots(run_dir, env=fake_env)
                    output = (result.stdout or "") + (result.stderr or "")
                    self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

                    expected_pngs = [plot_path_map[selected_key]]
                    actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
                    self.assertEqual(actual_pngs, expected_pngs)
                    artifacts = self._assert_plot_manifest_selection(
                        run_dir,
                        requested_artifact_keys=[selected_key],
                        expected_artifact_keys=[plot_artifact_key_map[selected_key]],
                    )
                    entry = artifacts[plot_artifact_key_map[selected_key]]
                    self.assertEqual(entry.get("status"), "written")
                    self.assertEqual(entry.get("source_artifact"), selected_key)
                    self.assertEqual(entry.get("error"), "")

    def test_build_plots_rejects_unknown_configured_artifact_keys_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_unknown_key_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary", "unknown_plot_key"]},
            )
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

            class _FailIfDiscoveryCalled:
                def load_plot_tables(self, *_args, **_kwargs):
                    raise AssertionError("plot table discovery should not run before unknown-key validation")

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load before unknown-key validation")
                )
                with self.assertRaisesRegex(ValueError, r"(?i)unsupported plot artifact key(?:s)?: .*unknown_plot_key"):
                    mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertFalse((run_dir / "plots").exists(), "Unsupported plot keys should fail before writing plots")
            self.assertFalse(
                (run_dir / "logs" / "plots_write_manifest.json").exists(),
                "Unsupported plot keys should fail before writing plot manifests",
            )

    def test_build_plots_rejects_unplottable_known_table_artifact_keys_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_unplottable_key_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["B_quant_deltas"]},
            )
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )

            class _FailIfDiscoveryCalled:
                def load_plot_tables(self, *_args, **_kwargs):
                    raise AssertionError("plot table discovery should not run before unsupported-key validation")

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load before unsupported-key validation")
                )
                with self.assertRaisesRegex(ValueError, r"(?i)unsupported plot artifact key(?:s)?: .*B_quant_deltas"):
                    mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertFalse((run_dir / "plots").exists(), "Unsupported plot keys should fail before writing plots")
            self.assertFalse(
                (run_dir / "logs" / "plots_write_manifest.json").exists(),
                "Unsupported plot keys should fail before writing plot manifests",
            )

    def test_build_plots_treats_empty_plots_section_as_default_selection(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_empty_plots_default_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(run_dir, plots={})
            captured = {}

            class _CapturePlotInputs:
                def load_plot_tables(self, _run_dir, artifact_keys):
                    captured["artifact_keys"] = list(artifact_keys)
                    return {}

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _CapturePlotInputs()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when no plot jobs are selected")
                )
                written = mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(written, [])
            self.assertEqual(
                captured.get("artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"],
            )
            self.assertFalse((run_dir / "plots").exists())
            # Policy: default-style config (plots omitted or empty object) should not emit
            # a plots manifest when no explicit selection was requested and no jobs ran.
            self.assertFalse((run_dir / "logs" / "plots_write_manifest.json").exists())

    def test_build_plots_treats_null_artifact_keys_as_default_selection(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_null_artifact_keys_default_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(run_dir, plots={"artifact_keys": None})
            captured = {}

            class _CapturePlotInputs:
                def load_plot_tables(self, _run_dir, artifact_keys):
                    captured["artifact_keys"] = list(artifact_keys)
                    return {}

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _CapturePlotInputs()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when no plot jobs are selected")
                )
                written = mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(written, [])
            self.assertEqual(
                captured.get("artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"],
            )
            self.assertFalse((run_dir / "plots").exists())
            # Policy: null artifact_keys is treated as "unset/default", matching plots={}
            # behavior for no-job runs (no plots manifest emitted).
            self.assertFalse((run_dir / "logs" / "plots_write_manifest.json").exists())

    def test_build_plots_treats_null_plots_section_as_default_selection(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_null_plots_default_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(run_dir, plots=None)
            cfg_path = run_dir / "analysis_config.json"
            cfg = self._read_json(cfg_path)
            cfg["plots"] = None
            cfg_path.write_text(json.dumps(cfg, indent=2))
            captured = {}

            class _CapturePlotInputs:
                def load_plot_tables(self, _run_dir, artifact_keys):
                    captured["artifact_keys"] = list(artifact_keys)
                    return {}

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _CapturePlotInputs()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when no plot jobs are selected")
                )
                written = mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(written, [])
            self.assertEqual(
                captured.get("artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"],
            )
            self.assertFalse((run_dir / "plots").exists())
            # Policy: null plots section is treated as unset/default, matching plots={}
            # behavior for no-job runs (no plots manifest emitted).
            self.assertFalse((run_dir / "logs" / "plots_write_manifest.json").exists())

    def test_build_plots_treats_empty_artifact_keys_as_select_nothing_without_loader_calls(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_empty_artifact_keys_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(run_dir, plots={"artifact_keys": []})

            class _FailIfDiscoveryCalled:
                def __init__(self):
                    self.calls = 0

                def load_plot_tables(self, _run_dir, artifact_keys):
                    self.calls += 1
                    raise AssertionError(
                        f"plot table discovery should not run when plots.artifact_keys selects nothing: {artifact_keys}"
                    )

            discovery = _FailIfDiscoveryCalled()

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: discovery
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when plots.artifact_keys selects nothing")
                )
                written = mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(written, [])
            self.assertEqual(
                discovery.calls,
                0,
                "Explicit empty selection should short-circuit before any plot-table loader calls",
            )
            self.assertFalse((run_dir / "plots").exists())
            # Policy: explicit empty selection is an intentional request, so emit an
            # audit manifest even when the selected set is empty.
            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), [])
            self.assertEqual(manifest.get("artifacts"), {})

    def test_build_plots_deduplicates_artifact_keys_in_first_seen_order(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_dedupe_artifact_keys_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(
                run_dir,
                plots={
                    "artifact_keys": [
                        "A_weight_global_summary",
                        "A_weight_global_summary",
                        "A_weight_layer_summary",
                        "A_weight_global_summary",
                    ]
                },
            )
            captured = {}

            class _CapturePlotInputs:
                def load_plot_tables(self, _run_dir, artifact_keys):
                    captured["artifact_keys"] = list(artifact_keys)
                    return {}

            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: _CapturePlotInputs()
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when no plot jobs are selected")
                )
                with self.assertRaisesRegex(RuntimeError, r"plot artifact\(s\) failed"):
                    mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(
                captured.get("artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary"],
            )
            self.assertFalse((run_dir / "plots").exists())
            # Policy: explicit non-empty selection should be auditable even if discovery
            # returns no tables and no plot jobs are emitted.
            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary"],
            )
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )
            for plot_artifact, source_artifact in [
                ("A_weight_global_summary__mean_abs__median", "A_weight_global_summary"),
                ("A_weight_layer_summary__mean_abs__median", "A_weight_layer_summary"),
            ]:
                entry = artifacts[plot_artifact]
                self.assertEqual(entry.get("status"), "error")
                self.assertEqual(entry.get("path"), "")
                self.assertEqual(entry.get("source_artifact"), source_artifact)
                self.assertEqual(
                    entry.get("error"),
                    f"Selected table artifact not found for plotting: {source_artifact}",
                )

    def test_build_plots_deduplicates_artifact_keys_in_outputs_and_manifest(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={
                    "artifact_keys": [
                        "A_weight_global_summary",
                        "A_weight_global_summary",
                        "A_weight_layer_summary",
                        "A_weight_global_summary",
                    ]
                },
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/A_weight_global_summary__mean_abs__median.png",
                "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)

            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["A_weight_global_summary", "A_weight_layer_summary"],
                expected_artifact_keys=[
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )

    def test_build_plots_manifest_is_invocation_scoped_on_rerun_with_narrower_selection(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[{"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[{"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1}],
            )

            self._write_minimal_run_config(run_dir)
            first = self._run_build_plots(run_dir, env=fake_env)
            first_output = (first.stdout or "") + (first.stderr or "")
            self.assertEqual(first.returncode, 0, f"initial build_plots failed unexpectedly:\n{first_output}")

            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )
            second = self._run_build_plots(run_dir, env=fake_env)
            second_output = (second.stdout or "") + (second.stderr or "")
            self.assertEqual(second.returncode, 0, f"rerun build_plots failed unexpectedly:\n{second_output}")

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            selected_entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(selected_entry.get("status"), "written")
            self.assertEqual(
                selected_entry.get("path"),
                "plots/global/A_weight_global_summary__mean_abs__median.png",
            )
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

            # Invocation scope means old PNGs from earlier runs may remain on disk
            # without being listed in this invocation's manifest.
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(
                actual_pngs,
                [
                    "plots/global/A_weight_global_summary__mean_abs__median.png",
                    "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
                    "plots/layer/A_weight_layer_summary__mean_abs__median.png",
                ],
            )
            manifest_written = sorted(
                entry.get("path", "")
                for entry in artifacts.values()
                if entry.get("status") == "written"
            )
            self.assertEqual(
                manifest_written,
                ["plots/global/A_weight_global_summary__mean_abs__median.png"],
            )

    def test_build_plots_selected_missing_inputs_error_even_with_non_module_loader(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_non_module_loader_contract", self._build_plots_path())
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            class _ObjectLoader:
                def load_plot_tables(self, _run_dir, artifact_keys):
                    self.artifact_keys = list(artifact_keys)
                    return {}

            loader = _ObjectLoader()
            orig_get_plot_inputs = mod._get_plot_inputs_module
            orig_load_pyplot = mod._load_pyplot
            try:
                mod._get_plot_inputs_module = lambda: loader
                mod._load_pyplot = lambda: (_ for _ in ()).throw(
                    AssertionError("matplotlib should not load when selected inputs are missing")
                )
                with self.assertRaisesRegex(RuntimeError, r"plot artifact\(s\) failed"):
                    mod.build_plots(run_dir)
            finally:
                mod._get_plot_inputs_module = orig_get_plot_inputs
                mod._load_pyplot = orig_load_pyplot

            self.assertEqual(loader.artifact_keys, ["A_weight_global_summary"])
            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "error")
            self.assertEqual(entry.get("path"), "")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(
                entry.get("error"),
                "Selected table artifact not found for plotting: A_weight_global_summary",
            )

    def test_build_plots_rejects_malformed_artifact_keys_config_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_artifact_keys_shape_contract", self._build_plots_path())

        cases = [
            ("string", "A_weight_global_summary"),
            ("int_top_level", 7),
            ("object_top_level", {"oops": "shape"}),
            ("blank_entry", [""]),
            ("whitespace_entry", ["   "]),
            ("null_entry", [None]),
            ("object_entry", [{}]),
            ("non_string_entry", ["A_weight_global_summary", 7]),
        ]

        class _FailIfDiscoveryCalled:
            def load_plot_tables(self, *_args, **_kwargs):
                raise AssertionError("plot table discovery should not run before artifact_keys shape validation")

        orig_get_plot_inputs = mod._get_plot_inputs_module
        orig_load_pyplot = mod._load_pyplot
        try:
            mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
            mod._load_pyplot = lambda: (_ for _ in ()).throw(
                AssertionError("matplotlib should not load before artifact_keys shape validation")
            )
            for _label, artifact_keys_value in cases:
                with self.subTest(case=_label):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        run_dir = Path(tmp_dir) / "run"
                        self._write_minimal_run_config(
                            run_dir,
                            plots={"artifact_keys": artifact_keys_value},
                        )
                        tables_dir = run_dir / "tables"
                        tables_dir.mkdir(parents=True, exist_ok=True)
                        self._write_csv(
                            tables_dir / "A_weight_global_summary.csv",
                            fieldnames=["proj", "mean_abs__median"],
                            rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
                        )

                        with self.assertRaisesRegex(
                            ValueError,
                            r"(?i)plots\.artifact_keys",
                        ):
                            mod.build_plots(run_dir)
                        self.assertFalse(
                            (run_dir / "plots").exists(),
                            "Malformed plots.artifact_keys should fail before writing plots",
                        )
                        self.assertFalse(
                            (run_dir / "logs" / "plots_write_manifest.json").exists(),
                            "Malformed plots.artifact_keys should fail before writing plot manifests",
                        )
        finally:
            mod._get_plot_inputs_module = orig_get_plot_inputs
            mod._load_pyplot = orig_load_pyplot

    def test_build_plots_rejects_malformed_plots_container_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_plots_container_shape_contract", self._build_plots_path())

        cases = [
            ("list_container", []),
            ("string_container", "oops"),
            ("int_container", 7),
        ]

        class _FailIfDiscoveryCalled:
            def load_plot_tables(self, *_args, **_kwargs):
                raise AssertionError("plot table discovery should not run before plots container shape validation")

        orig_get_plot_inputs = mod._get_plot_inputs_module
        orig_load_pyplot = mod._load_pyplot
        try:
            mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
            mod._load_pyplot = lambda: (_ for _ in ()).throw(
                AssertionError("matplotlib should not load before plots container shape validation")
            )
            for label, plots_value in cases:
                with self.subTest(case=label):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        run_dir = Path(tmp_dir) / "run"
                        self._write_minimal_run_config(run_dir, plots=plots_value)
                        with self.assertRaisesRegex(ValueError, r"(?i)plots"):
                            mod.build_plots(run_dir)
                        self.assertFalse(
                            (run_dir / "plots").exists(),
                            "Malformed plots container should fail before writing plots",
                        )
                        self.assertFalse(
                            (run_dir / "logs" / "plots_write_manifest.json").exists(),
                            "Malformed plots container should fail before writing plot manifests",
                        )
        finally:
            mod._get_plot_inputs_module = orig_get_plot_inputs
            mod._load_pyplot = orig_load_pyplot

    def test_build_plots_rejects_non_object_analysis_config_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_config_root_shape_contract", self._build_plots_path())
        cases = [
            ("list_root", []),
            ("null_root", None),
        ]

        class _FailIfDiscoveryCalled:
            def load_plot_tables(self, *_args, **_kwargs):
                raise AssertionError("plot table discovery should not run before analysis_config root validation")

        orig_get_plot_inputs = mod._get_plot_inputs_module
        orig_load_pyplot = mod._load_pyplot
        try:
            mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
            mod._load_pyplot = lambda: (_ for _ in ()).throw(
                AssertionError("matplotlib should not load before analysis_config root validation")
            )
            for label, cfg_root in cases:
                with self.subTest(case=label):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        run_dir = Path(tmp_dir) / "run"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / "analysis_config.json").write_text(json.dumps(cfg_root, indent=2))
                        with self.assertRaisesRegex(ValueError, r"(?i)analysis_config\.json.*object"):
                            mod.build_plots(run_dir)
                        self.assertFalse(
                            (run_dir / "plots").exists(),
                            "Malformed analysis_config root should fail before writing plots",
                        )
                        self.assertFalse(
                            (run_dir / "logs" / "plots_write_manifest.json").exists(),
                            "Malformed analysis_config root should fail before writing plot manifests",
                        )
        finally:
            mod._get_plot_inputs_module = orig_get_plot_inputs
            mod._load_pyplot = orig_load_pyplot

    def test_build_plots_rejects_invalid_json_analysis_config_before_discovery(self):
        self._assert_build_plots_entrypoint_exists()
        mod = _load_module("build_plots_invalid_json_contract", self._build_plots_path())

        class _FailIfDiscoveryCalled:
            def load_plot_tables(self, *_args, **_kwargs):
                raise AssertionError("plot table discovery should not run before analysis_config JSON parsing")

        orig_get_plot_inputs = mod._get_plot_inputs_module
        orig_load_pyplot = mod._load_pyplot
        try:
            mod._get_plot_inputs_module = lambda: _FailIfDiscoveryCalled()
            mod._load_pyplot = lambda: (_ for _ in ()).throw(
                AssertionError("matplotlib should not load before analysis_config JSON parsing")
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                run_dir = Path(tmp_dir) / "run"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "analysis_config.json").write_text("{\n")
                with self.assertRaisesRegex(ValueError, r"(?i)invalid json.*analysis_config\.json"):
                    mod.build_plots(run_dir)
                self.assertFalse(
                    (run_dir / "plots").exists(),
                    "Invalid analysis_config JSON should fail before writing plots",
                )
                self.assertFalse(
                    (run_dir / "logs" / "plots_write_manifest.json").exists(),
                    "Invalid analysis_config JSON should fail before writing plot manifests",
                )
        finally:
            mod._get_plot_inputs_module = orig_get_plot_inputs
            mod._load_pyplot = orig_load_pyplot

    def test_build_plots_falls_back_to_legacy_when_tables_manifest_is_corrupted(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "tables_write_manifest.json").write_text('{"artifacts": ')

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            expected_pngs = [
                "plots/global/A_weight_global_summary__mean_abs__median.png",
            ]
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, expected_pngs)
            artifacts = self._assert_plot_manifest_selection(
                run_dir,
                requested_artifact_keys=["A_weight_global_summary"],
                expected_artifact_keys=["A_weight_global_summary__mean_abs__median"],
            )
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(entry.get("error"), "")

    def test_build_plots_falls_back_to_legacy_when_tables_manifest_schema_is_unexpected(self):
        self._assert_build_plots_entrypoint_exists()
        cases = [
            (
                "artifacts_list",
                {"generated_at": "2026-03-07T00:00:00Z", "artifacts": []},
            ),
            (
                "selected_entry_missing_path",
                {
                    "generated_at": "2026-03-07T00:00:00Z",
                    "artifacts": {
                        "A_weight_global_summary": {
                            "format": "csv",
                            "fallback": False,
                            "error": "",
                            "rows": 1,
                        }
                    },
                },
            ),
        ]
        for label, manifest_payload in cases:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    run_dir = Path(tmp_dir) / "run"
                    fake_env = self._fake_matplotlib_env(Path(tmp_dir))
                    tables_dir = run_dir / "tables"
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    self._write_minimal_run_config(
                        run_dir,
                        plots={"artifact_keys": ["A_weight_global_summary"]},
                    )

                    self._write_csv(
                        tables_dir / "A_weight_global_summary.csv",
                        fieldnames=["proj", "mean_abs__median"],
                        rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
                    )
                    (tables_dir / "A_weight_layer_summary.parquet").write_bytes(b"not-a-real-parquet-file")
                    (tables_dir / "B_quant_global_summary.parquet").write_bytes(b"still-not-a-real-parquet-file")

                    logs_dir = run_dir / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / "tables_write_manifest.json").write_text(json.dumps(manifest_payload, indent=2))

                    result = self._run_build_plots(run_dir, env=fake_env)
                    output = (result.stdout or "") + (result.stderr or "")
                    self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

                    expected_pngs = [
                        "plots/global/A_weight_global_summary__mean_abs__median.png",
                    ]
                    actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
                    self.assertEqual(actual_pngs, expected_pngs)
                    artifacts = self._assert_plot_manifest_selection(
                        run_dir,
                        requested_artifact_keys=["A_weight_global_summary"],
                        expected_artifact_keys=["A_weight_global_summary__mean_abs__median"],
                    )
                    entry = artifacts["A_weight_global_summary__mean_abs__median"]
                    self.assertEqual(entry.get("status"), "written")
                    self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
                    self.assertEqual(entry.get("error"), "")

    def test_build_plots_uses_legacy_scan_when_manifest_unavailable(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            layer_plot = run_dir / "plots" / "layer" / "A_weight_layer_summary__mean_abs__median.png"
            self.assertTrue(layer_plot.exists(), f"Expected legacy-discovered layer plot missing: {layer_plot}")
            self.assertGreater(layer_plot.stat().st_size, 0)
            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, ["plots/layer/A_weight_layer_summary__mean_abs__median.png"])

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary", "B_quant_global_summary"],
            )
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_layer_summary__mean_abs__median"])
            entry = artifacts["A_weight_layer_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "written")
            self.assertEqual(entry.get("path"), "plots/layer/A_weight_layer_summary__mean_abs__median.png")
            self.assertEqual(entry.get("source_artifact"), "A_weight_layer_summary")
            self.assertEqual(entry.get("error"), "")
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_surfaces_axis_normalization_errors(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "oops", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )

            result = self._run_build_plots(run_dir)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertNotEqual(result.returncode, 0, "Invalid axis tokens should fail build_plots")
            self.assertIn("Invalid integer token", output)
            self.assertIn("layer", output)
            self.assertFalse(
                (run_dir / "plots").exists(),
                "Axis-normalization failures should happen before writing plots",
            )
            self.assertFalse(
                (run_dir / "logs" / "plots_write_manifest.json").exists(),
                "Axis-normalization failures should happen before writing plot manifests",
            )

    def test_build_plots_fails_with_clear_message_when_plot_dependency_missing(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )

            fake_site = Path(tmp_dir) / "fake_site"
            fake_matplotlib = fake_site / "matplotlib"
            fake_matplotlib.mkdir(parents=True, exist_ok=True)
            (fake_matplotlib / "__init__.py").write_text(
                "raise ImportError('test-forced missing matplotlib dependency')\n"
            )

            env = {
                "PYTHONPATH": str(fake_site),
            }
            result = self._run_build_plots(run_dir, env=env)
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("matplotlib", output.lower())
            self.assertIn("plot", output.lower())
            self.assertFalse(
                (run_dir / "plots").exists(),
                "Missing matplotlib should fail before writing plots",
            )
            self.assertFalse(
                (run_dir / "logs" / "plots_write_manifest.json").exists(),
                "Missing matplotlib should fail before writing plot manifests",
            )

    def test_build_plots_records_explicit_selection_errors_when_plot_dependency_missing(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )

            fake_site = Path(tmp_dir) / "fake_site"
            fake_matplotlib = fake_site / "matplotlib"
            fake_matplotlib.mkdir(parents=True, exist_ok=True)
            (fake_matplotlib / "__init__.py").write_text(
                "raise ImportError('test-forced missing matplotlib dependency')\n"
            )

            env = {"PYTHONPATH": str(fake_site)}
            result = self._run_build_plots(run_dir, env=env)
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("matplotlib", output.lower())
            self.assertIn("plot", output.lower())

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "error")
            self.assertEqual(entry.get("path"), "")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertIn("matplotlib", entry.get("error", "").lower())
            self.assertIn("plot", entry.get("error", "").lower())

            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, [], "Missing matplotlib should not emit plot files")
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_fails_when_recognized_artifact_is_missing_required_columns(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={
                    "artifact_keys": [
                        "A_weight_global_summary",
                        "A_weight_layer_summary",
                        "B_quant_global_summary",
                    ]
                },
            )

            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj"],
                rows=[{"layer": "0", "proj": "a_proj"}],
            )
            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": 1.0}],
            )
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[{"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1}],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("A_weight_layer_summary", output)
            self.assertIn("mean_abs__median", output)

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                    "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
                ],
            )

            global_entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(global_entry.get("status"), "written")
            self.assertEqual(global_entry.get("error"), "")
            self.assertEqual(
                global_entry.get("path"),
                "plots/global/A_weight_global_summary__mean_abs__median.png",
            )
            self.assertTrue((run_dir / global_entry["path"]).exists())

            layer_entry = artifacts["A_weight_layer_summary__mean_abs__median"]
            self.assertEqual(layer_entry.get("status"), "error")
            self.assertEqual(layer_entry.get("path"), "")
            self.assertEqual(layer_entry.get("source_artifact"), "A_weight_layer_summary")
            self.assertIn("required columns", layer_entry.get("error", ""))
            self.assertIn("mean_abs__median", layer_entry.get("error", ""))

            quant_entry = artifacts["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"]
            self.assertEqual(quant_entry.get("status"), "written")
            self.assertEqual(quant_entry.get("error"), "")
            self.assertEqual(
                quant_entry.get("path"),
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            )
            self.assertEqual(quant_entry.get("source_artifact"), "B_quant_global_summary")
            self.assertTrue((run_dir / quant_entry["path"]).exists())

            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_selected_load_errors_and_continues_to_later_artifacts(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={
                    "artifact_keys": [
                        "A_weight_global_summary",
                        "A_weight_layer_summary",
                        "B_quant_global_summary",
                    ]
                },
            )

            (tables_dir / "A_weight_global_summary.parquet").write_bytes(b"not-a-real-parquet-file")
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )
            self._write_csv(
                tables_dir / "B_quant_global_summary.csv",
                fieldnames=["proj", "scheme", "w_rel_fro__median"],
                rows=[{"proj": "a_proj", "scheme": "scheme_a", "w_rel_fro__median": 0.1}],
            )

            result = self._run_build_plots(run_dir, env=fake_env)

            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("[build_plots] error:", output)

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                [
                    "A_weight_global_summary",
                    "A_weight_layer_summary",
                    "B_quant_global_summary",
                ],
            )

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                    "B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme",
                ],
            )

            global_entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(global_entry.get("status"), "error")
            self.assertEqual(global_entry.get("path"), "")
            self.assertEqual(global_entry.get("source_artifact"), "A_weight_global_summary")
            self.assertNotEqual(global_entry.get("error", "").strip(), "")

            layer_entry = artifacts["A_weight_layer_summary__mean_abs__median"]
            self.assertEqual(layer_entry.get("status"), "written")
            self.assertEqual(
                layer_entry.get("path"),
                "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            )
            self.assertEqual(layer_entry.get("source_artifact"), "A_weight_layer_summary")
            self.assertEqual(layer_entry.get("error"), "")
            self.assertTrue((run_dir / layer_entry["path"]).exists())

            quant_entry = artifacts["B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme"]
            self.assertEqual(quant_entry.get("status"), "written")
            self.assertEqual(
                quant_entry.get("path"),
                "plots/global/B_quant_global_summary__w_rel_fro__median_by_proj_and_scheme.png",
            )
            self.assertEqual(quant_entry.get("source_artifact"), "B_quant_global_summary")
            self.assertEqual(quant_entry.get("error"), "")
            self.assertTrue((run_dir / quant_entry["path"]).exists())

            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_all_selected_load_errors_in_manifest(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            (tables_dir / "A_weight_global_summary.parquet").write_bytes(b"not-a-real-parquet-file")

            result = self._run_build_plots(run_dir, env=fake_env)

            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("[build_plots] error:", output)

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "error")
            self.assertEqual(entry.get("path"), "")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertNotEqual(entry.get("error", "").strip(), "")

            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, [], "All-load-error runs should not emit plot files")
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_selected_missing_inputs_and_continues_to_later_artifacts(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary", "A_weight_layer_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )

            result = self._run_build_plots(run_dir, env=fake_env)

            self.assertNotEqual(result.returncode, 0)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertIn("[build_plots] error:", output)

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary"],
            )

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )

            global_entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(global_entry.get("status"), "error")
            self.assertEqual(global_entry.get("path"), "")
            self.assertEqual(global_entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(
                global_entry.get("error"),
                "Selected table artifact not found for plotting: A_weight_global_summary",
            )

            layer_entry = artifacts["A_weight_layer_summary__mean_abs__median"]
            self.assertEqual(layer_entry.get("status"), "written")
            self.assertEqual(
                layer_entry.get("path"),
                "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            )
            self.assertEqual(layer_entry.get("source_artifact"), "A_weight_layer_summary")
            self.assertEqual(layer_entry.get("error"), "")
            self.assertTrue((run_dir / layer_entry["path"]).exists())

            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_all_selected_missing_inputs_in_manifest(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            result = self._run_build_plots(run_dir, env=fake_env)

            self.assertNotEqual(result.returncode, 0)

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "error")
            self.assertEqual(entry.get("path"), "")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(
                entry.get("error"),
                "Selected table artifact not found for plotting: A_weight_global_summary",
            )

            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, [], "All-missing-input runs should not emit plot files")
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_skipped_artifacts_in_manifest(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary", "A_weight_layer_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": ""}],
            )
            self._write_csv(
                tables_dir / "A_weight_layer_summary.csv",
                fieldnames=["layer", "proj", "mean_abs__median"],
                rows=[
                    {"layer": "0", "proj": "a_proj", "mean_abs__median": 1.0},
                    {"layer": "1", "proj": "a_proj", "mean_abs__median": 2.0},
                ],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(
                manifest.get("requested_artifact_keys"),
                ["A_weight_global_summary", "A_weight_layer_summary"],
            )

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(
                sorted(artifacts),
                [
                    "A_weight_global_summary__mean_abs__median",
                    "A_weight_layer_summary__mean_abs__median",
                ],
            )

            global_entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(global_entry.get("status"), "skipped")
            self.assertEqual(global_entry.get("path"), "")
            self.assertEqual(global_entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(global_entry.get("error"), "")

            layer_entry = artifacts["A_weight_layer_summary__mean_abs__median"]
            self.assertEqual(layer_entry.get("status"), "written")
            self.assertEqual(
                layer_entry.get("path"),
                "plots/layer/A_weight_layer_summary__mean_abs__median.png",
            )
            self.assertEqual(layer_entry.get("source_artifact"), "A_weight_layer_summary")
            self.assertEqual(layer_entry.get("error"), "")
            self.assertTrue((run_dir / layer_entry["path"]).exists())

            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_records_all_skipped_runs_in_manifest(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(
                run_dir,
                plots={"artifact_keys": ["A_weight_global_summary"]},
            )

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj", "mean_abs__median"],
                rows=[{"proj": "a_proj", "mean_abs__median": ""}],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")
            self.assertEqual(result.returncode, 0, f"build_plots failed unexpectedly:\n{output}")

            manifest_path = run_dir / "logs" / "plots_write_manifest.json"
            self.assertTrue(manifest_path.exists(), f"Expected plots manifest missing: {manifest_path}")
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("requested_artifact_keys"), ["A_weight_global_summary"])

            artifacts = manifest.get("artifacts", {})
            self.assertEqual(sorted(artifacts), ["A_weight_global_summary__mean_abs__median"])
            entry = artifacts["A_weight_global_summary__mean_abs__median"]
            self.assertEqual(entry.get("status"), "skipped")
            self.assertEqual(entry.get("path"), "")
            self.assertEqual(entry.get("source_artifact"), "A_weight_global_summary")
            self.assertEqual(entry.get("error"), "")

            actual_pngs = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "plots").rglob("*.png"))
            self.assertEqual(actual_pngs, [], "All-skipped runs should not emit plot files")
            self._assert_manifest_written_pngs_exist(run_dir, artifacts)

    def test_build_plots_fails_fast_for_missing_run_dir(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_run_dir = Path(tmp_dir) / "missing-run-dir"
            result = self._run_build_plots(missing_run_dir)
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("analysis_config.json", output)


if __name__ == "__main__":
    unittest.main()
