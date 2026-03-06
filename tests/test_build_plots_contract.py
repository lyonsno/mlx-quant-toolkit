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

    def _write_minimal_run_config(self, run_dir: Path):
        (run_dir / "analysis_config.json").write_text(
            json.dumps({"output": {"format": "csv", "compression": None}}, indent=2)
        )

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
                run_dir / "plots" / "global" / "B_quant_global_summary__w_rel_fro__median.png",
            ]
            for path in expected_plots:
                self.assertTrue(path.exists(), f"Expected plot output missing: {path}")
                self.assertGreater(path.stat().st_size, 0, f"Expected non-empty image file: {path}")

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

    def test_build_plots_fails_when_recognized_artifact_is_missing_required_columns(self):
        self._assert_build_plots_entrypoint_exists()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            fake_env = self._fake_matplotlib_env(Path(tmp_dir))
            tables_dir = run_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            self._write_minimal_run_config(run_dir)

            self._write_csv(
                tables_dir / "A_weight_global_summary.csv",
                fieldnames=["proj"],
                rows=[{"proj": "a_proj"}],
            )

            result = self._run_build_plots(run_dir, env=fake_env)
            output = (result.stdout or "") + (result.stderr or "")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("A_weight_global_summary", output)
            self.assertIn("mean_abs__median", output)

    def test_build_plots_fails_fast_for_missing_run_dir(self):
        self._assert_build_plots_entrypoint_exists()
        missing_run_dir = Path(tempfile.gettempdir()) / "build-plots-missing-run-dir-contract"
        result = self._run_build_plots(missing_run_dir)
        output = (result.stdout or "") + (result.stderr or "")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("analysis_config.json", output)


if __name__ == "__main__":
    unittest.main()
