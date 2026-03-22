import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliEntrypointsContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.pyproject_path = self.repo_root / "pyproject.toml"
        self.cli_module_path = self.repo_root / "mlx_quant_toolkit" / "cli.py"
        self.packaged_scripts_dir = self.repo_root / "mlx_quant_toolkit" / "scripts"

    def _load_pyproject_project_table(self):
        data = tomllib.loads(self.pyproject_path.read_text())
        project = data.get("project", {})
        self.assertIsInstance(project, dict)
        return project

    def _run_cli_help(self, cli_func_name: str, prog_name: str) -> str:
        env = dict(os.environ)
        pythonpath = [str(self.repo_root)]
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"sys.argv=['{prog_name}','--help']; "
                    f"from mlx_quant_toolkit.cli import {cli_func_name}; "
                    f"{cli_func_name}()"
                ),
            ],
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.returncode, 0, output)
        return output

    def test_pyproject_declares_expected_console_scripts(self):
        project = self._load_pyproject_project_table()
        self.assertEqual(
            project.get("scripts"),
            {
                "mlx-quant-init": "mlx_quant_toolkit.cli:init_run_cli",
                "mlx-quant-collect": "mlx_quant_toolkit.cli:collect_data_cli",
                "mlx-quant-build-tables": "mlx_quant_toolkit.cli:build_tables_cli",
                "mlx-quant-build-plots": "mlx_quant_toolkit.cli:build_plots_cli",
            },
        )

    def test_pyproject_declares_package_local_scripts_for_release_builds(self):
        data = tomllib.loads(self.pyproject_path.read_text())
        tool_cfg = data.get("tool", {})
        self.assertIsInstance(tool_cfg, dict)
        setuptools_cfg = tool_cfg.get("setuptools", {})
        self.assertIsInstance(setuptools_cfg, dict)
        packages_cfg = setuptools_cfg.get("packages", [])
        self.assertIsInstance(packages_cfg, list)
        self.assertIn("mlx_quant_toolkit", packages_cfg)
        self.assertIn("mlx_quant_toolkit.scripts", packages_cfg)
        self.assertNotIn("scripts", packages_cfg)
        package_dir_cfg = setuptools_cfg.get("package-dir", {})
        self.assertFalse(
            package_dir_cfg,
            "CLI packaging should not remap mlx_quant_toolkit.scripts to the top-level scripts/ tree",
        )

    def test_repo_contains_package_local_scripts_tree(self):
        self.assertTrue(
            self.packaged_scripts_dir.exists(),
            f"Expected package-local scripts dir missing: {self.packaged_scripts_dir}",
        )
        self.assertTrue(
            (self.packaged_scripts_dir / "__init__.py").exists(),
            "Expected package-local scripts package marker",
        )

    def test_cli_module_exports_expected_wrappers(self):
        self.assertTrue(
            self.cli_module_path.exists(),
            f"Expected CLI shim missing: {self.cli_module_path}",
        )
        mod = _load_module("mlx_quant_toolkit_cli_exports", self.cli_module_path)
        for wrapper_name in (
            "init_run_cli",
            "collect_data_cli",
            "build_tables_cli",
            "build_plots_cli",
        ):
            self.assertTrue(hasattr(mod, wrapper_name), f"Expected wrapper missing: {wrapper_name}")
            self.assertTrue(callable(getattr(mod, wrapper_name)), f"Expected callable wrapper: {wrapper_name}")

    def test_cli_wrappers_dispatch_to_expected_script_mains(self):
        self.assertTrue(
            self.cli_module_path.exists(),
            f"Expected CLI shim missing: {self.cli_module_path}",
        )
        mod = _load_module("mlx_quant_toolkit_cli_dispatch", self.cli_module_path)

        for wrapper_name, script_name in (
            ("init_run_cli", "init_run"),
            ("collect_data_cli", "collect_data"),
            ("build_tables_cli", "build_tables"),
            ("build_plots_cli", "build_plots"),
        ):
            with self.subTest(wrapper=wrapper_name):
                stub_main = mock.Mock()
                stub_module = SimpleNamespace(main=stub_main)
                with mock.patch.object(mod, "_get_script_module", return_value=stub_module) as get_module:
                    getattr(mod, wrapper_name)()
                get_module.assert_called_once_with(script_name)
                stub_main.assert_called_once_with()

    def test_cli_get_script_module_prefers_package_local_namespace(self):
        self.assertTrue(
            self.cli_module_path.exists(),
            f"Expected CLI shim missing: {self.cli_module_path}",
        )
        mod = _load_module("mlx_quant_toolkit_cli_package_local", self.cli_module_path)
        with mock.patch.object(
            mod,
            "_load_local_script_module",
            side_effect=AssertionError("local file fallback should not be needed when package-local scripts exist"),
        ):
            script_mod = mod._get_script_module("init_run")
        self.assertEqual(script_mod.__name__, "mlx_quant_toolkit.scripts.init_run")

    def test_cli_init_wrapper_ignores_shadowed_top_level_scripts_package(self):
        self.assertTrue(
            self.cli_module_path.exists(),
            f"Expected CLI shim missing: {self.cli_module_path}",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_pkg = tmp_path / "scripts"
            fake_pkg.mkdir()
            (fake_pkg / "__init__.py").write_text("")
            (fake_pkg / "init_run.py").write_text(
                "def main():\n"
                "    raise RuntimeError('shadowed fake init_run executed')\n"
            )
            env = dict(os.environ)
            pythonpath = [str(tmp_path), str(self.repo_root)]
            existing = env.get("PYTHONPATH")
            if existing:
                pythonpath.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.argv=['mlx-quant-init','--help']; "
                        "from mlx_quant_toolkit.cli import init_run_cli; "
                        "init_run_cli()"
                    ),
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("usage: mlx-quant-init", output)
        self.assertNotIn("shadowed fake init_run executed", output)

    def test_cli_help_texts_describe_each_pipeline_stage(self):
        cases = (
            (
                "init_run_cli",
                "mlx-quant-init",
                [
                    "Create a new analysis run directory",
                    "Root directory where run directories are created",
                    "Stable model identifier used under the run root",
                    "Human-readable run name within the model directory",
                ],
            ),
            (
                "collect_data_cli",
                "mlx-quant-collect",
                [
                    "Scan model files, extract expert matrices, compute stats, and optionally run MLX quantization simulation",
                    "Run directory created by init_run",
                    "Override the model_path from analysis_config.json",
                ],
            ),
            (
                "build_tables_cli",
                "mlx-quant-build-tables",
                [
                    "Aggregate matrix_stats and quant_sim into summary tables",
                    "Run directory containing data/ artifacts from collect_data",
                ],
            ),
            (
                "build_plots_cli",
                "mlx-quant-build-plots",
                [
                    "Build plots from table artifacts",
                    "Run directory containing tables/ and logs/",
                ],
            ),
        )

        for cli_func_name, prog_name, expected_phrases in cases:
            with self.subTest(cli=prog_name):
                output = self._run_cli_help(cli_func_name, prog_name)
                normalized_output = " ".join(output.split())
                self.assertIn(f"usage: {prog_name}", normalized_output)
                for phrase in expected_phrases:
                    self.assertIn(phrase, normalized_output)


if __name__ == "__main__":
    unittest.main()
