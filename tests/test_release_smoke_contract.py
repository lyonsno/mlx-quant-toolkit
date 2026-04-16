import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseSmokeContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script_path = self.repo_root / "scripts" / "release_smoke.py"
        self.repo_python = self.repo_root / ".venv" / "bin" / "python"

    def test_release_smoke_module_exports_runner_and_entrypoint(self):
        self.assertTrue(
            self.script_path.exists(),
            f"Expected release smoke script at {self.script_path}",
        )
        mod = _load_module("release_smoke_contract_exports", self.script_path)
        self.assertTrue(callable(mod.run_release_smoke))
        self.assertTrue(callable(mod.main))

    def test_run_release_smoke_dry_run_does_not_stage_repo_copy(self):
        mod = _load_module("release_smoke_contract_dry_run", self.script_path)

        buf = io.StringIO()
        with mock.patch.object(
            mod,
            "_stage_project_snapshot",
            side_effect=AssertionError("dry-run should not stage a project snapshot"),
        ):
            with redirect_stdout(buf):
                mod.run_release_smoke(
                    self.repo_root,
                    dry_run=True,
                    uv_cache_dir="/tmp/uv-cache",
                    python_executable=sys.executable,
                )

        output = buf.getvalue()
        self.assertIn("[release-smoke][dry-run] build", output)
        self.assertIn("uv build", output)

    def test_run_release_smoke_rejects_multiple_wheels_before_install(self):
        mod = _load_module("release_smoke_contract_multiple_wheels", self.script_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            class _FixedTempDir:
                def __enter__(self_inner):
                    return str(tmp_path)

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            def _fake_stage_project_snapshot(project_root, staged_repo):
                staged_repo.mkdir(parents=True, exist_ok=True)

            def _fake_run_command(cmd, cwd, env):
                cmd_list = list(cmd)
                if cmd_list[:2] == ["uv", "build"]:
                    out_dir = Path(cmd_list[-1])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "one.whl").write_text("wheel-1")
                    (out_dir / "two.whl").write_text("wheel-2")
                    return type("_Result", (), {"stdout": "", "stderr": ""})()
                raise AssertionError(f"unexpected command after wheel guard: {cmd_list}")

            with mock.patch.object(mod.tempfile, "TemporaryDirectory", return_value=_FixedTempDir()):
                with mock.patch.object(mod, "_stage_project_snapshot", side_effect=_fake_stage_project_snapshot):
                    with mock.patch.object(mod, "_run_command", side_effect=_fake_run_command):
                        with self.assertRaisesRegex(RuntimeError, "expected exactly one wheel"):
                            mod.run_release_smoke(
                                self.repo_root,
                                dry_run=False,
                                uv_cache_dir="/tmp/uv-cache",
                                python_executable=sys.executable,
                            )

    def test_run_release_smoke_stages_clean_snapshot_without_untracked_package_files(self):
        mod = _load_module("release_smoke_contract_clean_snapshot", self.script_path)

        with tempfile.TemporaryDirectory() as project_tmp:
            project_root = Path(project_tmp)
            (project_root / "mlx_quant_toolkit").mkdir(parents=True, exist_ok=True)
            (project_root / "pyproject.toml").write_text(
                "[build-system]\n"
                "requires = ['setuptools>=61']\n"
                "build-backend = 'setuptools.build_meta'\n"
                "[project]\n"
                "name = 'snapshot-fixture'\n"
                "version = '0.0.0'\n"
            )
            (project_root / "README.md").write_text("fixture\n")
            (project_root / "LICENSE").write_text("fixture\n")
            (project_root / "mlx_quant_toolkit" / "__init__.py").write_text("__all__ = []\n")

            subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "config", "user.email", "tests@example.com"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Release Smoke Tests"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "add", "pyproject.toml", "README.md", "LICENSE", "mlx_quant_toolkit/__init__.py"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "fixture"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )

            poison_path = project_root / "mlx_quant_toolkit" / "untracked_poison.py"
            poison_path.write_text("raise RuntimeError('should not be staged')\n")

            with tempfile.TemporaryDirectory() as stage_tmp:
                stage_root = Path(stage_tmp)

                class _FixedTempDir:
                    def __enter__(self_inner):
                        return str(stage_root)

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                def _fake_run_command(cmd, cwd, env):
                    if list(cmd[:2]) == ["uv", "build"]:
                        self.assertFalse((cwd / "mlx_quant_toolkit" / "untracked_poison.py").exists())
                        raise RuntimeError("stop after staged snapshot inspection")
                    raise AssertionError(f"unexpected command: {cmd}")

                with mock.patch.object(mod.tempfile, "TemporaryDirectory", return_value=_FixedTempDir()):
                    with mock.patch.object(mod, "_run_command", side_effect=_fake_run_command):
                        with self.assertRaisesRegex(RuntimeError, "stop after staged snapshot inspection"):
                            mod.run_release_smoke(
                                project_root,
                                dry_run=False,
                                uv_cache_dir="/tmp/uv-cache",
                                python_executable=sys.executable,
                            )

    def test_python_runtime_site_packages_prefers_selected_venv(self):
        self.assertTrue(
            self.repo_python.exists(),
            f"Expected repo-local interpreter at {self.repo_python}",
        )
        mod = _load_module("release_smoke_contract_runtime_paths", self.script_path)
        env = mod._installed_surface_env(dict(os.environ))

        expected = subprocess.run(
            [
                str(self.repo_python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        runtime_paths = mod._python_runtime_site_packages(
            str(self.repo_python),
            env,
            self.repo_root,
        )

        self.assertIn(expected, runtime_paths)
        self.assertTrue(
            any(path.startswith(str(self.repo_root / ".venv")) for path in runtime_paths),
            runtime_paths,
        )

    def test_cli_help_probe_fails_when_prefix_package_is_missing_even_if_runtime_site_has_fallback(self):
        mod = _load_module("release_smoke_contract_cli_prefix_ownership", self.script_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            install_prefix = tmp_path / "install"
            bin_dir = install_prefix / "bin"
            prefix_purelib = (
                install_prefix
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            runtime_site = tmp_path / "runtime-site"
            runtime_pkg = runtime_site / "mlx_quant_toolkit"
            bin_dir.mkdir(parents=True, exist_ok=True)
            prefix_purelib.mkdir(parents=True, exist_ok=True)
            runtime_pkg.mkdir(parents=True, exist_ok=True)

            (runtime_pkg / "__init__.py").write_text("")
            (runtime_pkg / "cli.py").write_text(
                "import sys\n"
                "def init_run_cli():\n"
                "    if '--help' in sys.argv:\n"
                "        print('usage: fallback-package')\n"
                "        return 0\n"
                "    raise SystemExit(2)\n"
            )

            wrapper_path = bin_dir / "mlx-quant-init"
            wrapper_path.write_text(
                f"#!{sys.executable}\n"
                "from mlx_quant_toolkit.cli import init_run_cli\n"
                "raise SystemExit(init_run_cli())\n"
            )
            os.chmod(wrapper_path, 0o755)

            cmd = mod._python_wrapper_command(sys.executable, install_prefix, "mlx-quant-init")
            env = mod._cli_surface_env_with_dependencies(
                dict(os.environ),
                prefix_purelib,
                install_prefix,
                [str(runtime_site)],
            )
            result = subprocess.run(
                cmd,
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

        output = (result.stdout or "") + (result.stderr or "")
        self.assertNotEqual(result.returncode, 0, output)
        self.assertNotIn("usage: fallback-package", output)

    def test_cli_help_probe_checks_wrapper_executability(self):
        mod = _load_module("release_smoke_contract_cli_wrapper_exec", self.script_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            install_prefix = tmp_path / "install"
            bin_dir = install_prefix / "bin"
            prefix_purelib = (
                install_prefix
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            pkg_dir = prefix_purelib / "mlx_quant_toolkit"
            bin_dir.mkdir(parents=True, exist_ok=True)
            pkg_dir.mkdir(parents=True, exist_ok=True)

            (pkg_dir / "__init__.py").write_text("")
            (pkg_dir / "cli.py").write_text(
                "import sys\n"
                "def init_run_cli():\n"
                "    if '--help' in sys.argv:\n"
                "        print('usage: prefix-package')\n"
                "        return 0\n"
                "    raise SystemExit(2)\n"
            )

            wrapper_path = bin_dir / "mlx-quant-init"
            wrapper_path.write_text(
                f"#!{sys.executable}\n"
                "from mlx_quant_toolkit.cli import init_run_cli\n"
                "raise SystemExit(init_run_cli())\n"
            )
            os.chmod(wrapper_path, 0o644)

            cmd = mod._python_wrapper_command(sys.executable, install_prefix, "mlx-quant-init")
            env = mod._cli_surface_env_with_dependencies(
                dict(os.environ),
                prefix_purelib,
                install_prefix,
                [str(prefix_purelib)],
            )
            with self.assertRaises(PermissionError):
                subprocess.run(
                    cmd,
                    cwd=tmp_path,
                    env=env,
                    capture_output=True,
                    text=True,
                )

    def test_release_smoke_runs_end_to_end_with_repo_venv(self):
        self.assertTrue(
            self.repo_python.exists(),
            f"Expected repo-local interpreter at {self.repo_python}",
        )
        result = subprocess.run(
            [
                str(self.repo_python),
                str(self.script_path),
                "--uv-cache-dir",
                "/tmp/uv-cache",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.returncode, 0, output)


if __name__ == "__main__":
    unittest.main()
