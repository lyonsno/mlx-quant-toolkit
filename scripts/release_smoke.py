#!/usr/bin/env python3
"""
Run a clean build/install smoke for release packaging.

Checks:
- clean copy build with uv
- wheel install into temp target
- package-local import surface works
- top-level "scripts" namespace is not the install surface
- CLI wrappers resolve and print --help
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


_INSTALLED_SURFACE_EXCLUDED_ENV_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
)


def _cmd_display(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def _run_command(cmd: Sequence[str], cwd: Path, env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        details = f"\n{output}" if output else ""
        raise RuntimeError(f"command failed ({result.returncode}): {_cmd_display(cmd)}{details}")
    return result


def _build_check_snippets() -> List[tuple[str, str]]:
    return [
        (
            "import-scripts-module",
            "import mlx_quant_toolkit.scripts.init_run as m; print(m.__name__)",
        ),
        (
            "scripts-namespace-surface",
            (
                "import importlib.util; "
                "spec = importlib.util.find_spec(\"scripts\"); "
                "print(\"NONE\" if spec is None else repr(spec)); "
                "raise SystemExit(0 if spec is None else 1)"
            ),
        ),
    ]


def _python_with_prefix_site_packages(python_executable: str, install_prefix: Path, snippet: str) -> List[str]:
    prefix_text = str(install_prefix)
    bootstrap = (
        "import sys, sysconfig; "
        f"prefix={prefix_text!r}; "
        "sys.path.insert(0, sysconfig.get_path('purelib', vars={'base': prefix, 'platbase': prefix})); "
    )
    return [python_executable, "-I", "-S", "-c", bootstrap + snippet]


def _installed_bin_path(install_prefix: Path, executable_name: str) -> Path:
    return install_prefix / "bin" / executable_name


def _python_wrapper_command(
    python_executable: str,
    install_prefix: Path,
    executable_name: str,
) -> List[str]:
    return [str(_installed_bin_path(install_prefix, executable_name)), "--help"]


def _workspace_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "repo", tmp_path / "out", tmp_path / "install"


def _installed_surface_env(base_env: Dict[str, str]) -> Dict[str, str]:
    return {
        key: value
        for key, value in base_env.items()
        if key not in _INSTALLED_SURFACE_EXCLUDED_ENV_VARS
    }


def _cli_surface_env(base_env: Dict[str, str], purelib: Path) -> Dict[str, str]:
    env = _installed_surface_env(base_env)
    env["PYTHONPATH"] = str(purelib)
    return env


def _stage_project_snapshot(project_root: Path, staged_repo: Path) -> None:
    git_ls_files = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(project_root),
        capture_output=True,
        check=False,
    )
    if git_ls_files.returncode != 0:
        shutil.copytree(
            project_root,
            staged_repo,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "build",
                "dist",
                "__pycache__",
                "*.egg-info",
            ),
        )
        return

    staged_repo.mkdir(parents=True, exist_ok=True)
    for raw_path in git_ls_files.stdout.split(b"\0"):
        if not raw_path:
            continue
        rel_path = Path(raw_path.decode())
        src = project_root / rel_path
        if not src.exists() or src.is_dir():
            continue
        dst = staged_repo / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _python_runtime_site_packages(
    python_executable: str,
    env: Dict[str, str],
    cwd: Path,
) -> List[str]:
    result = _run_command(
        [
            python_executable,
            "-I",
            "-c",
            (
                "import sysconfig; "
                "print(sysconfig.get_path('purelib')); "
                "print(sysconfig.get_path('platlib'))"
            ),
        ],
        cwd=cwd,
        env=env,
    )
    seen = set()
    ordered = []
    for line in (result.stdout or "").splitlines():
        if line and line not in seen:
            seen.add(line)
            ordered.append(line)
    return ordered


def _cli_pythonpath(prefix_purelib: Path, runtime_site_packages: Sequence[str]) -> str:
    guard_dir = _ensure_cli_guard_dir(prefix_purelib)
    entries = [str(guard_dir), str(prefix_purelib)] + [
        path for path in runtime_site_packages if path != str(prefix_purelib)
    ]
    return os.pathsep.join(entries)


def _ensure_cli_guard_dir(prefix_purelib: Path) -> Path:
    install_prefix = prefix_purelib.parents[2]
    guard_dir = install_prefix / ".release_smoke_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    guard_file = guard_dir / "sitecustomize.py"
    guard_file.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "prefix = Path(os.environ['MLX_RELEASE_SMOKE_PREFIX_PURELIB']).resolve()\n"
        "pkg_dir = prefix / 'mlx_quant_toolkit'\n"
        "pkg_file = prefix / 'mlx_quant_toolkit.py'\n"
        "if not pkg_dir.exists() and not pkg_file.exists():\n"
        "    sys.stderr.write(\n"
        "        f'mlx_quant_toolkit is not installed under release-smoke prefix: {prefix}\\n'\n"
        "    )\n"
        "    raise SystemExit(1)\n"
    )
    return guard_dir


def _cli_surface_env_with_dependencies(
    base_env: Dict[str, str],
    prefix_purelib: Path,
    runtime_site_packages: Sequence[str],
) -> Dict[str, str]:
    env = _installed_surface_env(base_env)
    env["PYTHONPATH"] = _cli_pythonpath(prefix_purelib, runtime_site_packages)
    env["PYTHONNOUSERSITE"] = "1"
    env["MLX_RELEASE_SMOKE_PREFIX_PURELIB"] = str(prefix_purelib)
    return env


def _prefix_purelib_path(python_executable: str, install_prefix: Path, env: Dict[str, str], cwd: Path) -> Path:
    result = _run_command(
        [
            python_executable,
            "-I",
            "-S",
            "-c",
            (
                "import sysconfig; "
                f"prefix={str(install_prefix)!r}; "
                "print(sysconfig.get_path('purelib', vars={'base': prefix, 'platbase': prefix}))"
            ),
        ],
        cwd=cwd,
        env=env,
    )
    return Path((result.stdout or "").strip())


def _iter_release_commands(
    python_executable: str,
    uv_executable: str,
    staged_repo: Path,
    out_dir: Path,
    install_prefix: Path,
) -> Iterable[tuple[str, List[str], Path]]:
    yield (
        "build",
        [
            uv_executable,
            "build",
            "--python",
            python_executable,
            "--no-build-isolation",
            "--out-dir",
            str(out_dir),
        ],
        staged_repo,
    )

    wheel_glob = str(out_dir / "*.whl")
    yield (
        "install-wheel",
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            python_executable,
            "--no-deps",
            "--prefix",
            str(install_prefix),
            wheel_glob,
        ],
        staged_repo,
    )

    for label, snippet in _build_check_snippets():
        yield (
            label,
            _python_with_prefix_site_packages(python_executable, install_prefix, snippet),
            out_dir.parent,
        )
    for label, executable_name in (
        ("cli-init-help", "mlx-quant-init"),
        ("cli-collect-help", "mlx-quant-collect"),
        ("cli-build-tables-help", "mlx-quant-build-tables"),
        ("cli-build-plots-help", "mlx-quant-build-plots"),
    ):
        yield (
            label,
            _python_wrapper_command(python_executable, install_prefix, executable_name),
            out_dir.parent,
        )


def run_release_smoke(
    project_root: Path,
    *,
    dry_run: bool = False,
    uv_cache_dir: str | None = None,
    python_executable: str | None = None,
    uv_executable: str = "uv",
) -> None:
    project_root = project_root.resolve()
    python_executable = python_executable or os.environ.get("PYTHON", "") or sys.executable
    env = os.environ.copy()
    if uv_cache_dir:
        env["UV_CACHE_DIR"] = uv_cache_dir

    if dry_run:
        staged_repo, out_dir, install_prefix = _workspace_paths(Path("/tmp/mlx-release-smoke-dry-run"))
        steps = list(_iter_release_commands(python_executable, uv_executable, staged_repo, out_dir, install_prefix))
        for label, cmd, cwd in steps:
            print(f"[release-smoke][dry-run] {label} cwd={cwd}")
            print(f"[release-smoke][dry-run]   {_cmd_display(cmd)}")
        return

    with tempfile.TemporaryDirectory(prefix="mlx-release-smoke-") as tmp:
        tmp_path = Path(tmp)
        staged_repo, out_dir, install_prefix = _workspace_paths(tmp_path)

        _stage_project_snapshot(project_root, staged_repo)
        out_dir.mkdir(parents=True, exist_ok=True)
        install_prefix.mkdir(parents=True, exist_ok=True)

        steps = list(_iter_release_commands(python_executable, uv_executable, staged_repo, out_dir, install_prefix))
        installed_surface_env = _installed_surface_env(env)
        purelib: Path | None = None
        cli_surface_env: Dict[str, str] | None = None

        for label, cmd, cwd in steps:
            if label in {"build", "install-wheel"}:
                run_env = env
            elif label.startswith("cli-"):
                if purelib is None:
                    purelib = _prefix_purelib_path(
                        python_executable,
                        install_prefix,
                        installed_surface_env,
                        out_dir.parent,
                    )
                    cli_surface_env = _cli_surface_env_with_dependencies(
                        env,
                        purelib,
                        _python_runtime_site_packages(python_executable, installed_surface_env, out_dir.parent),
                    )
                assert cli_surface_env is not None
                run_env = cli_surface_env
            else:
                run_env = installed_surface_env
            if label == "install-wheel":
                wheel_matches = sorted(out_dir.glob("*.whl"))
                if len(wheel_matches) != 1:
                    raise RuntimeError(f"expected exactly one wheel in {out_dir}, found {len(wheel_matches)}")
                cmd = [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    python_executable,
                    "--no-deps",
                    "--prefix",
                    str(install_prefix),
                    str(wheel_matches[0]),
                ]
            print(f"[release-smoke] {label}: {_cmd_display(cmd)}")
            result = _run_command(cmd, cwd=cwd, env=run_env)
            text = (result.stdout or "") + (result.stderr or "")
            if text.strip():
                print(text.strip())

        if purelib is None:
            purelib = _prefix_purelib_path(
                python_executable,
                install_prefix,
                installed_surface_env,
                out_dir.parent,
            )
        if (purelib / "scripts").exists():
            raise RuntimeError("installed prefix unexpectedly contains top-level scripts package")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run clean build/install/import release smoke checks for packaging."
    )
    ap.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to stage for release smoke",
    )
    ap.add_argument(
        "--python",
        default=None,
        help="Python executable to use for wheel install and checks (default: PYTHON env or current interpreter)",
    )
    ap.add_argument(
        "--uv",
        default="uv",
        help="uv executable to use for build step",
    )
    ap.add_argument(
        "--uv-cache-dir",
        default=None,
        help="Optional UV_CACHE_DIR value for build stability in restricted environments",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them",
    )
    args = ap.parse_args()

    run_release_smoke(
        Path(args.project_root),
        dry_run=bool(args.dry_run),
        uv_cache_dir=args.uv_cache_dir,
        python_executable=args.python,
        uv_executable=args.uv,
    )


if __name__ == "__main__":
    main()
