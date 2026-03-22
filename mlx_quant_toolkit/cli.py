"""Thin console-script shims for the existing pipeline entry points."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType


def _get_script_module(module_name: str) -> ModuleType:
    package_module_name = f"mlx_quant_toolkit.scripts.{module_name}"
    try:
        return import_module(package_module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("mlx_quant_toolkit.scripts", package_module_name):
            raise
    return _load_local_script_module(module_name)


def _load_local_script_module(module_name: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / f"{module_name}.py"
    if not module_path.exists():
        raise ModuleNotFoundError(
            f"Unable to locate packaged or local script module for {module_name}"
        )

    local_module_key = f"{__name__}.__local__.{module_name}"
    existing = sys.modules.get(local_module_key)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        try:
            if existing_file is not None and Path(existing_file).resolve() == module_path:
                return existing
        except Exception:
            pass

    spec = importlib.util.spec_from_file_location(local_module_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load local script module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[local_module_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        current = sys.modules.get(local_module_key)
        if current is module:
            sys.modules.pop(local_module_key, None)
        raise
    return module


def init_run_cli() -> None:
    _get_script_module("init_run").main()


def collect_data_cli() -> None:
    _get_script_module("collect_data").main()


def build_tables_cli() -> None:
    _get_script_module("build_tables").main()


def build_plots_cli() -> None:
    _get_script_module("build_plots").main()


__all__ = [
    "init_run_cli",
    "collect_data_cli",
    "build_tables_cli",
    "build_plots_cli",
]
