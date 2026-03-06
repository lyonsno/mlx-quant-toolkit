from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_TABLE_ARTIFACTS_MODULE = None


def _local_helper_module_key(module_name: str) -> str:
    return f"{__name__}.__local__.{module_name}"


def _load_local_helper_module(module_name: str):
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    local_module_key = _local_helper_module_key(module_name)
    for key in (local_module_key, module_name):
        existing = sys.modules.get(key)
        if existing is None:
            continue
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None:
            try:
                if Path(existing_file).resolve() == module_path:
                    return existing
            except Exception:
                pass

    spec = importlib.util.spec_from_file_location(local_module_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    had_prior_entry = local_module_key in sys.modules
    prior_entry = sys.modules.get(local_module_key)
    sys.modules[local_module_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        current = sys.modules.get(local_module_key)
        if current is module:
            if had_prior_entry:
                sys.modules[local_module_key] = prior_entry
            else:
                sys.modules.pop(local_module_key, None)
        raise
    return module


def _get_table_artifacts_module():
    global _TABLE_ARTIFACTS_MODULE
    if _TABLE_ARTIFACTS_MODULE is None:
        _TABLE_ARTIFACTS_MODULE = _load_local_helper_module("table_artifacts")
    return _TABLE_ARTIFACTS_MODULE


def _coerce_axis_series_to_nullable_int(series: pd.Series, column_name: str) -> pd.Series:
    raw = series.copy()
    # Treat None/NaN and blank strings as missing axis values.
    missing_mask = raw.isna()
    if pd.api.types.is_string_dtype(raw.dtype) or pd.api.types.is_object_dtype(raw.dtype):
        as_string = raw.astype("string")
        missing_mask = missing_mask | as_string.str.strip().eq("")

    numeric = pd.to_numeric(raw.where(~missing_mask, other=pd.NA), errors="coerce")

    invalid_mask = (~missing_mask) & numeric.isna()
    if invalid_mask.any():
        bad_row = invalid_mask[invalid_mask].index[0]
        bad_value = raw.loc[bad_row]
        raise ValueError(
            f"Invalid integer token in axis column '{column_name}' at row {bad_row}: {bad_value!r}"
        )

    non_missing_mask = ~missing_mask
    for row_idx, value in numeric[non_missing_mask].items():
        value_f = float(value)
        if not math.isfinite(value_f) or not value_f.is_integer():
            raise ValueError(
                f"Invalid integer token in axis column '{column_name}' at row {row_idx}: {raw.loc[row_idx]!r}"
            )

    return numeric.astype("Int64")


def normalize_plot_axis_columns(
    df: pd.DataFrame,
    axis_columns: Iterable[str] = ("layer", "block4"),
) -> pd.DataFrame:
    normalized = df.copy()
    for column_name in axis_columns:
        if column_name not in normalized.columns:
            continue
        normalized[column_name] = _coerce_axis_series_to_nullable_int(
            normalized[column_name],
            column_name,
        )
    return normalized


def _read_table_artifact(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table artifact format for plotting: path={path}, format={fmt!r}")


def load_plot_tables(
    run_dir: str | Path,
    artifact_keys: Iterable[str] | None = None,
    axis_columns: Iterable[str] = ("layer", "block4"),
) -> dict[str, pd.DataFrame]:
    run_dir_path = Path(run_dir).expanduser().resolve()
    table_artifacts = _get_table_artifacts_module()

    if artifact_keys is None:
        artifact_keys = table_artifacts.DEFAULT_TABLE_ARTIFACT_KEYS

    discovered = table_artifacts.discover_table_artifacts(
        run_dir_path,
        artifact_keys=artifact_keys,
    )

    loaded: dict[str, pd.DataFrame] = {}
    for artifact_key, meta in discovered.items():
        raw_path = str(meta.get("path", "")).strip()
        if not raw_path:
            continue

        path_obj = Path(raw_path)
        abs_path = path_obj if path_obj.is_absolute() else (run_dir_path / path_obj)
        fmt = str(meta.get("format", "")).strip().lower()

        df = _read_table_artifact(abs_path, fmt)
        loaded[artifact_key] = normalize_plot_axis_columns(df, axis_columns=axis_columns)

    return loaded
