from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


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
