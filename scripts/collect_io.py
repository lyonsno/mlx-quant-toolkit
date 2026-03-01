from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from safetensors import safe_open


# bfloat16 handling for safetensors -> numpy paths

def _ensure_numpy_bfloat16() -> bool:
    """
    Return True if np.dtype("bfloat16") works after best-effort registration.
    Needed because safetensors' numpy backend uses the dtype string "bfloat16".
    """
    try:
        np.dtype("bfloat16")
        return True
    except Exception:
        pass

    try:
        import ml_dtypes  # type: ignore

        # Register dtype aliases so np.dtype("bfloat16") resolves on builds that allow it.
        try:
            np.sctypeDict["bfloat16"] = ml_dtypes.bfloat16
            np.sctypeDict["bf16"] = ml_dtypes.bfloat16
        except Exception:
            pass

        np.dtype("bfloat16")
        return True
    except Exception:
        return False


_BF16_READY = _ensure_numpy_bfloat16()


def _safe_read_json_dict(path: Path) -> Dict[str, Any]:
    """
    Best-effort JSON reader that returns {} on any failure.
    Intended for optional run artifacts like manifest.json and metadata logs.
    """
    try:
        raw = path.read_text()
    except Exception:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _write_df(df: pd.DataFrame, path: Path, fmt: str, compression: str | None) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False, compression=compression)
            return {
                "path": path,
                "format": "parquet",
                "fallback": False,
                "error": "",
            }
        except Exception as e:
            print(f"[warn] parquet write failed ({e}); falling back to CSV for {path}")
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return {
                "path": csv_path,
                "format": "csv",
                "fallback": True,
                "error": f"{type(e).__name__}: {e}",
            }
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return {
        "path": csv_path,
        "format": "csv",
        "fallback": False,
        "error": "",
    }


def _iter_weight_files(model_path: Path, exts: List[str]) -> Iterable[Path]:
    if model_path.is_file():
        if model_path.suffix in exts:
            yield model_path
        return
    for root, _, files in os.walk(model_path):
        for fn in files:
            p = Path(root) / fn
            if p.suffix in exts:
                yield p


def _iter_tensors_from_file(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="numpy") as f:
            for name in f.keys():
                try:
                    yield name, f.get_tensor(name)
                except TypeError as e:
                    if "bfloat16" in str(e).lower():
                        ok = _ensure_numpy_bfloat16()
                        if ok:
                            yield name, f.get_tensor(name)
                            continue
                        raise RuntimeError(
                            "This checkpoint contains bfloat16 tensors, but NumPy cannot decode 'bfloat16'. "
                            "Install ml-dtypes (uv add ml-dtypes) or load via torch backend instead."
                        ) from e
                    raise
    elif path.suffix == ".npz":
        # Ensure np.load context is closed quickly to avoid fd pressure while scanning many shards.
        with np.load(str(path), allow_pickle=False) as data:
            for name in data.files:
                yield name, data[name]
