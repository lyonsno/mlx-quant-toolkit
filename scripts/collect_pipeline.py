from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def record_example(dst: List[str], value: str, limit: int = 25) -> None:
    if value in dst:
        return
    if len(dst) >= limit:
        return
    dst.append(value)


def _load_quant_public_columns() -> tuple[str, ...]:
    path = Path(__file__).with_name("collect_quant.py")
    module_name = "collect_quant"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == path.resolve():
            return tuple(existing.QUANT_SIM_COLUMNS)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load quant helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return tuple(module.QUANT_SIM_COLUMNS)


_PUBLIC_QUANT_SIM_COLUMNS = frozenset(_load_quant_public_columns())


def _merge_quant_row_fields(dst: Dict[str, Any], qr: pd.Series) -> None:
    for key, value in qr.items():
        if key == "expert_id_in_bank":
            continue
        if key not in _PUBLIC_QUANT_SIM_COLUMNS:
            continue
        dst[key] = value


def process_one_bank(
    *,
    bank_obj: Any,
    bank_erc: np.ndarray,
    layer_idx: Optional[int],
    cfg_stats: Dict[str, Any],
    cache_idx_dir: Path,
    matrix_rows: List[Dict[str, Any]],
    quant_rows: List[Dict[str, Any]],
    mlx_enabled: bool,
    schemes: List[Dict[str, Any]],
    mlx_device: str,
    per_expert_weight_stats: Callable[[np.ndarray, Dict[str, Any], Path], Dict[str, np.ndarray]],
    mlx_quant_sim: Callable[[np.ndarray, List[Dict[str, Any]], Dict[str, Any], str], Tuple[pd.DataFrame, List[str]]],
    warn_log: Optional[List[str]] = None,
) -> None:
    e_count, rows, cols = bank_erc.shape
    layer = layer_idx
    if layer is None:
        layer = bank_obj.layer_base
    # If still None, use -1 (unknown); global stats still remain valid.
    layer_val = int(layer) if layer is not None else -1
    block4 = (layer_val // 4) if layer_val >= 0 else None

    stats = per_expert_weight_stats(bank_erc, cfg_stats, cache_idx_dir)

    if bank_obj.is_shared_expert:
        expert_ids = np.full((e_count,), -1, dtype=np.int32)
        routed = np.zeros((e_count,), dtype=bool)
        shared = np.ones((e_count,), dtype=bool)
    elif bank_obj.expert_single_id is not None and e_count == 1:
        expert_ids = np.array([int(bank_obj.expert_single_id)], dtype=np.int32)
        routed = np.ones((e_count,), dtype=bool)
        shared = np.zeros((e_count,), dtype=bool)
    else:
        expert_ids = np.arange(e_count, dtype=np.int32)
        routed = np.ones((e_count,), dtype=bool)
        shared = np.zeros((e_count,), dtype=bool)

    for e in range(e_count):
        row = {
            "file": bank_obj.source_file,
            "source_tensor": bank_obj.source_tensor,
            "derived_tensor": bank_obj.derived_tensor,
            "layer": layer_val,
            "block4": block4,
            "proj": bank_obj.proj,
            "expert_id": int(expert_ids[e]),
            "is_routed_expert": bool(routed[e]),
            "is_shared_expert": bool(shared[e]),
            "rows": int(rows),
            "cols": int(cols),
            "dtype": str(bank_erc.dtype),
        }
        for k, v in stats.items():
            row[k] = float(v[e]) if np.ndim(v) == 1 else float(v)
        matrix_rows.append(row)

    if mlx_enabled and schemes:
        qdf, warns = mlx_quant_sim(bank_erc, schemes, cfg_stats, mlx_device)
        if warn_log is not None:
            warn_log.extend(warns)

        for _, qr in qdf.iterrows():
            e_in_bank = int(qr["expert_id_in_bank"])
            if bank_obj.is_shared_expert:
                exp_id = -1
            elif bank_obj.expert_single_id is not None and e_count == 1:
                exp_id = int(bank_obj.expert_single_id)
            else:
                exp_id = int(e_in_bank)

            quant_row = {
                "file": bank_obj.source_file,
                "source_tensor": bank_obj.source_tensor,
                "derived_tensor": bank_obj.derived_tensor,
                "layer": layer_val,
                "block4": block4,
                "proj": bank_obj.proj,
                "expert_id": exp_id,
                "is_shared_expert": bool(bank_obj.is_shared_expert),
                "rows": int(rows),
                "cols": int(cols),
            }
            _merge_quant_row_fields(quant_row, qr)
            quant_rows.append(quant_row)


def process_extracted_banks(
    *,
    extracted: Iterable[Any],
    cfg_stats: Dict[str, Any],
    cache_idx_dir: Path,
    matrix_rows: List[Dict[str, Any]],
    quant_rows: List[Dict[str, Any]],
    mlx_enabled: bool,
    schemes: List[Dict[str, Any]],
    mlx_device: str,
    per_expert_weight_stats: Callable[[np.ndarray, Dict[str, Any], Path], Dict[str, np.ndarray]],
    mlx_quant_sim: Callable[[np.ndarray, List[Dict[str, Any]], Dict[str, Any], str], Tuple[pd.DataFrame, List[str]]],
    warn_log: List[str],
    process_one_bank_fn: Callable[..., None] = process_one_bank,
) -> None:
    for bank_obj in extracted:
        bank = bank_obj.bank

        # Canonical shapes allowed: (L,E,R,C), (E,R,C), (R,C).
        if bank.ndim == 2:
            bank = bank[None, ...]  # -> (1,R,C)
        if bank.ndim == 3:
            process_one_bank_fn(
                bank_obj=bank_obj,
                bank_erc=bank,
                layer_idx=bank_obj.layer_base,
                cfg_stats=cfg_stats,
                cache_idx_dir=cache_idx_dir,
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=mlx_enabled,
                schemes=schemes,
                mlx_device=mlx_device,
                per_expert_weight_stats=per_expert_weight_stats,
                mlx_quant_sim=mlx_quant_sim,
                warn_log=warn_log,
            )
        elif bank.ndim == 4:
            layer_count = bank.shape[0]
            for li in range(layer_count):
                layer_idx = (bank_obj.layer_base + li) if bank_obj.layer_base is not None else li
                process_one_bank_fn(
                    bank_obj=bank_obj,
                    bank_erc=bank[li],
                    layer_idx=layer_idx,
                    cfg_stats=cfg_stats,
                    cache_idx_dir=cache_idx_dir,
                    matrix_rows=matrix_rows,
                    quant_rows=quant_rows,
                    mlx_enabled=mlx_enabled,
                    schemes=schemes,
                    mlx_device=mlx_device,
                    per_expert_weight_stats=per_expert_weight_stats,
                    mlx_quant_sim=mlx_quant_sim,
                    warn_log=warn_log,
                )
        else:
            warn_log.append(
                f"[extract] unsupported canonical ndim={bank.ndim} for {bank_obj.derived_tensor}"
            )
