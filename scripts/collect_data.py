#!/usr/bin/env python3
"""
Collect raw MoE expert weight statistics + MLX-faithful quantization sims.

Run with only:
  python collect_data.py --run-dir /path/to/run

Override model path if you want:
  python collect_data.py --run-dir ... --model-path /path/to/model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from safetensors import safe_open

try:
    import mlx.core as mx
except Exception:
    mx = None


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


# Attempt registration up front; still retry on-demand below.
_BF16_READY = _ensure_numpy_bfloat16()


def _is_floatlike_dtype(dtype: np.dtype) -> bool:
    try:
        if np.issubdtype(dtype, np.floating):
            return True
    except Exception:
        pass
    return "bfloat16" in str(dtype).lower()


# ------------------------- IO helpers ----------------------------------------

def _write_df(df: pd.DataFrame, path: Path, fmt: str, compression: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False, compression=compression)
            return
        except Exception as e:
            print(f"[warn] parquet write failed ({e}); falling back to CSV for {path}")
            df.to_csv(path.with_suffix(".csv"), index=False)
            return
    df.to_csv(path.with_suffix(".csv"), index=False)


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "analysis_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path} (run init_run.py first)")
    return json.loads(cfg_path.read_text())


def _iter_weight_files(model_path: Path, exts: List[str]) -> Iterable[Path]:
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
        data = np.load(str(path), allow_pickle=False)
        for name in data.files:
            yield name, data[name]


# ------------------------- parsing -------------------------------------------

@dataclass
class Rule:
    name: str
    enabled: bool
    regex: Any
    ndim: Optional[int]
    layout: Dict[str, Optional[int]]
    proj_group: Optional[int]
    expert_group: Optional[int]
    packed_split: Optional[Dict[str, Any]]


class PackedSplitError(RuntimeError):
    pass


def _compile_rules(cfg: Dict[str, Any]) -> List[Rule]:
    out: List[Rule] = []
    for r in cfg.get("extract_rules", []):
        out.append(
            Rule(
                name=r["name"],
                enabled=bool(r.get("enabled", True)),
                regex=re.compile(r["match"]),
                ndim=r.get("ndim", None),
                layout=r.get("layout", {}),
                proj_group=r.get("proj_group", None),
                expert_group=r.get("expert_group", None),
                packed_split=r.get("packed_split", None),
            )
        )
    return out


def _parse_int_from_regex(regex: re.Pattern, text: str) -> Optional[int]:
    m = regex.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_shared_expert(name: str, keywords: List[str]) -> bool:
    n = name.lower()
    return all(k.lower() in n for k in keywords)


def _infer_proj(name: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    n = name.lower()
    for canonical, aliases in alias_map.items():
        for a in aliases:
            if a.lower() in n:
                return canonical
    return None


def _split_along_axis(x: np.ndarray, axis: int, splits: List[int]) -> List[np.ndarray]:
    if not isinstance(axis, Integral):
        raise ValueError(f"axis must be an int; got {type(axis).__name__}")
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"axis {axis} is out of bounds for ndim={x.ndim}")

    splits_list = list(splits)
    if not splits_list:
        raise ValueError("splits must be non-empty")

    total = 0
    for s in splits_list:
        if not isinstance(s, Integral):
            raise ValueError(f"split sizes must be integers; got {type(s).__name__}")
        if s <= 0:
            raise ValueError(f"split sizes must be positive; got {s}")
        total += int(s)

    axis_len = x.shape[axis]
    if total != axis_len:
        raise ValueError(f"split sizes sum to {total}, expected {axis_len} along axis {axis}")

    outs = []
    start = 0
    for s in splits_list:
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(start, start + int(s))
        outs.append(x[tuple(sl)])
        start += int(s)
    return outs


def _canonicalize_layout(arr: np.ndarray, layout: Dict[str, Optional[int]]) -> np.ndarray:
    """
    Reorder axes into one of:
      (L, E, R, C) if layer_axis and expert_axis are provided
      (E, R, C) if expert_axis is provided
      (R, C) otherwise

    NOTE: this does NOT transpose rows/cols unless the user sets rows_axis/cols_axis that way.
    """
    layer_axis = layout.get("layer_axis", None)
    expert_axis = layout.get("expert_axis", None)
    rows_axis = layout.get("rows_axis", None)
    cols_axis = layout.get("cols_axis", None)

    if rows_axis is None or cols_axis is None:
        raise ValueError("layout must include rows_axis and cols_axis")

    axes = []
    if layer_axis is not None:
        axes.append(layer_axis)
    if expert_axis is not None:
        axes.append(expert_axis)
    axes.extend([rows_axis, cols_axis])

    if len(axes) != arr.ndim:
        # If the tensor has extra batch axes, we don't guess: require an explicit adapter rule.
        raise ValueError(f"layout axes {axes} do not cover ndim={arr.ndim}")

    return np.transpose(arr, axes)


@dataclass
class ExtractedBank:
    source_file: str
    source_tensor: str
    derived_tensor: str
    proj: str
    is_shared_expert: bool
    layer_base: Optional[int]     # if the name encodes a layer
    expert_single_id: Optional[int]  # if the name encodes a single expert id
    bank: np.ndarray              # (L,E,R,C) or (E,R,C) or (R,C)


def _apply_rules(
    name: str,
    arr: np.ndarray,
    fpath: Path,
    rules: List[Rule],
    layer_re: re.Pattern,
    expert_re: re.Pattern,
    alias_map: Dict[str, List[str]],
    shared_keywords: List[str],
) -> Optional[List[ExtractedBank]]:
    for r in rules:
        if not r.enabled:
            continue
        m = r.regex.match(name)
        if not m:
            continue
        if r.ndim is not None and arr.ndim != r.ndim:
            continue

        is_shared = _is_shared_expert(name, shared_keywords)

        layer_base = _parse_int_from_regex(layer_re, name)
        expert_single_id = None
        if r.expert_group is not None:
            try:
                expert_single_id = int(m.group(r.expert_group))
            except Exception:
                expert_single_id = None

        # Determine proj (direct vs packed)
        if r.packed_split is None:
            if r.proj_group is not None:
                raw = m.group(r.proj_group)
                proj = _infer_proj(raw, {k: [k] for k in alias_map.keys()}) or raw
            else:
                proj = _infer_proj(name, alias_map)
            if proj is None:
                return None

            canon = _canonicalize_layout(arr, r.layout)
            derived = f"{name}::{proj}"
            return [ExtractedBank(
                source_file=str(fpath),
                source_tensor=name,
                derived_tensor=derived,
                proj=proj,
                is_shared_expert=is_shared,
                layer_base=layer_base,
                expert_single_id=expert_single_id,
                bank=canon
            )]

        # packed split
        packed = r.packed_split
        proj_list = packed["projs"]
        splits = packed["splits"]
        axis_kind = packed["axis"]  # "rows" or "cols"
        canon = _canonicalize_layout(arr, r.layout)

        # find split axis index in canonical
        if canon.ndim == 4:
            rows_i, cols_i = 2, 3
        elif canon.ndim == 3:
            rows_i, cols_i = 1, 2
        elif canon.ndim == 2:
            rows_i, cols_i = 0, 1
        else:
            raise ValueError(f"Unsupported canonical ndim={canon.ndim}")

        split_axis = rows_i if axis_kind == "rows" else cols_i
        try:
            parts = _split_along_axis(canon, split_axis, splits)
        except Exception as e:
            msg = f"packed_split failed for rule={r.name} tensor={name}: {e}"
            raise PackedSplitError(msg) from e
        if len(parts) != len(proj_list):
            msg = f"packed_split projs and splits length mismatch for rule={r.name} tensor={name}"
            raise PackedSplitError(msg)

        banks: List[ExtractedBank] = []
        for proj, part in zip(proj_list, parts):
            derived = f"{name}::split[{axis_kind}]::{proj}"
            banks.append(ExtractedBank(
                source_file=str(fpath),
                source_tensor=name,
                derived_tensor=derived,
                proj=proj,
                is_shared_expert=is_shared,
                layer_base=layer_base,
                expert_single_id=expert_single_id,
                bank=part
            ))
        return banks

    return None


def _fallback_extract(
    name: str,
    arr: np.ndarray,
    fpath: Path,
    layer_re: re.Pattern,
    expert_re: re.Pattern,
    alias_map: Dict[str, List[str]],
    shared_keywords: List[str],
) -> Optional[List[ExtractedBank]]:
    proj = _infer_proj(name, alias_map)
    if proj is None:
        return None

    is_shared = _is_shared_expert(name, shared_keywords)
    layer_base = _parse_int_from_regex(layer_re, name)
    expert_single_id = _parse_int_from_regex(expert_re, name)

    # Heuristic layouts:
    if arr.ndim == 3:
        # (E,R,C) assumed
        derived = f"{name}::{proj}"
        return [ExtractedBank(str(fpath), name, derived, proj, is_shared, layer_base, None, arr)]
    if arr.ndim == 2:
        derived = f"{name}::{proj}"
        return [ExtractedBank(str(fpath), name, derived, proj, is_shared, layer_base, expert_single_id, arr)]
    return None


# ------------------------- stats ---------------------------------------------

def _get_sample_indices(cache_dir: Path, total: int, k: int, seed: int) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"idx_N{total}_k{k}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    rng = np.random.default_rng(seed=seed)
    if k >= total:
        idx = np.arange(total, dtype=np.int64)
    else:
        idx = rng.choice(total, size=k, replace=False).astype(np.int64)
    np.save(path, idx)
    return idx


def _per_expert_weight_stats(bank: np.ndarray, cfg_stats: Dict[str, Any], cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    bank: (E,R,C) float-ish
    returns dict of metric -> (E,)
    """
    eps = float(cfg_stats["eps"])
    percentiles = cfg_stats.get("percentiles_abs", [50.0, 99.0, 99.9])
    sample_k = int(cfg_stats.get("sample_per_matrix", 8192))
    seed = int(cfg_stats.get("sample_seed", 1337))
    group_p = float(cfg_stats.get("group_outlier_percentile", 95.0))
    group_sizes = cfg_stats.get("group_sizes_lastdim", [32, 64])

    # work in float32 for stability/speed
    w = bank.astype(np.float32, copy=False)
    E, R, C = w.shape
    abs_w = np.abs(w)

    mean = w.mean(axis=(1, 2))
    std = w.std(axis=(1, 2))
    mean_abs = abs_w.mean(axis=(1, 2))
    rms = np.sqrt((w * w).mean(axis=(1, 2)))
    max_abs = abs_w.max(axis=(1, 2))

    # sampled percentiles per expert, same indices for all experts
    flat = abs_w.reshape(E, -1)
    idx = _get_sample_indices(cache_dir, total=R * C, k=sample_k, seed=seed)
    samp = flat[:, idx]  # (E,k)
    pvals = {}
    for q in percentiles:
        pvals[q] = np.percentile(samp, q, axis=1)

    p50 = pvals.get(50.0, np.percentile(samp, 50.0, axis=1))
    p99 = pvals.get(99.0, np.percentile(samp, 99.0, axis=1))
    p999 = pvals.get(99.9, np.percentile(samp, 99.9, axis=1))

    outlier_max_over_mean = max_abs / (mean_abs + eps)
    outlier_p99_over_median = p99 / (p50 + eps)
    outlier_p999_over_median = p999 / (p50 + eps)

    stats: Dict[str, np.ndarray] = {
        "mean": mean,
        "std": std,
        "mean_abs": mean_abs,
        "rms": rms,
        "max_abs": max_abs,
        "p50_abs": p50,
        "p99_abs": p99,
        "p999_abs": p999,
        "outlier_max_over_mean": outlier_max_over_mean,
        "outlier_p99_over_median": outlier_p99_over_median,
        "outlier_p999_over_median": outlier_p999_over_median,
    }

    # groupwise outliers along LAST DIM (matches MLX quant grouping semantics)
    for G in group_sizes:
        G = int(G)
        if G <= 0 or (C % G) != 0:
            stats[f"g{G}_p{int(group_p)}_outlier"] = np.full((E,), np.nan, dtype=np.float32)
            stats[f"g{G}_max_outlier"] = np.full((E,), np.nan, dtype=np.float32)
            continue

        resh = abs_w.reshape(E, R, C // G, G)
        gmax = resh.max(axis=-1)
        gmean = resh.mean(axis=-1)
        ratio = gmax / (gmean + eps)      # (E,R,C//G)
        ratio_flat = ratio.reshape(E, -1)
        stats[f"g{G}_p{int(group_p)}_outlier"] = np.percentile(ratio_flat, group_p, axis=1)
        stats[f"g{G}_max_outlier"] = ratio_flat.max(axis=1)

    return stats


QUANT_SIM_COLUMNS = [
    "scheme",
    "mode",
    "bits",
    "group_size",
    "expert_id_in_bank",
    "w_rel_fro",
    "w_rel_max",
    "scale_mean",
    "scale_max",
    "bias_mean",
    "bias_max",
    "error",
]


def _mlx_quant_sim(
    bank: np.ndarray,
    schemes: List[Dict[str, Any]],
    cfg_stats: Dict[str, Any],
    device: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    bank: (E,R,C) numpy
    returns: quant_sim dataframe rows (per expert per scheme) AND list of warnings strings
    """
    eps = float(cfg_stats["eps"])
    warns: List[str] = []

    if mx is None:
        msg = "mlx is not importable; skipping quantization simulation"
        warnings.warn(msg)
        return pd.DataFrame(columns=QUANT_SIM_COLUMNS), [msg]

    if device == "cpu":
        try:
            mx.set_default_device(mx.cpu)
        except Exception:
            pass
    elif device == "gpu":
        try:
            mx.set_default_device(mx.gpu)
        except Exception:
            pass

    # Use float16 to reduce memory; errors are relative so OK for ranking.
    w = bank.astype(np.float16, copy=False)
    w_mx = mx.array(w)

    rows = []
    for s in schemes:
        if not s.get("enabled", True):
            continue
        name = s["name"]
        mode = s["mode"]
        bits = int(s.get("bits", 4))
        group_size = int(s.get("group_size", 32))

        try:
            q = mx.quantize(w_mx, group_size=group_size, bits=bits, mode=mode)  #
            if mode == "affine":
                wq, scales, biases = q
            else:
                wq, scales = q
                biases = None

            w_hat = mx.dequantize(
                wq, scales, biases,
                group_size=group_size, bits=bits, mode=mode,
                dtype=w_mx.dtype
            )  #

            diff = w_hat - w_mx

            num = mx.sqrt(mx.sum(diff * diff, axis=(1, 2)))
            den = mx.sqrt(mx.sum(w_mx * w_mx, axis=(1, 2))) + eps
            rel_fro = num / den

            rel_max = mx.max(mx.abs(diff), axis=(1, 2)) / (mx.max(mx.abs(w_mx), axis=(1, 2)) + eps)

            # scale/bias stats (useful for diagnosing "why is this matrix hard?")
            s_mean = mx.mean(scales, axis=tuple(range(scales.ndim))[0:scales.ndim-0])  # placeholder
            # Instead of guessing axis, do per-expert reduction explicitly:
            # scales shape is usually (E,R,C//G) for 3D input, so reduce axes (1,2)
            if scales.ndim >= 3:
                scales_mean = mx.mean(scales, axis=(1, 2))
                scales_max = mx.max(scales, axis=(1, 2))
            else:
                # fallback
                scales_mean = mx.mean(scales, axis=0)
                scales_max = mx.max(scales, axis=0)

            if biases is not None:
                if biases.ndim >= 3:
                    biases_mean = mx.mean(biases, axis=(1, 2))
                    biases_max = mx.max(biases, axis=(1, 2))
                else:
                    biases_mean = mx.mean(biases, axis=0)
                    biases_max = mx.max(biases, axis=0)
            else:
                biases_mean = None
                biases_max = None

            mx.eval(rel_fro, rel_max, scales_mean, scales_max)
            rel_fro_np = np.array(rel_fro).astype(np.float32)
            rel_max_np = np.array(rel_max).astype(np.float32)
            scales_mean_np = np.array(scales_mean).astype(np.float32)
            scales_max_np = np.array(scales_max).astype(np.float32)
            if biases_mean is not None:
                mx.eval(biases_mean, biases_max)
                biases_mean_np = np.array(biases_mean).astype(np.float32)
                biases_max_np = np.array(biases_max).astype(np.float32)
            else:
                biases_mean_np = None
                biases_max_np = None

            E = rel_fro_np.shape[0]
            for e in range(E):
                rows.append({
                    "scheme": name,
                    "mode": mode,
                    "bits": bits,
                    "group_size": group_size,
                    "expert_id_in_bank": e,
                    "w_rel_fro": float(rel_fro_np[e]),
                    "w_rel_max": float(rel_max_np[e]),
                    "scale_mean": float(scales_mean_np[e]) if scales_mean_np.ndim == 1 else float(scales_mean_np),
                    "scale_max": float(scales_max_np[e]) if scales_max_np.ndim == 1 else float(scales_max_np),
                    "bias_mean": (float(biases_mean_np[e]) if biases_mean_np is not None else None),
                    "bias_max": (float(biases_max_np[e]) if biases_max_np is not None else None),
                    "error": None
                })

        except Exception as err:
            warns.append(f"[quant_sim] scheme={name} failed: {err}")
            # still emit rows with error so you can see coverage
            E = bank.shape[0]
            err_msg = f"{type(err).__name__}: {err}"
            for e_in_bank in range(E):
                rows.append({
                    "scheme": name,
                    "mode": mode,
                    "bits": bits,
                    "group_size": group_size,
                    "expert_id_in_bank": e_in_bank,
                    "w_rel_fro": None,
                    "w_rel_max": None,
                    "scale_mean": None,
                    "scale_max": None,
                    "bias_mean": None,
                    "bias_max": None,
                    "error": err_msg,
                })

    return pd.DataFrame(rows), warns


# ------------------------- main collection -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-path", default=None, help="Override config model_path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_config(run_dir)

    if args.model_path is not None:
        cfg["model_path"] = str(Path(args.model_path).expanduser().resolve())

    model_path = Path(cfg["model_path"]).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"model_path does not exist: {model_path}")

    scan_cfg = cfg["scan"]
    exts = scan_cfg.get("extensions", [".safetensors", ".npz"])
    max_files = scan_cfg.get("max_files", None)
    experts_only = bool(scan_cfg.get("experts_only", True))
    include_shared = bool(scan_cfg.get("include_shared_expert", True))
    inventory_all = bool(scan_cfg.get("inventory_all_tensors", True))

    parsing = cfg["parsing"]
    layer_re = re.compile(parsing["layer_regex"])
    expert_re = re.compile(parsing["expert_regex"])
    alias_map = parsing["proj_aliases"]
    shared_keywords = parsing.get("shared_expert_keywords", ["shared", "expert"])
    strict_packed_split = bool(parsing.get("strict_packed_split", True))

    rules = _compile_rules(cfg)

    mlx_cfg = cfg.get("mlx", {})
    mlx_enabled = bool(mlx_cfg.get("enabled", True))
    mlx_device = mlx_cfg.get("device", "cpu")

    schemes = [s for s in cfg.get("quant_schemes", []) if s.get("enabled", True)]
    cfg_stats = cfg["stats"]
    cache_idx_dir = run_dir / "cache" / "sampled_indices"

    debug_cfg = cfg.get("debug", {})
    dump_unmatched = bool(debug_cfg.get("dump_unmatched_tensors", True))
    progress_every = int(debug_cfg.get("print_progress_every_files", 1))

    # output collectors
    inventory_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    quant_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    warn_log: List[str] = []

    if mlx_enabled and schemes and mx is None:
        msg = "mlx is not importable; skipping quantization simulations"
        warnings.warn(msg)
        warn_log.append(f"[quant_sim] {msg}")
        mlx_enabled = False

    files = list(_iter_weight_files(model_path, exts))
    files.sort()
    if max_files is not None:
        files = files[: int(max_files)]

    t0 = time.time()
    print(f"[collect] scanning {len(files)} files under {model_path}")

    for fi, fpath in enumerate(files, start=1):
        if progress_every and (fi % progress_every == 0 or fi == 1):
            print(f"[collect] ({fi}/{len(files)}) {fpath}")

        for name, arr in _iter_tensors_from_file(fpath):
            # inventory
            if inventory_all:
                try:
                    nbytes = int(arr.nbytes)
                except Exception:
                    nbytes = None
                inventory_rows.append({
                    "file": str(fpath),
                    "tensor_name": name,
                    "dtype": str(arr.dtype),
                    "shape": tuple(arr.shape),
                    "ndim": int(arr.ndim),
                    "nbytes": nbytes
                })

            # only float weights for stats/sims
            if not _is_floatlike_dtype(arr.dtype):
                continue

            is_shared = _is_shared_expert(name, shared_keywords)
            is_expertish = ("experts" in name.lower()) or is_shared

            if experts_only:
                if not is_expertish:
                    continue
                if is_shared and not include_shared:
                    continue

            # try explicit rules, else fallback heuristics
            extracted = None
            try:
                extracted = _apply_rules(
                    name,
                    arr,
                    fpath,
                    rules,
                    layer_re,
                    expert_re,
                    alias_map,
                    shared_keywords,
                )
            except PackedSplitError as e:
                if strict_packed_split:
                    raise
                warn_log.append(f"[extract] {e}")
            except Exception as e:
                warn_log.append(f"[extract] rule application failed for {name}: {e}")

            if extracted is None:
                try:
                    extracted = _fallback_extract(name, arr, fpath, layer_re, expert_re, alias_map, shared_keywords)
                except Exception as e:
                    warn_log.append(f"[extract] fallback failed for {name}: {e}")
                    extracted = None

            if extracted is None:
                if dump_unmatched and experts_only and is_expertish:
                    unmatched_rows.append({
                        "file": str(fpath),
                        "tensor_name": name,
                        "dtype": str(arr.dtype),
                        "shape": tuple(arr.shape),
                        "ndim": int(arr.ndim),
                        "reason": "no_rule_match_or_proj_infer"
                    })
                continue

            for bank_obj in extracted:
                bank = bank_obj.bank

                # bank canonical shapes allowed: (L,E,R,C) (E,R,C) (R,C)
                if bank.ndim == 2:
                    bank = bank[None, ...]   # (1,R,C)
                if bank.ndim == 3:
                    # (E,R,C) good
                    pass
                elif bank.ndim == 4:
                    # iterate layers
                    pass
                else:
                    warn_log.append(f"[extract] unsupported canonical ndim={bank.ndim} for {bank_obj.derived_tensor}")
                    continue

                def process_one(layer_idx: Optional[int], bank_erc: np.ndarray):
                    E, R, C = bank_erc.shape
                    layer = layer_idx
                    if layer is None:
                        layer = bank_obj.layer_base
                    # If still None, use -1 (unknown); you can still do global stats.
                    layer_val = int(layer) if layer is not None else -1
                    block4 = (layer_val // 4) if layer_val >= 0 else None

                    stats = _per_expert_weight_stats(bank_erc, cfg_stats, cache_idx_dir)

                    # expert ids
                    if bank_obj.is_shared_expert:
                        expert_ids = np.full((E,), -1, dtype=np.int32)
                        routed = np.zeros((E,), dtype=bool)
                        shared = np.ones((E,), dtype=bool)
                    elif bank_obj.expert_single_id is not None and E == 1:
                        expert_ids = np.array([int(bank_obj.expert_single_id)], dtype=np.int32)
                        routed = np.ones((E,), dtype=bool)
                        shared = np.zeros((E,), dtype=bool)
                    else:
                        expert_ids = np.arange(E, dtype=np.int32)
                        routed = np.ones((E,), dtype=bool)
                        shared = np.zeros((E,), dtype=bool)

                    # matrix_stats rows
                    for e in range(E):
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
                            "rows": int(R),
                            "cols": int(C),
                            "dtype": str(bank_erc.dtype),
                        }
                        for k, v in stats.items():
                            row[k] = float(v[e]) if np.ndim(v) == 1 else float(v)
                        matrix_rows.append(row)

                    # quant sims
                    if mlx_enabled and schemes:
                        qdf, warns = _mlx_quant_sim(bank_erc, schemes, cfg_stats, mlx_device)
                        warn_log.extend(warns)

                        # attach identifiers to each quant row
                        for _, qr in qdf.iterrows():
                            e_in_bank = int(qr["expert_id_in_bank"])
                            if bank_obj.is_shared_expert:
                                exp_id = -1
                            elif bank_obj.expert_single_id is not None and E == 1:
                                exp_id = int(bank_obj.expert_single_id)
                            else:
                                exp_id = int(e_in_bank)

                            quant_rows.append({
                                "file": bank_obj.source_file,
                                "source_tensor": bank_obj.source_tensor,
                                "derived_tensor": bank_obj.derived_tensor,
                                "layer": layer_val,
                                "block4": block4,
                                "proj": bank_obj.proj,
                                "expert_id": exp_id,
                                "is_shared_expert": bool(bank_obj.is_shared_expert),
                                "rows": int(R),
                                "cols": int(C),

                                "scheme": qr["scheme"],
                                "mode": qr["mode"],
                                "bits": qr["bits"],
                                "group_size": qr["group_size"],
                                "w_rel_fro": qr["w_rel_fro"],
                                "w_rel_max": qr["w_rel_max"],
                                "scale_mean": qr["scale_mean"],
                                "scale_max": qr["scale_max"],
                                "bias_mean": qr["bias_mean"],
                                "bias_max": qr["bias_max"],
                                "error": qr["error"],
                            })

                if bank.ndim == 3:
                    process_one(layer_idx=bank_obj.layer_base, bank_erc=bank)
                else:
                    # (L,E,R,C)
                    L = bank.shape[0]
                    for li in range(L):
                        layer_idx = (bank_obj.layer_base + li) if bank_obj.layer_base is not None else li
                        process_one(layer_idx=layer_idx, bank_erc=bank[li])

    # write outputs
    fmt = cfg.get("output", {}).get("format", "parquet")
    compression = cfg.get("output", {}).get("compression", None)

    inv_df = pd.DataFrame(inventory_rows)
    ms_df = pd.DataFrame(matrix_rows)
    if quant_rows:
        qs_df = pd.DataFrame(quant_rows)
    else:
        qs_df = pd.DataFrame(columns=[
            "file",
            "source_tensor",
            "derived_tensor",
            "layer",
            "block4",
            "proj",
            "expert_id",
            "is_shared_expert",
            "rows",
            "cols",
            *QUANT_SIM_COLUMNS,
        ])
    um_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
    wl_df = pd.DataFrame({"warning": warn_log}) if warn_log else pd.DataFrame()

    _write_df(inv_df, run_dir / "data" / "tensor_inventory.parquet", fmt, compression)
    _write_df(ms_df,  run_dir / "data" / "matrix_stats.parquet", fmt, compression)
    _write_df(qs_df,  run_dir / "data" / "quant_sim.parquet", fmt, compression)
    if dump_unmatched:
        _write_df(um_df, run_dir / "data" / "unmatched_tensors.parquet", fmt, compression)
    if not wl_df.empty:
        _write_df(wl_df, run_dir / "logs" / "warnings.parquet", fmt, compression)

    dt = time.time() - t0
    print(f"[collect] done in {dt:.1f}s")
    print(f"[collect] tensor_inventory rows: {len(inv_df)}")
    print(f"[collect] matrix_stats rows:     {len(ms_df)}")
    print(f"[collect] quant_sim rows:        {len(qs_df)}")
    if dump_unmatched:
        print(f"[collect] unmatched rows:        {len(um_df)}")
    if not wl_df.empty:
        print(f"[collect] warnings:              {len(wl_df)}")


if __name__ == "__main__":
    main()
