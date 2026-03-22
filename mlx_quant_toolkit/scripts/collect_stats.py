from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


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

    # Work in float32 for stability and predictable performance.
    w = bank.astype(np.float32, copy=False)
    e_count, rows, cols = w.shape
    abs_w = np.abs(w)

    mean = w.mean(axis=(1, 2))
    std = w.std(axis=(1, 2))
    mean_abs = abs_w.mean(axis=(1, 2))
    rms = np.sqrt((w * w).mean(axis=(1, 2)))
    max_abs = abs_w.max(axis=(1, 2))

    # Sampled percentiles per expert, using shared indices for apples-to-apples comparison.
    flat = abs_w.reshape(e_count, -1)
    idx = _get_sample_indices(cache_dir, total=rows * cols, k=sample_k, seed=seed)
    samp = flat[:, idx]
    pvals: Dict[float, np.ndarray] = {}
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

    # Groupwise outliers along the last dimension (matching quant grouping semantics).
    for group_size in group_sizes:
        group_size = int(group_size)
        if group_size <= 0 or (cols % group_size) != 0:
            stats[f"g{group_size}_p{int(group_p)}_outlier"] = np.full((e_count,), np.nan, dtype=np.float32)
            stats[f"g{group_size}_max_outlier"] = np.full((e_count,), np.nan, dtype=np.float32)
            continue

        resh = abs_w.reshape(e_count, rows, cols // group_size, group_size)
        gmax = resh.max(axis=-1)
        gmean = resh.mean(axis=-1)
        ratio = gmax / (gmean + eps)
        ratio_flat = ratio.reshape(e_count, -1)
        stats[f"g{group_size}_p{int(group_p)}_outlier"] = np.percentile(ratio_flat, group_p, axis=1)
        stats[f"g{group_size}_max_outlier"] = ratio_flat.max(axis=1)

    return stats
