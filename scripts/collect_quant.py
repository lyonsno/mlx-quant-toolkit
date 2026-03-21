from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import ml_dtypes
import numpy as np
import pandas as pd

mx = None

_DEFAULT_QUANT_COMPUTE_DTYPE = "fp16"
_DEFAULT_QUANT_SPECTRAL_POWER_ITERS = 12
_DEFAULT_QUANT_GRAM_SAMPLE_K = 128


def _load_mlx():
    global mx
    if mx is not None:
        return mx
    try:
        import mlx.core as mx_mod
    except Exception:
        return None
    mx = mx_mod
    return mx


def _get_positive_int_config(cfg: Dict[str, Any], key: str, default: int) -> int:
    raw_value = cfg.get(key, default)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
        raise ValueError(f"{key} must be an integer")
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _resolve_quant_compute_dtype(cfg: Dict[str, Any]) -> tuple[Any, str]:
    # Backward-compat bridge: legacy configs without this key keep the old
    # implicit fp16 behavior, while fresh init_run templates write explicit bf16.
    raw_value = cfg.get("quant_compute_dtype", _DEFAULT_QUANT_COMPUTE_DTYPE)
    if not isinstance(raw_value, str):
        raise ValueError("quant_compute_dtype must be one of: bf16, fp16, fp32")

    value = raw_value.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return ml_dtypes.bfloat16, "bfloat16"
    if value in {"fp16", "float16"}:
        return np.float16, "float16"
    if value in {"fp32", "float32"}:
        return np.float32, "float32"
    raise ValueError("quant_compute_dtype must be one of: bf16, fp16, fp32")


QUANT_SIM_COLUMNS = [
    "scheme",
    "mode",
    "bits",
    "group_size",
    "expert_id_in_bank",
    "w_rel_fro",
    "w_rel_max",
    "w_rel_spectral",
    "w_gram_cos_drift_sampled_rms",
    "scale_mean",
    "scale_max",
    "bias_mean",
    "bias_max",
    "error",
]


def _spectral_norm_estimate(matrix: np.ndarray, eps: float, power_iters: int) -> float:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or 0 in matrix.shape:
        return 0.0
    if max(matrix.shape) <= 64:
        return float(np.linalg.norm(matrix, ord=2))

    # Use an inexpensive deterministic fallback if the all-ones seed lies in
    # the nullspace of the matrix. This keeps the large-matrix estimator from
    # spuriously collapsing to zero on non-zero matrices.
    v = np.ones((matrix.shape[1],), dtype=np.float32)
    v /= np.linalg.norm(v) + eps
    init_u = matrix @ v
    init_u_norm = np.linalg.norm(init_u)
    if init_u_norm <= eps:
        for j in range(matrix.shape[1]):
            candidate = np.zeros((matrix.shape[1],), dtype=np.float32)
            candidate[j] = 1.0
            init_u = matrix @ candidate
            init_u_norm = np.linalg.norm(init_u)
            if init_u_norm > eps:
                v = candidate
                break
        else:
            return 0.0

    for _ in range(max(1, power_iters)):
        u = matrix @ v
        u_norm = np.linalg.norm(u)
        if u_norm <= eps:
            return 0.0
        u = u / (u_norm + eps)

        v = matrix.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm <= eps:
            return 0.0
        v = v / (v_norm + eps)

    return float(np.linalg.norm(matrix @ v))


def _sample_axis_indices(axis_count: int, sample_k: int, rng: np.random.Generator) -> np.ndarray:
    if axis_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if sample_k <= 0 or axis_count <= sample_k:
        return np.arange(axis_count, dtype=np.int64)
    return np.sort(rng.choice(axis_count, size=sample_k, replace=False).astype(np.int64))


def _normalized_offdiag_gram_drift_sampled(
    matrix: np.ndarray,
    quantized: np.ndarray,
    eps: float,
    sample_k: int,
    seed: int,
) -> float:
    matrix = np.asarray(matrix, dtype=np.float32)
    quantized = np.asarray(quantized, dtype=np.float32)
    axis_count = matrix.shape[0]
    if axis_count <= 1:
        return 0.0

    rng = np.random.default_rng(seed)
    idx = _sample_axis_indices(axis_count, sample_k, rng)
    sampled = matrix[idx, :]
    sampled_quantized = quantized[idx, :]

    sampled_norms = np.linalg.norm(sampled, axis=1, keepdims=True) + eps
    sampled_quantized_norms = np.linalg.norm(sampled_quantized, axis=1, keepdims=True) + eps
    sampled_unit = sampled / sampled_norms
    sampled_quantized_unit = sampled_quantized / sampled_quantized_norms

    diff = (sampled_unit @ sampled_unit.T) - (sampled_quantized_unit @ sampled_quantized_unit.T)
    np.fill_diagonal(diff, 0.0)
    sampled_count = sampled.shape[0]
    if sampled_count <= 1:
        return 0.0
    return float(np.linalg.norm(diff, ord="fro") / np.sqrt(sampled_count * (sampled_count - 1)))


def _gram_cos_drift_sampled_rms(
    matrix: np.ndarray,
    quantized: np.ndarray,
    eps: float,
    sample_k: int,
    seed: int,
) -> float:
    row_drift = _normalized_offdiag_gram_drift_sampled(matrix, quantized, eps, sample_k, seed)
    col_drift = _normalized_offdiag_gram_drift_sampled(matrix.T, quantized.T, eps, sample_k, seed + 1)
    return max(row_drift, col_drift)


def _mlx_quant_sim(
    bank: np.ndarray,
    schemes: List[Dict[str, Any]],
    cfg_stats: Dict[str, Any],
    device: str,
    load_mlx: Optional[Callable[[], Any]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    bank: (E,R,C) numpy
    returns: quant_sim dataframe rows (per expert per scheme) AND list of warnings strings
    """
    eps = float(cfg_stats["eps"])
    spectral_power_iters = _get_positive_int_config(
        cfg_stats,
        "quant_spectral_power_iters",
        _DEFAULT_QUANT_SPECTRAL_POWER_ITERS,
    )
    gram_sample_k = _get_positive_int_config(
        cfg_stats,
        "quant_gram_sample_k",
        _DEFAULT_QUANT_GRAM_SAMPLE_K,
    )
    quant_compute_dtype, quant_compute_dtype_name = _resolve_quant_compute_dtype(cfg_stats)
    sample_seed = int(cfg_stats.get("sample_seed", 1337))
    warns: List[str] = []

    loader = load_mlx or _load_mlx
    mx_mod = loader()
    if mx_mod is None:
        msg = "mlx is not importable; skipping quantization simulation"
        warnings.warn(msg)
        return pd.DataFrame(columns=QUANT_SIM_COLUMNS), [msg]

    if device == "cpu":
        try:
            mx_mod.set_default_device(mx_mod.cpu)
        except Exception:
            pass
    elif device == "gpu":
        try:
            mx_mod.set_default_device(mx_mod.gpu)
        except Exception:
            pass

    # Keep a NumPy view in the chosen compute dtype for downstream metric math, but
    # construct the MLX array from float32 input plus an MLX dtype token. Real MLX
    # rejects NumPy bf16 ndarrays in mx.array(...).
    w = bank.astype(quant_compute_dtype, copy=False)
    mlx_quant_compute_dtype = getattr(mx_mod, quant_compute_dtype_name, quant_compute_dtype)
    w_mx = mx_mod.array(bank.astype(np.float32, copy=False), dtype=mlx_quant_compute_dtype)

    rows = []
    for s in schemes:
        if not s.get("enabled", True):
            continue
        name = s["name"]
        mode = s["mode"]
        bits = int(s.get("bits", 4))
        group_size = int(s.get("group_size", 32))

        try:
            q = mx_mod.quantize(w_mx, group_size=group_size, bits=bits, mode=mode)
            if mode == "affine":
                wq, scales, biases = q
            else:
                wq, scales = q
                biases = None

            w_hat = mx_mod.dequantize(
                wq,
                scales,
                biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                dtype=mlx_quant_compute_dtype,
            )

            diff = w_hat - w_mx

            num = mx_mod.sqrt(mx_mod.sum(diff * diff, axis=(1, 2)))
            den = mx_mod.sqrt(mx_mod.sum(w_mx * w_mx, axis=(1, 2))) + eps
            rel_fro = num / den

            rel_max = mx_mod.max(mx_mod.abs(diff), axis=(1, 2)) / (
                mx_mod.max(mx_mod.abs(w_mx), axis=(1, 2)) + eps
            )

            # scale/bias stats (useful for diagnosing "why is this matrix hard?")
            # Keep this placeholder expression for parity with existing behavior.
            _ = mx_mod.mean(scales, axis=tuple(range(scales.ndim))[0:scales.ndim - 0])
            if scales.ndim >= 3:
                scales_mean = mx_mod.mean(scales, axis=(1, 2))
                scales_max = mx_mod.max(scales, axis=(1, 2))
            else:
                scales_mean = mx_mod.mean(scales, axis=0)
                scales_max = mx_mod.max(scales, axis=0)

            if biases is not None:
                if biases.ndim >= 3:
                    biases_mean = mx_mod.mean(biases, axis=(1, 2))
                    biases_max = mx_mod.max(biases, axis=(1, 2))
                else:
                    biases_mean = mx_mod.mean(biases, axis=0)
                    biases_max = mx_mod.max(biases, axis=0)
            else:
                biases_mean = None
                biases_max = None

            mx_mod.eval(w_hat, rel_fro, rel_max, scales_mean, scales_max)
            rel_fro_np = np.array(rel_fro).astype(np.float32)
            rel_max_np = np.array(rel_max).astype(np.float32)
            scales_mean_np = np.array(scales_mean).astype(np.float32)
            scales_max_np = np.array(scales_max).astype(np.float32)
            e_count = rel_fro_np.shape[0]
            rel_spectral_np = np.zeros((rel_fro_np.shape[0],), dtype=np.float32)
            gram_cos_drift_sampled_rms_np = np.zeros((rel_fro_np.shape[0],), dtype=np.float32)
            for e in range(e_count):
                w_e = w[e].astype(np.float32, copy=False)
                w_hat_e = np.array(w_hat[e]).astype(np.float32)
                rel_spectral_np[e] = float(
                    _spectral_norm_estimate(w_hat_e - w_e, eps, spectral_power_iters)
                    / (_spectral_norm_estimate(w_e, eps, spectral_power_iters) + eps)
                )
                gram_cos_drift_sampled_rms_np[e] = float(
                    _gram_cos_drift_sampled_rms(
                        w_e,
                        w_hat_e,
                        eps,
                        gram_sample_k,
                        sample_seed + e,
                    )
                )
            if biases_mean is not None:
                mx_mod.eval(biases_mean, biases_max)
                biases_mean_np = np.array(biases_mean).astype(np.float32)
                biases_max_np = np.array(biases_max).astype(np.float32)
            else:
                biases_mean_np = None
                biases_max_np = None

            for e in range(e_count):
                rows.append(
                    {
                        "scheme": name,
                        "mode": mode,
                        "bits": bits,
                        "group_size": group_size,
                        "expert_id_in_bank": e,
                        "w_rel_fro": float(rel_fro_np[e]),
                        "w_rel_max": float(rel_max_np[e]),
                        "w_rel_spectral": float(rel_spectral_np[e]),
                        "w_gram_cos_drift_sampled_rms": float(gram_cos_drift_sampled_rms_np[e]),
                        "scale_mean": float(scales_mean_np[e])
                        if scales_mean_np.ndim == 1
                        else float(scales_mean_np),
                        "scale_max": float(scales_max_np[e])
                        if scales_max_np.ndim == 1
                        else float(scales_max_np),
                        "bias_mean": (
                            float(biases_mean_np[e]) if biases_mean_np is not None else None
                        ),
                        "bias_max": (
                            float(biases_max_np[e]) if biases_max_np is not None else None
                        ),
                        "error": None,
                    }
                )

        except Exception as err:
            warns.append(f"[quant_sim] scheme={name} failed: {err}")
            # Still emit rows with error so coverage remains visible in output tables.
            e_count = bank.shape[0]
            err_msg = f"{type(err).__name__}: {err}"
            for e_in_bank in range(e_count):
                rows.append(
                    {
                        "scheme": name,
                        "mode": mode,
                        "bits": bits,
                        "group_size": group_size,
                        "expert_id_in_bank": e_in_bank,
                        "w_rel_fro": None,
                        "w_rel_max": None,
                        "w_rel_spectral": None,
                        "w_gram_cos_drift_sampled_rms": None,
                        "scale_mean": None,
                        "scale_max": None,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": err_msg,
                    }
                )

    return pd.DataFrame(rows), warns
