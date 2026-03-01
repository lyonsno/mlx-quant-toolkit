from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

mx = None


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
    load_mlx: Optional[Callable[[], Any]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    bank: (E,R,C) numpy
    returns: quant_sim dataframe rows (per expert per scheme) AND list of warnings strings
    """
    eps = float(cfg_stats["eps"])
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

    # Use float16 to reduce memory; errors are relative so OK for ranking.
    w = bank.astype(np.float16, copy=False)
    w_mx = mx_mod.array(w)

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
                dtype=w_mx.dtype,
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

            mx_mod.eval(rel_fro, rel_max, scales_mean, scales_max)
            rel_fro_np = np.array(rel_fro).astype(np.float32)
            rel_max_np = np.array(rel_max).astype(np.float32)
            scales_mean_np = np.array(scales_mean).astype(np.float32)
            scales_max_np = np.array(scales_max).astype(np.float32)
            if biases_mean is not None:
                mx_mod.eval(biases_mean, biases_max)
                biases_mean_np = np.array(biases_mean).astype(np.float32)
                biases_max_np = np.array(biases_max).astype(np.float32)
            else:
                biases_mean_np = None
                biases_max_np = None

            e_count = rel_fro_np.shape[0]
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
                        "scale_mean": None,
                        "scale_max": None,
                        "bias_mean": None,
                        "bias_max": None,
                        "error": err_msg,
                    }
                )

    return pd.DataFrame(rows), warns
