#!/usr/bin/env python3

import json
import re
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LAYER_KEYS = ["num_hidden_layers", "n_layer", "num_layers"]
_HIDDEN_KEYS = ["hidden_size", "d_model"]
_NUM_EXPERTS_KEYS = ["num_experts", "n_routed_experts", "num_local_experts"]
_NUM_EXPERTS_PER_TOK_KEYS = ["num_experts_per_tok", "experts_per_token"]
_MOE_INTERMEDIATE_KEYS = ["moe_intermediate_size", "expert_intermediate_size"]
_SHARED_INTERMEDIATE_KEYS = ["shared_expert_intermediate_size"]
_N_SHARED_EXPERTS_KEYS = ["n_shared_experts"]

_EXTRA_KEYS = [
    "decoder_sparse_step",
    "full_attention_interval",
    "first_k_dense_replace",
    "mlp_only_layers",
]

_LOG_KEYS = list(dict.fromkeys(
    _LAYER_KEYS
    + _HIDDEN_KEYS
    + _NUM_EXPERTS_KEYS
    + _NUM_EXPERTS_PER_TOK_KEYS
    + _MOE_INTERMEDIATE_KEYS
    + _SHARED_INTERMEDIATE_KEYS
    + _N_SHARED_EXPERTS_KEYS
    + _EXTRA_KEYS
    + ["model_type", "architectures"]
))


def find_config_json(model_path: Path) -> Optional[Path]:
    if model_path.is_file():
        base = model_path.parent
    else:
        base = model_path

    root_cfg = base / "config.json"
    if root_cfg.is_file():
        return root_cfg

    for rel in [
        "model/config.json",
        "configs/config.json",
        "config/config.json",
        "hf/config.json",
    ]:
        cand = base / rel
        if cand.is_file():
            return cand

    return None


def parse_config_json(path: Path) -> Dict[str, Any]:
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

    if not isinstance(data, dict):
        return {}

    return data


def find_safetensors_index_json(model_path: Path) -> Optional[Path]:
    if model_path.is_file():
        base = model_path.parent
    else:
        base = model_path

    root_index = base / "model.safetensors.index.json"
    if root_index.is_file():
        return root_index

    matches = sorted(base.glob("*.safetensors.index.json"))
    if matches:
        return matches[0]
    return None


def parse_safetensors_index(path: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    try:
        raw = path.read_text()
    except Exception as e:
        raise ValueError(f"unable to read index: {e}") from e

    if not raw.strip():
        raise ValueError("index file is empty")

    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"invalid JSON in index: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("index JSON must be an object")

    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index JSON missing weight_map")

    out: Dict[str, str] = {}
    for k, v in weight_map.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
        else:
            raise ValueError("index weight_map must map string tensor names to string shard names")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return out, metadata


def trim_config_for_log(config: Dict[str, Any], max_chars: int = 200_000) -> Dict[str, Any]:
    try:
        payload = json.dumps(config)
    except Exception:
        return {}

    if len(payload) <= max_chars:
        return config

    trimmed = {k: config.get(k) for k in _LOG_KEYS if k in config}
    trimmed["_meta_trimmed"] = {
        "original_keys": len(config),
        "original_chars": len(payload),
    }
    return trimmed


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        s = value.strip()
        if re.match(r"^[+-]?\d+$", s):
            try:
                return int(s)
            except Exception:
                return None
    return None


def _coerce_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            iv = _coerce_int(item)
            if iv is None:
                return None
            out.append(iv)
        return out
    return None


def _first_int(cfg: Dict[str, Any], keys: List[str]) -> Optional[int]:
    for k in keys:
        if k in cfg:
            val = _coerce_int(cfg.get(k))
            if val is not None:
                return val
    return None


@dataclass
class ModelShapeBudget:
    num_hidden_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    n_shared_experts: Optional[int] = None
    decoder_sparse_step: Optional[int] = None
    full_attention_interval: Optional[int] = None
    first_k_dense_replace: Optional[int] = None
    mlp_only_layers: Optional[List[int]] = None

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> "ModelShapeBudget":
        if not isinstance(cfg, dict):
            return cls()

        mlp_only = None
        if "mlp_only_layers" in cfg:
            mlp_only = _coerce_int_list(cfg.get("mlp_only_layers"))

        return cls(
            num_hidden_layers=_first_int(cfg, _LAYER_KEYS),
            hidden_size=_first_int(cfg, _HIDDEN_KEYS),
            moe_intermediate_size=_first_int(cfg, _MOE_INTERMEDIATE_KEYS),
            shared_expert_intermediate_size=_first_int(cfg, _SHARED_INTERMEDIATE_KEYS),
            num_experts=_first_int(cfg, _NUM_EXPERTS_KEYS),
            num_experts_per_tok=_first_int(cfg, _NUM_EXPERTS_PER_TOK_KEYS),
            n_shared_experts=_first_int(cfg, _N_SHARED_EXPERTS_KEYS),
            decoder_sparse_step=_coerce_int(cfg.get("decoder_sparse_step")),
            full_attention_interval=_coerce_int(cfg.get("full_attention_interval")),
            first_k_dense_replace=_coerce_int(cfg.get("first_k_dense_replace")),
            mlp_only_layers=mlp_only,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
