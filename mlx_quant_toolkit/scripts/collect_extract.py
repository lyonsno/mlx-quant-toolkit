from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


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
    tokens = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not tokens:
        return False

    # Keep shared_expert_keywords conjunctive: every configured token must appear.
    return all(t in n for t in tokens)


def _infer_proj(name: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    n = name.lower()
    for canonical, aliases in alias_map.items():
        for a in aliases:
            alias = a.lower()
            if alias.startswith(".") and alias.endswith("."):
                if alias in n:
                    return canonical
            else:
                pattern = r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])"
                if re.search(pattern, n):
                    return canonical
    return None


def _suggest_proj(raw_proj: str, alias_map: Dict[str, List[str]]) -> Tuple[str, str]:
    """
    Suggest a canonical projection name for an unmapped token.
    Returns (suggested_proj, suggested_match), or ("", "") when no close match exists.
    """
    if not alias_map:
        return "", ""

    canonical_by_lower = {k.lower(): k for k in alias_map.keys()}
    match_to_original: Dict[str, str] = {}

    for canonical, aliases in alias_map.items():
        cl = canonical.lower()
        if cl not in match_to_original:
            match_to_original[cl] = canonical
        for alias in aliases:
            al = str(alias).lower()
            if al not in match_to_original:
                match_to_original[al] = str(alias)

    if not match_to_original:
        return "", ""

    raw_lower = raw_proj.lower()
    candidates = list(match_to_original.keys())
    matches = difflib.get_close_matches(raw_lower, candidates, n=1, cutoff=0.6)
    if not matches:
        return "", ""

    matched_lower = matches[0]
    matched_text = match_to_original[matched_lower]

    if matched_lower in canonical_by_lower:
        return canonical_by_lower[matched_lower], matched_text

    suggested_proj = _infer_proj(matched_text, alias_map)
    if suggested_proj is None:
        return "", matched_text
    return suggested_proj, matched_text


def _record_proj_issue(
    acc: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
    *,
    context: str,
    rule_name: str,
    raw_proj: str,
    resolved_proj: str,
    action: str,
    source_file: str,
    source_tensor: str,
    derived_tensor: str,
    suggested_proj: str,
    suggested_match: str,
) -> None:
    """
    Aggregate repeated proj canonicalization issues by semantic key.
    Count increments per decision-time event; example fields keep first seen values.
    """
    key = (context, rule_name, raw_proj, resolved_proj, action)
    if key in acc:
        acc[key]["count"] = int(acc[key]["count"]) + 1
        return

    acc[key] = {
        "context": context,
        "rule_name": rule_name,
        "raw_proj": raw_proj,
        "resolved_proj": resolved_proj,
        "action": action,
        "count": 1,
        "example_file": source_file,
        "example_source_tensor": source_tensor,
        "example_derived_tensor": derived_tensor,
        "suggested_proj": suggested_proj,
        "suggested_match": suggested_match,
    }


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

    outs: List[np.ndarray] = []
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

    axes: List[int] = []
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
    proj_group_strict: bool,
    proj_issue_acc: Optional[Dict[Tuple[str, str, str, str, str], Dict[str, Any]]] = None,
    unmatched_reason_override: Optional[Dict[Tuple[str, str], str]] = None,
    skip_fallback: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[List[ExtractedBank]]:
    canonical_keys_by_lower: Dict[str, str] = {}
    if alias_map:
        canonical_keys_by_lower = {k.lower(): k for k in alias_map.keys()}

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
                raw_lower = raw.lower()
                inferred = _infer_proj(raw, alias_map)
                is_canonical_key = raw_lower in canonical_keys_by_lower
                if is_canonical_key:
                    proj = canonical_keys_by_lower[raw_lower]
                else:
                    proj = inferred
                if proj is None:
                    unmapped = bool(alias_map) and (not is_canonical_key) and (inferred is None)
                    if proj_group_strict:
                        key = (str(fpath), name)
                        if skip_fallback is not None:
                            skip_fallback.add(key)
                        if unmatched_reason_override is not None:
                            if unmapped:
                                unmatched_reason_override[key] = "proj_group_strict_unmapped"
                            else:
                                unmatched_reason_override[key] = "proj_group_strict_no_alias_map"
                        if unmapped and proj_issue_acc is not None:
                            suggested_proj, suggested_match = _suggest_proj(raw, alias_map)
                            _record_proj_issue(
                                proj_issue_acc,
                                context="proj_group",
                                rule_name=r.name,
                                raw_proj=raw,
                                resolved_proj=raw,
                                action="dropped_strict",
                                source_file=str(fpath),
                                source_tensor=name,
                                derived_tensor=f"{name}::{raw}",
                                suggested_proj=suggested_proj,
                                suggested_match=suggested_match,
                            )
                        return None
                    proj = raw
                    if unmapped and proj_issue_acc is not None:
                        suggested_proj, suggested_match = _suggest_proj(raw, alias_map)
                        _record_proj_issue(
                            proj_issue_acc,
                            context="proj_group",
                            rule_name=r.name,
                            raw_proj=raw,
                            resolved_proj=proj,
                            action="kept_raw",
                            source_file=str(fpath),
                            source_tensor=name,
                            derived_tensor=f"{name}::{proj}",
                            suggested_proj=suggested_proj,
                            suggested_match=suggested_match,
                        )
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
        for raw_proj, part in zip(proj_list, parts):
            if not alias_map:
                resolved_proj = raw_proj
            else:
                # Normalize explicit canonical keys case-insensitively before alias inference.
                raw_proj_lower = raw_proj.lower()
                if raw_proj_lower in canonical_keys_by_lower:
                    resolved_proj = canonical_keys_by_lower[raw_proj_lower]
                    inferred = resolved_proj
                else:
                    inferred = _infer_proj(raw_proj, alias_map)
                    resolved_proj = inferred or raw_proj

                unmapped = (
                    bool(alias_map)
                    and (raw_proj_lower not in canonical_keys_by_lower)
                    and (inferred is None)
                )
                if unmapped and proj_issue_acc is not None:
                    suggested_proj, suggested_match = _suggest_proj(raw_proj, alias_map)
                    _record_proj_issue(
                        proj_issue_acc,
                        context="packed_split",
                        rule_name=r.name,
                        raw_proj=raw_proj,
                        resolved_proj=resolved_proj,
                        action="kept_raw",
                        source_file=str(fpath),
                        source_tensor=name,
                        derived_tensor=f"{name}::split[{axis_kind}]::{resolved_proj}",
                        suggested_proj=suggested_proj,
                        suggested_match=suggested_match,
                    )

            derived = f"{name}::split[{axis_kind}]::{resolved_proj}"
            banks.append(ExtractedBank(
                source_file=str(fpath),
                source_tensor=name,
                derived_tensor=derived,
                proj=resolved_proj,
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
