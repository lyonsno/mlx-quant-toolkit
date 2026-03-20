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
import importlib.util
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

mx = None


def _is_floatlike_dtype(dtype: np.dtype) -> bool:
    try:
        if np.issubdtype(dtype, np.floating):
            return True
    except Exception:
        pass
    return "bfloat16" in str(dtype).lower()


# ------------------------- IO helpers ----------------------------------------

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


def _utc_now_iso() -> str:
    # Keep this as a simple ISO-ish UTC string so it stays human-readable and easy to parse.
    # Example: "2026-01-06T12:34:56Z"
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path_for_manifest(path: Path, run_dir: Path) -> str:
    # Prefer run_dir-relative paths for portability, fall back to absolute.
    try:
        return str(path.relative_to(run_dir))
    except Exception:
        return str(path)


def _path_for_scan_plan(path: Path, model_path: Path) -> str:
    # Prefer model_path-relative paths so scan plans remain portable.
    base = model_path.parent if model_path.is_file() else model_path
    return os.path.relpath(str(path), start=str(base))


def _artifact_entry(meta: Dict[str, Any], run_dir: Path, rows: int) -> Dict[str, Any]:
    # Normalize the write metadata into a JSON-friendly manifest entry.
    entry = dict(meta)
    entry["rows"] = int(rows)
    raw_path = entry.get("path")
    if isinstance(raw_path, Path):
        entry["path"] = _path_for_manifest(raw_path, run_dir)
    elif isinstance(raw_path, str):
        entry["path"] = _path_for_manifest(Path(raw_path), run_dir)
    return entry


def _resolve_path_str(path_like: Any) -> Optional[str]:
    if path_like is None:
        return None
    try:
        return str(Path(path_like).expanduser().resolve())
    except Exception:
        return str(path_like)


def _system_exit_is_error(exc: SystemExit) -> bool:
    code = exc.code
    if code is None:
        return False
    if isinstance(code, int):
        return code != 0
    return True


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, str):
            return code
        if code is None:
            return "SystemExit"
        return f"SystemExit({code})"
    text = str(exc)
    return text if text else type(exc).__name__


def _write_run_failure_artifact(exc: BaseException, ctx: Dict[str, Any]) -> None:
    run_dir_raw = ctx.get("run_dir")
    if run_dir_raw is None:
        return

    run_dir = Path(run_dir_raw)
    manifest = ctx.get("manifest")
    manifest_dict = manifest if isinstance(manifest, dict) else {}
    cli_overrides = ctx.get("cli_overrides")
    cli_overrides_dict = cli_overrides if isinstance(cli_overrides, dict) else {}

    configured_model_path = ctx.get("configured_model_path")
    model_path = ctx.get("model_path")
    model_path_source = None
    if "model_path" in cli_overrides_dict:
        model_path_source = "cli_override"
    elif configured_model_path is not None:
        model_path_source = "config"

    index_path = ctx.get("index_path")
    index_path_found = ctx.get("index_path_found")
    index_context_path = None
    try:
        if index_path is not None and Path(index_path).exists():
            index_context_path = _resolve_path_str(index_path)
        elif index_path_found is not None and Path(index_path_found).exists():
            index_context_path = _resolve_path_str(index_path_found)
    except Exception:
        index_context_path = _resolve_path_str(index_path) or _resolve_path_str(index_path_found)

    payload: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "status": "error",
        "run": {
            "run_dir": _resolve_path_str(run_dir),
            "model_id": manifest_dict.get("model_id"),
            "run_name": manifest_dict.get("run_name"),
            "created_at": manifest_dict.get("created_at"),
        },
        "model_path": {
            "configured": _resolve_path_str(configured_model_path),
            "resolved": _resolve_path_str(model_path),
            "source": model_path_source,
        },
        "cli_overrides": cli_overrides_dict,
        "index": {
            "status": str(ctx.get("index_status", "not_initialized")),
            "searched": bool(ctx.get("index_searched", False)),
            "found": bool(ctx.get("index_found", False)),
            "active": bool(ctx.get("index_active", False)),
            "index_path": index_context_path,
        },
        "error": {
            "type": type(exc).__name__,
            "message": _exception_message(exc),
        },
    }

    index_error = ctx.get("index_error")
    if index_error:
        payload["index"]["error"] = str(index_error)

    tb_text = traceback.format_exc()
    if tb_text and tb_text.strip():
        payload["traceback"] = tb_text

    try:
        _write_json(payload, run_dir / "logs" / "run_failure.json")
    except Exception:
        # Best effort only: failing to write failure metadata should not replace the original error.
        pass


def _clear_run_failure_artifact(run_dir: Path) -> None:
    try:
        (run_dir / "logs" / "run_failure.json").unlink(missing_ok=True)
    except Exception:
        # Best effort only: stale failure metadata should not break successful runs.
        pass


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "analysis_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path} (run init_run.py first)")
    return json.loads(cfg_path.read_text())


_METADATA_MODULE = None
_METADATA_LOADED = False


def _get_metadata_module() -> Optional[Any]:
    global _METADATA_MODULE, _METADATA_LOADED
    if _METADATA_LOADED:
        return _METADATA_MODULE
    _METADATA_LOADED = True

    path = Path(__file__).resolve().parent / "metadata.py"
    if not path.exists():
        _METADATA_MODULE = None
        return None

    spec = importlib.util.spec_from_file_location("metadata", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _METADATA_MODULE = module
    return module


def _load_local_helper_module(module_name: str) -> Any:
    """
    Load a sibling helper module by file path so this script works both when run
    directly (`python scripts/collect_data.py`) and when loaded via importlib in tests.
    """
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None:
            try:
                if Path(existing_file).resolve() == module_path:
                    return existing
            except Exception:
                pass

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_shard_id(shard: str) -> str:
    return shard.replace("\\", "/")


# ------------------------- parsing -------------------------------------------
# Helper extraction:
# Keep compatibility by re-exporting moved helpers from collect_data so existing
# tests and downstream imports keep working while the file is split incrementally.
_collect_extract_mod = _load_local_helper_module("collect_extract")
_collect_stats_mod = _load_local_helper_module("collect_stats")
_collect_io_mod = _load_local_helper_module("collect_io")
_collect_quant_mod = _load_local_helper_module("collect_quant")
_collect_pipeline_mod = _load_local_helper_module("collect_pipeline")
_collect_reporting_mod = _load_local_helper_module("collect_reporting")

Rule = _collect_extract_mod.Rule
PackedSplitError = _collect_extract_mod.PackedSplitError
ExtractedBank = _collect_extract_mod.ExtractedBank
_compile_rules = _collect_extract_mod._compile_rules
_parse_int_from_regex = _collect_extract_mod._parse_int_from_regex
_is_shared_expert = _collect_extract_mod._is_shared_expert
_infer_proj = _collect_extract_mod._infer_proj
_suggest_proj = _collect_extract_mod._suggest_proj
_record_proj_issue = _collect_extract_mod._record_proj_issue
_split_along_axis = _collect_extract_mod._split_along_axis
_canonicalize_layout = _collect_extract_mod._canonicalize_layout
_apply_rules = _collect_extract_mod._apply_rules
_fallback_extract = _collect_extract_mod._fallback_extract

_get_sample_indices = _collect_stats_mod._get_sample_indices
_per_expert_weight_stats = _collect_stats_mod._per_expert_weight_stats

_safe_read_json_dict = _collect_io_mod._safe_read_json_dict
_write_json = _collect_io_mod._write_json
_write_df = _collect_io_mod._write_df
_iter_weight_files = _collect_io_mod._iter_weight_files
_iter_tensors_from_file = _collect_io_mod._iter_tensors_from_file

QUANT_SIM_COLUMNS = _collect_quant_mod.QUANT_SIM_COLUMNS
record_example = _collect_pipeline_mod.record_example
process_one_bank = _collect_pipeline_mod.process_one_bank
process_extracted_banks = _collect_pipeline_mod.process_extracted_banks
build_index_report_data = _collect_reporting_mod.build_index_report_data
build_index_summary = _collect_reporting_mod.build_index_summary


def _looks_like_moe_proj_tensor(
    name: str,
    alias_map: Dict[str, List[str]],
    expert_re: re.Pattern,
) -> bool:
    name_lower = name.lower()
    moe_namespace_pattern = r"(?:^|\.)moe\."
    if re.search(moe_namespace_pattern, name_lower) is None:
        return False

    # Keep the .moe. fallback narrow: only a direct `.moe.<proj-token>` child
    # or `.moe.<expert-id>.<proj-token>` path counts as expertish. This admits
    # alias-only names like `.moe.w2.weight` and explicit single-expert tensors
    # such as `.moe.7.w2.weight` while excluding nested router paths like
    # `.moe.router.w1.weight`.
    tokens: set[str] = set()
    for canonical, aliases in alias_map.items():
        canonical_token = str(canonical).strip().lower()
        if canonical_token:
            tokens.add(canonical_token)
        for alias in aliases:
            alias_token = str(alias).strip().lower()
            if alias_token and not (alias_token.startswith(".") and alias_token.endswith(".")):
                tokens.add(alias_token)

    for token in tokens:
        pattern = moe_namespace_pattern + re.escape(token) + r"(?:\.|$)"
        if re.search(pattern, name_lower):
            return True

    match = expert_re.search(name)
    if match is None:
        return False

    suffix = name_lower[match.end() :]
    for token in tokens:
        if re.match(r"(?:\.)?" + re.escape(token) + r"(?:\.|$)", suffix):
            return True
    return False


def _looks_like_empty_collect_moe_candidate(name: str, expert_re: re.Pattern) -> bool:
    name_lower = name.lower()
    if re.search(r"(?:^|\.)moe\.", name_lower) is None:
        return False

    match = expert_re.search(name)
    if match is None:
        return False

    suffix = name_lower[match.end() :]
    token_match = re.match(r"(?:\.)?([a-z0-9_]+)\.weight$", suffix)
    if token_match is None:
        return False

    token = token_match.group(1)
    if token in {"w1", "w2", "w3", "down_proj", "gate_proj", "up_proj"}:
        return True
    return token.startswith(("down", "gate", "up")) and ("proj" in token)


def _matches_enabled_rule_candidate(name: str, arr: np.ndarray, rules: List[Rule]) -> bool:
    # Keep explicit enabled rules authoritative for experts_only admission so
    # non-strict proj_group names can still reach the extractor that owns the
    # actual normalization contract.
    for rule in rules:
        if not rule.enabled:
            continue
        if rule.ndim is not None and arr.ndim != rule.ndim:
            continue
        if rule.regex.match(name):
            return True
    return False


def _mlx_quant_sim(
    bank: np.ndarray,
    schemes: List[Dict[str, Any]],
    cfg_stats: Dict[str, Any],
    device: str,
) -> Tuple[pd.DataFrame, List[str]]:
    # Preserve collect_data.mx monkeypatch behavior by injecting this module's
    # loader into the extracted quant helper implementation.
    return _collect_quant_mod._mlx_quant_sim(
        bank,
        schemes,
        cfg_stats,
        device,
        load_mlx=_load_mlx,
    )


# ------------------------- main collection -----------------------------------

def _main_impl(args: argparse.Namespace, failure_ctx: Dict[str, Any]) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    failure_ctx["run_dir"] = run_dir
    cfg = _load_config(run_dir)
    manifest = _safe_read_json_dict(run_dir / "manifest.json")
    failure_ctx["manifest"] = manifest

    # Preserve the config-provided model_path so run_context can show overrides.
    configured_model_path = cfg.get("model_path")
    failure_ctx["configured_model_path"] = configured_model_path
    cli_overrides: Dict[str, Any] = {}
    failure_ctx["cli_overrides"] = cli_overrides

    if args.model_path is not None:
        override_path = Path(args.model_path).expanduser().resolve()
        cfg["model_path"] = str(override_path)
        cli_overrides["model_path"] = str(override_path)

    model_path = Path(cfg["model_path"]).expanduser().resolve()
    failure_ctx["model_path"] = model_path
    if not model_path.exists():
        raise SystemExit(f"model_path does not exist: {model_path}")

    scan_cfg = cfg["scan"]
    exts = scan_cfg.get("extensions", [".safetensors", ".npz"])
    max_files = scan_cfg.get("max_files", None)
    experts_only = bool(scan_cfg.get("experts_only", True))
    include_shared = bool(scan_cfg.get("include_shared_expert", True))
    inventory_all = bool(scan_cfg.get("inventory_all_tensors", True))
    use_index = bool(scan_cfg.get("use_safetensors_index_json", True))
    strict_index = bool(scan_cfg.get("strict_index", False))
    model_path_is_file = bool(model_path.is_file())
    model_path_kind = "file" if model_path_is_file else "dir"

    # Fail fast when strict_index is enabled but index discovery is disabled.
    if strict_index and not use_index:
        raise SystemExit("strict_index requires use_safetensors_index_json=true")

    parsing = cfg["parsing"]
    layer_re = re.compile(parsing["layer_regex"])
    expert_re = re.compile(parsing["expert_regex"])
    alias_map = parsing["proj_aliases"]
    shared_keywords = parsing.get("shared_expert_keywords", ["shared", "expert"])
    strict_packed_split = bool(parsing.get("strict_packed_split", True))
    proj_group_strict = bool(parsing.get("proj_group_strict", False))

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

    metadata_cfg = cfg.get("metadata", {})
    metadata_enabled = bool(metadata_cfg.get("enabled", False))
    metadata_config_path = metadata_cfg.get("config_path", None)

    # output collectors
    inventory_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    quant_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    warn_log: List[str] = []
    proj_issue_acc: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    unmatched_reason_override: Dict[Tuple[str, str], str] = {}
    skip_fallback: Set[Tuple[str, str]] = set()

    # Run-health summary accumulators.
    #
    # High-level goal: produce a compact "did this run look sane?" report that can be
    # inspected quickly without loading large parquet/CSV artifacts.
    extracted_by_rule = 0
    extracted_by_fallback = 0
    unmatched_expertish = 0
    example_rule_extracted: List[str] = []
    example_fallback_extracted: List[str] = []
    example_unmatched_expertish: List[str] = []
    smoke_candidate_tensors_observed = 0
    example_smoke_candidates: List[str] = []

    if metadata_enabled:
        meta_mod = _get_metadata_module()
        if meta_mod is None:
            warn_log.append("[meta] metadata module unavailable; skipping")
        else:
            cfg_path = None
            if metadata_config_path:
                override = Path(metadata_config_path).expanduser()
                if not override.is_absolute():
                    base = model_path if model_path.is_dir() else model_path.parent
                    override = base / override
                override = override.resolve()
                if override.exists():
                    cfg_path = override
                else:
                    warn_log.append(f"[meta] config_path not found: {override}")

            if cfg_path is None:
                cfg_path = meta_mod.find_config_json(model_path)

            if cfg_path is None or not cfg_path.exists():
                warn_log.append("[meta] config.json not found; skipping")
            else:
                parsed = meta_mod.parse_config_json(cfg_path)
                if not parsed:
                    warn_log.append(f"[meta] config.json empty or invalid: {cfg_path}")
                else:
                    budget = meta_mod.ModelShapeBudget.from_config_dict(parsed)
                    shape_budget = budget.to_dict()
                    raw_cfg = meta_mod.trim_config_for_log(parsed)

                    logs_dir = run_dir / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    # CONTRACT SURFACE: logs/model_config.raw.json + logs/model_shape_budget.json
                    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
                    # Tests: rg 'model_shape_budget.json' tests/
                    (logs_dir / "model_config.raw.json").write_text(
                        json.dumps(raw_cfg, indent=2)
                    )
                    (logs_dir / "model_shape_budget.json").write_text(
                        json.dumps(
                            {
                                "config_path": str(cfg_path),
                                "shape_budget": shape_budget,
                            },
                            indent=2,
                        )
                    )

                    parts = []
                    if budget.hidden_size is not None:
                        parts.append(f"hidden={budget.hidden_size}")
                    if budget.num_hidden_layers is not None:
                        parts.append(f"layers={budget.num_hidden_layers}")
                    if budget.moe_intermediate_size is not None:
                        parts.append(f"moe_int={budget.moe_intermediate_size}")
                    if budget.shared_expert_intermediate_size is not None:
                        parts.append(f"shared_int={budget.shared_expert_intermediate_size}")
                    if budget.num_experts is not None:
                        parts.append(f"experts={budget.num_experts}")
                    if budget.num_experts_per_tok is not None:
                        parts.append(f"topk={budget.num_experts_per_tok}")

                    if parts:
                        print("[meta] " + " ".join(parts))
                    else:
                        warn_log.append(f"[meta] no recognized fields in {cfg_path}")

    index_path = None
    index_path_found = None
    weight_map = None
    index_metadata = {}
    expected_shards: set[str] = set()
    expected_shard_paths: Dict[str, Path] = {}
    extra_safetensors_files_on_disk: set[str] = set()
    index_active = False
    index_searched = False
    index_found = False
    index_status = "disabled"
    index_error = None
    failure_ctx.update(
        {
            "index_status": index_status,
            "index_searched": bool(index_searched),
            "index_found": bool(index_found),
            "index_active": bool(index_active),
            "index_path": index_path,
            "index_path_found": index_path_found,
            "index_error": index_error,
        }
    )

    if use_index:
        index_searched = True
        index_status = "not_found"
        index_mod = _get_metadata_module()
        if index_mod is None:
            index_status = "unavailable"
            index_error = "metadata module unavailable"
            warn_log.append("[index] metadata module unavailable; skipping index")
        else:
            find_fn = getattr(index_mod, "find_safetensors_index_json", None)
            parse_fn = getattr(index_mod, "parse_safetensors_index", None)
            if find_fn is None or parse_fn is None:
                index_status = "unavailable"
                index_error = "index helpers unavailable"
                warn_log.append("[index] index helpers unavailable; skipping index")
            else:
                index_path_found = find_fn(model_path)
                index_path = index_path_found
                if index_path is not None and index_path.exists():
                    index_found = True
                    try:
                        weight_map, index_metadata = parse_fn(index_path)
                        weight_map = {k: _normalize_shard_id(v) for k, v in weight_map.items()}
                    except Exception as e:
                        warn_log.append(f"[index] failed to parse index {index_path}: {e}")
                        index_path = None
                        weight_map = None
                        index_status = "error"
                        index_error = f"{type(e).__name__}: {e}"
                    else:
                        index_active = True
                        index_status = "active"
                        for shard in weight_map.values():
                            if Path(shard).suffix in exts:
                                expected_shards.add(shard)
                        expected_shard_paths = {
                            shard: index_path.parent / shard for shard in expected_shards
                        }
                        for p in index_path.parent.iterdir():
                            if p.is_file() and p.suffix == ".safetensors":
                                extra_safetensors_files_on_disk.add(p.name)
                        extra_safetensors_files_on_disk -= expected_shards
                # If find_fn returned a candidate path that does not exist, keep found False.

    index_parsed = bool(index_active and weight_map is not None)
    failure_ctx.update(
        {
            "index_status": index_status,
            "index_searched": bool(index_searched),
            "index_found": bool(index_found),
            "index_active": bool(index_parsed),
            "index_path": index_path,
            "index_path_found": index_path_found,
            "index_error": index_error,
        }
    )

    # Make "strict" mean "an active index must exist" when index discovery is enabled.
    if strict_index and use_index and not index_parsed:
        raise SystemExit(f"strict_index requires an active index (status: {index_status})")

    index_used_for_scan = bool(index_parsed)
    index_discovered_but_ignored_due_to_file_model_path = False
    # Treat file model_path as an explicit anchor; index metadata is for reporting only.
    if model_path_is_file and index_parsed:
        index_used_for_scan = False
        index_discovered_but_ignored_due_to_file_model_path = True
        print(
            f"[index] index found at {index_path}; "
            "but model_path is a file; scanning only the anchor file. "
            "Pass the directory to scan the indexed shard set."
        )
    failure_ctx["index_active"] = bool(index_used_for_scan)

    if mlx_enabled and schemes:
        if _load_mlx() is None:
            msg = "mlx is not importable; skipping quantization simulations"
            warnings.warn(msg)
            warn_log.append(f"[quant_sim] {msg}")
            mlx_enabled = False

    files: List[Path] = []
    missing_shards: set[str] = set()
    scanned_shards: set[str] = set()
    observed_tensor_names: set[str] = set()
    tensors_observed = 0
    path_to_shard_id: Dict[Path, str] = {}

    if index_used_for_scan:
        path_to_shard_id = {path: shard for shard, path in expected_shard_paths.items()}
        for shard in sorted(expected_shards):
            path = expected_shard_paths[shard]
            if path.exists():
                files.append(path)
            else:
                missing_shards.add(shard)
        if missing_shards:
            msg = "[index] missing shard(s) referenced by index: " + ", ".join(sorted(missing_shards))
            if strict_index:
                raise SystemExit(msg)
    else:
        files = list(_iter_weight_files(model_path, exts))
        files.sort()
    if max_files is not None:
        files = files[: int(max_files)]

    # Capture the final scan plan after index decisions + max_files are applied.
    scan_plan = {
        "use_safetensors_index_json": use_index,
        "strict_index": strict_index,
        "model_path_kind": model_path_kind,
        "index_discovered_but_ignored_due_to_file_model_path": (
            index_discovered_but_ignored_due_to_file_model_path
        ),
        "extensions": exts,
        "max_files": int(max_files) if max_files is not None else None,
        "experts_only": experts_only,
        "include_shared_expert": include_shared,
        "inventory_all_tensors": inventory_all,
        "scan_mode": "index" if index_used_for_scan else "walk",
        "scanned_files_count": int(len(files)),
        "scanned_files_example": [_path_for_scan_plan(p, model_path) for p in files[:3]],
    }

    t0 = time.time()
    print(f"[collect] scanning {len(files)} files under {model_path}")

    for fi, fpath in enumerate(files, start=1):
        if progress_every and (fi % progress_every == 0 or fi == 1):
            print(f"[collect] ({fi}/{len(files)}) {fpath}")

        if index_used_for_scan:
            shard_id = path_to_shard_id.get(fpath)
            if shard_id is not None:
                scanned_shards.add(shard_id)

        for name, arr in _iter_tensors_from_file(fpath):
            tensors_observed += 1
            if index_used_for_scan:
                observed_tensor_names.add(name)
            # inventory
            if inventory_all:
                try:
                    nbytes = int(arr.nbytes)
                except Exception:
                    nbytes = None
                row = {
                    "file": str(fpath),
                    "tensor_name": name,
                    "dtype": str(arr.dtype),
                    "shape": tuple(arr.shape),
                    "ndim": int(arr.ndim),
                    "nbytes": nbytes
                }
                if index_used_for_scan:
                    row["in_index"] = name in weight_map
                    row["index_shard"] = weight_map.get(name)
                inventory_rows.append(row)

            # only float weights for stats/sims
            if not _is_floatlike_dtype(arr.dtype):
                continue

            if experts_only and _looks_like_empty_collect_moe_candidate(name, expert_re):
                smoke_candidate_tensors_observed += 1
                record_example(example_smoke_candidates, name)

            is_shared = _is_shared_expert(name, shared_keywords)
            name_lower = name.lower()
            matches_enabled_rule = _matches_enabled_rule_candidate(name, arr, rules)
            is_expertish = (
                ("experts" in name_lower)
                or _looks_like_moe_proj_tensor(name, alias_map, expert_re)
                or is_shared
                or matches_enabled_rule
            )
            tensor_key = (str(fpath), name)

            if experts_only:
                if not is_expertish:
                    continue
                if is_shared and not include_shared:
                    continue

            # try explicit rules, else fallback heuristics
            extracted = None
            extracted_via: Optional[str] = None
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
                    proj_group_strict,
                    proj_issue_acc=proj_issue_acc,
                    unmatched_reason_override=unmatched_reason_override,
                    skip_fallback=skip_fallback,
                )
                if extracted is not None:
                    extracted_via = "rule"
            except PackedSplitError as e:
                if strict_packed_split:
                    raise
                warn_log.append(f"[extract] {e}")
            except Exception as e:
                warn_log.append(f"[extract] rule application failed for {name}: {e}")

            if extracted is None:
                if tensor_key not in skip_fallback:
                    try:
                        extracted = _fallback_extract(name, arr, fpath, layer_re, expert_re, alias_map, shared_keywords)
                        if extracted is not None:
                            extracted_via = "fallback"
                    except Exception as e:
                        warn_log.append(f"[extract] fallback failed for {name}: {e}")
                        extracted = None

            if extracted is None:
                if experts_only and is_expertish:
                    unmatched_expertish += 1
                    record_example(example_unmatched_expertish, name)
                reason = unmatched_reason_override.get(
                    tensor_key,
                    "no_rule_match_or_proj_infer",
                )

                if dump_unmatched and experts_only and is_expertish:
                    unmatched_rows.append({
                        "file": str(fpath),
                        "tensor_name": name,
                        "dtype": str(arr.dtype),
                        "shape": tuple(arr.shape),
                        "ndim": int(arr.ndim),
                        "reason": reason,
                    })
                continue

            if extracted_via == "rule":
                extracted_by_rule += 1
                record_example(example_rule_extracted, name)
            elif extracted_via == "fallback":
                extracted_by_fallback += 1
                record_example(example_fallback_extracted, name)

            process_extracted_banks(
                extracted=extracted,
                cfg_stats=cfg_stats,
                cache_idx_dir=cache_idx_dir,
                matrix_rows=matrix_rows,
                quant_rows=quant_rows,
                mlx_enabled=mlx_enabled,
                schemes=schemes,
                mlx_device=mlx_device,
                per_expert_weight_stats=_per_expert_weight_stats,
                mlx_quant_sim=_mlx_quant_sim,
                warn_log=warn_log,
            )

    missing_shards_report: List[str] = []
    extra_scanned_shards: List[str] = []
    missing_tensors: List[str] = []
    extra_tensors: List[str] = []
    extra_on_disk: List[str] = []

    if index_used_for_scan and weight_map is not None:
        report = build_index_report_data(
            expected_shards=expected_shards,
            scanned_shards=scanned_shards,
            weight_map=weight_map,
            observed_tensor_names=observed_tensor_names,
            extra_safetensors_files_on_disk=extra_safetensors_files_on_disk,
            index_metadata=index_metadata,
        )
        missing_shards_report = list(report.get("missing_shards", []))
        extra_scanned_shards = list(report.get("extra_scanned_shards", []))
        missing_tensors = list(report.get("missing_tensors", []))
        extra_tensors = list(report.get("extra_tensors", []))
        extra_on_disk = list(report.get("extra_safetensors_files_on_disk", []))

        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # CONTRACT SURFACE: logs/index_report.json
        # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
        # Tests: rg 'index_report.json' tests/
        (logs_dir / "index_report.json").write_text(json.dumps(report, indent=2))

        if missing_shards_report:
            warn_log.append("[index] missing shards: " + ", ".join(missing_shards_report))
        if extra_scanned_shards:
            warn_log.append("[index] extra scanned shards: " + ", ".join(extra_scanned_shards))
        if missing_tensors:
            warn_log.append("[index] missing tensors: " + ", ".join(missing_tensors))
        if extra_tensors:
            warn_log.append("[index] extra tensors: " + ", ".join(extra_tensors))
        if extra_on_disk:
            warn_log.append("[index] extra safetensors files on disk: " + ", ".join(extra_on_disk))

    # Always write run-health report, even when index and metadata are disabled.
    #
    # This is deliberately JSON (not parquet/CSV) so it's quick to inspect and easy
    # to parse from tools without knowing the configured output format.
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
    proj_df = pd.DataFrame(list(proj_issue_acc.values())) if proj_issue_acc else pd.DataFrame()

    # CONTRACT SURFACE: write_manifest.artifacts stable key map
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'write_manifest.json' tests/
    artifacts: Dict[str, Any] = {}
    artifacts["tensor_inventory"] = _artifact_entry(
        _write_df(inv_df, run_dir / "data" / "tensor_inventory.parquet", fmt, compression),
        run_dir,
        len(inv_df),
    )
    artifacts["matrix_stats"] = _artifact_entry(
        _write_df(ms_df, run_dir / "data" / "matrix_stats.parquet", fmt, compression),
        run_dir,
        len(ms_df),
    )
    artifacts["quant_sim"] = _artifact_entry(
        _write_df(qs_df, run_dir / "data" / "quant_sim.parquet", fmt, compression),
        run_dir,
        len(qs_df),
    )
    if dump_unmatched:
        artifacts["unmatched_tensors"] = _artifact_entry(
            _write_df(um_df, run_dir / "data" / "unmatched_tensors.parquet", fmt, compression),
            run_dir,
            len(um_df),
        )
    if not proj_df.empty:
        # Keep detailed canonicalization diagnostics in a dedicated report artifact.
        report_meta = _artifact_entry(
            _write_df(
                proj_df,
                run_dir / "logs" / "proj_canonicalization_report.parquet",
                fmt,
                compression,
            ),
            run_dir,
            len(proj_df),
        )
        artifacts["proj_canonicalization_report"] = report_meta

        kept_raw = proj_df[proj_df["action"] == "kept_raw"] if "action" in proj_df.columns else pd.DataFrame()
        if not kept_raw.empty:
            packed_split_occurrences = int(
                kept_raw.loc[kept_raw["context"] == "packed_split", "count"].sum()
            )
            proj_group_occurrences = int(
                kept_raw.loc[kept_raw["context"] == "proj_group", "count"].sum()
            )
            unique_raw = int(kept_raw["raw_proj"].nunique())
            total_occurrences = packed_split_occurrences + proj_group_occurrences
            warn_log.append(
                "[proj] unmapped proj tokens kept raw: "
                f"packed_split={packed_split_occurrences}, "
                f"proj_group={proj_group_occurrences} "
                f"(unique={unique_raw}, occurrences={total_occurrences}). "
                f"See {report_meta['path']}"
            )

        dropped_strict = proj_df[
            (proj_df["action"] == "dropped_strict") & (proj_df["context"] == "proj_group")
        ] if "action" in proj_df.columns and "context" in proj_df.columns else pd.DataFrame()
        if not dropped_strict.empty:
            dropped_occurrences = int(dropped_strict["count"].sum())
            dropped_unique = int(dropped_strict["raw_proj"].nunique())
            warn_log.append(
                "[proj] strict proj_group dropped tensors due to unmapped proj tokens: "
                f"occurrences={dropped_occurrences} "
                f"(unique={dropped_unique}). "
                f"See {report_meta['path']}"
            )

    strict_no_alias_occurrences = sum(
        1
        for reason in unmatched_reason_override.values()
        if reason == "proj_group_strict_no_alias_map"
    )
    if strict_no_alias_occurrences > 0:
        if "unmatched_tensors" in artifacts:
            unmatched_meta = artifacts.get("unmatched_tensors", {})
            unmatched_path = str(unmatched_meta.get("path") or "unmatched_tensors.*")
            warn_log.append(
                "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty; "
                f"strict proj_group drops occurred (occurrences={strict_no_alias_occurrences}). "
                f"See {unmatched_path} for details."
            )
        elif not dump_unmatched:
            warn_log.append(
                "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty; "
                f"strict proj_group drops occurred (occurrences={strict_no_alias_occurrences}). "
                "Enable debug.dump_unmatched_tensors=true to write unmatched_tensors.*."
            )
        else:
            warn_log.append(
                "[config] parsing.proj_group_strict=true but parsing.proj_aliases is empty; "
                f"strict proj_group drops occurred (occurrences={strict_no_alias_occurrences}). "
                "unmatched_tensors.* was not written in this run (this can happen if nothing was eligible to dump); "
                "check logs/write_manifest.json."
            )

    effectively_empty_collect_triggered = bool(
        experts_only
        and smoke_candidate_tensors_observed > 0
        and len(matrix_rows) == 0
        and unmatched_expertish == 0
    )
    smoke_checks = {
        "effectively_empty_collect": {
            "triggered": effectively_empty_collect_triggered,
            "candidate_tensors_observed": int(smoke_candidate_tensors_observed),
            "candidate_examples": list(example_smoke_candidates),
        }
    }
    if effectively_empty_collect_triggered:
        examples_text = ", ".join(example_smoke_candidates[:3])
        warn_log.append(
            "[smoke] effectively empty collect: experts_only=true observed plausible "
            f"MoE expert candidates={smoke_candidate_tensors_observed} but wrote "
            "zero matrix_stats rows and recorded zero unmatched expertish tensors. "
            f"Examples: {examples_text}. "
            "Check parsing.expert_regex, extract_rules, and projection naming."
        )

    wl_df = pd.DataFrame({"warning": warn_log}) if warn_log else pd.DataFrame()
    if not wl_df.empty:
        # CONTRACT SURFACE: logs/warnings.{parquet|csv} + write_manifest.artifacts["warnings"]
        # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
        # Tests: rg 'warnings.parquet' tests/
        artifacts["warnings"] = _artifact_entry(
            _write_df(wl_df, run_dir / "logs" / "warnings.parquet", fmt, compression),
            run_dir,
            len(wl_df),
        )

    write_manifest = {
        "generated_at": _utc_now_iso(),
        "requested_format": fmt,
        "requested_compression": compression,
        "artifacts": artifacts,
    }
    # CONTRACT SURFACE: logs/write_manifest.json
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'write_manifest.json' tests/
    _write_json(write_manifest, run_dir / "logs" / "write_manifest.json")

    dt = time.time() - t0

    outputs_written = {
        "format": fmt,
        "tensor_inventory_rows": int(len(inv_df)),
        "matrix_stats_rows": int(len(ms_df)),
        "quant_sim_rows": int(len(qs_df)),
        "unmatched_tensors_rows": int(len(um_df)) if dump_unmatched else 0,
        "warnings_rows": int(len(wl_df)),
        "wrote_unmatched_tensors": bool(dump_unmatched),
        "wrote_warnings": bool(not wl_df.empty),
        "wrote_index_report": bool(index_used_for_scan),
    }

    index_summary: Dict[str, Any] = build_index_summary(
        index_used_for_scan=index_used_for_scan,
        index_parsed=index_parsed,
        index_path=index_path,
        strict_index=strict_index,
        expected_shards=expected_shards,
        scanned_shards=scanned_shards,
        missing_shards_report=missing_shards_report,
        extra_scanned_shards=extra_scanned_shards,
        missing_tensors=missing_tensors,
        extra_tensors=extra_tensors,
        extra_on_disk=extra_on_disk,
        index_metadata=index_metadata,
    )

    run_health = {
        "status": "success",
        "generated_at": _utc_now_iso(),
        "duration_seconds": float(dt),
        "run": {
            "run_dir": str(run_dir),
            "model_id": manifest.get("model_id"),
            "run_name": manifest.get("run_name"),
            "created_at": manifest.get("created_at"),
            "model_path": str(model_path),
        },
        "config_used": cfg,
        "scan_summary": {
            "files_scanned": int(len(files)),
            "tensors_observed": int(tensors_observed),
        },
        "extraction_summary": {
            "extracted_by_rule": int(extracted_by_rule),
            "extracted_by_fallback": int(extracted_by_fallback),
            "unmatched_expertish": int(unmatched_expertish),
        },
        "outputs_written": outputs_written,
        "tensor_name_formats": {
            "layer_regex": parsing.get("layer_regex"),
            "expert_regex": parsing.get("expert_regex"),
            "enabled_extract_rules": [
                {"name": r.name, "match": r.regex.pattern, "ndim": r.ndim}
                for r in rules
                if r.enabled
            ],
            "proj_aliases": alias_map,
        },
        "tensor_name_examples": {
            "rule_extracted": example_rule_extracted,
            "fallback_extracted": example_fallback_extracted,
            "unmatched_expertish": example_unmatched_expertish,
        },
        "smoke_checks": smoke_checks,
        "derived_tensor_formats": [
            "<raw_tensor_name>::<proj>",
            "<raw_tensor_name>::split[rows]::<proj>",
            "<raw_tensor_name>::split[cols]::<proj>",
        ],
        "index_summary": index_summary,
    }
    # CONTRACT SURFACE: logs/run_health.json
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'run_health.json' tests/
    _write_json(run_health, run_dir / "logs" / "run_health.json")

    # Persist a durable context snapshot so "haunted runs" can be reconstructed.
    index_context_path = None
    if index_path is not None and index_path.exists():
        index_context_path = index_path
    elif index_path_found is not None and index_path_found.exists():
        index_context_path = index_path_found
    index_info = {
        "status": index_status,
        "searched": bool(index_searched),
        "found": bool(index_found),
        "parsed": bool(index_parsed),
        "active": bool(index_used_for_scan),
        "used_for_scan": bool(index_used_for_scan),
        "index_path": str(index_context_path) if index_context_path is not None else None,
    }
    if index_error:
        index_info["error"] = index_error

    configured_model_path_resolved = None
    if configured_model_path:
        configured_model_path_resolved = str(Path(configured_model_path).expanduser().resolve())

    run_context = {
        "generated_at": _utc_now_iso(),
        "run": {
            "run_dir": str(run_dir),
            "model_id": manifest.get("model_id"),
            "run_name": manifest.get("run_name"),
            "created_at": manifest.get("created_at"),
        },
        "model_path": {
            "configured": configured_model_path_resolved,
            "resolved": str(model_path),
            "source": "cli_override" if "model_path" in cli_overrides else "config",
        },
        "cli_overrides": cli_overrides,
        "scan_plan": scan_plan,
        "index": index_info,
    }
    # CONTRACT SURFACE: logs/run_context.json
    # Prefer additive changes; don't rename/remove without explicit request. See README: Run outputs / Auditability artifacts.
    # Tests: rg 'run_context.json' tests/
    _write_json(run_context, run_dir / "logs" / "run_context.json")
    _clear_run_failure_artifact(run_dir)
    print(f"[collect] done in {dt:.1f}s")
    print(f"[collect] tensor_inventory rows: {len(inv_df)}")
    print(f"[collect] matrix_stats rows:     {len(ms_df)}")
    print(f"[collect] quant_sim rows:        {len(qs_df)}")
    if dump_unmatched:
        print(f"[collect] unmatched rows:        {len(um_df)}")
    if not wl_df.empty:
        print(f"[collect] warnings:              {len(wl_df)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-path", default=None, help="Override config model_path")
    args = ap.parse_args()

    failure_ctx: Dict[str, Any] = {
        "run_dir": Path(args.run_dir).expanduser().resolve(),
        "manifest": {},
        "configured_model_path": None,
        "model_path": None,
        "cli_overrides": {},
        "index_status": "not_initialized",
        "index_searched": False,
        "index_found": False,
        "index_active": False,
        "index_path": None,
        "index_path_found": None,
        "index_error": None,
    }

    try:
        _main_impl(args, failure_ctx)
    except SystemExit as exc:
        if _system_exit_is_error(exc):
            _write_run_failure_artifact(exc, failure_ctx)
        raise
    except Exception as exc:
        _write_run_failure_artifact(exc, failure_ctx)
        raise


if __name__ == "__main__":
    main()
