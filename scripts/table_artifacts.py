from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

_TABLE_FILE_SUFFIXES = {".csv", ".parquet"}


DEFAULT_TABLE_ARTIFACT_KEYS = [
    "A_weight_layer_summary",
    "A_weight_block4_summary",
    "A_weight_global_summary",
    "B_quant_layer_summary",
    "B_quant_block4_summary",
    "B_quant_global_summary",
    "B_quant_deltas",
]


def _safe_read_json_dict(path: Path) -> Dict[str, Any]:
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_existing_table_path(abs_path: Path, run_dir: Path) -> str | None:
    # Validate containment using resolved paths so symlink targets cannot escape run_dir.
    try:
        rel_path = abs_path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return None

    rel_path_obj = Path(rel_path)
    rel_path_posix = rel_path_obj.as_posix()
    if rel_path_obj.is_absolute() or rel_path_posix in {"", "."}:
        return None
    if not rel_path_posix.startswith("tables/"):
        return None
    if rel_path_obj.suffix.lower() not in _TABLE_FILE_SUFFIXES:
        return None
    if not abs_path.is_file():
        return None

    return rel_path_posix


def _normalize_manifest_entry(entry: Dict[str, Any], run_dir: Path) -> Dict[str, Any] | None:
    raw_path = entry.get("path")
    if raw_path is None:
        return None
    path_text = str(raw_path).strip()
    if not path_text:
        return None

    path_obj = Path(path_text)
    abs_path = path_obj if path_obj.is_absolute() else (run_dir / path_obj)
    rel_path_posix = _normalize_existing_table_path(abs_path, run_dir)
    if rel_path_posix is None:
        return None

    return {
        "path": rel_path_posix,
        "format": str(entry.get("format", "")),
        "fallback": bool(entry.get("fallback", False)),
        "error": str(entry.get("error", "")),
        "rows": _safe_int(entry.get("rows", 0), default=0),
        "source": "manifest",
    }


def _legacy_scan_entry(run_dir: Path, artifact_key: str) -> Dict[str, Any] | None:
    parquet_path = run_dir / "tables" / f"{artifact_key}.parquet"
    csv_path = run_dir / "tables" / f"{artifact_key}.csv"

    parquet_rel = _normalize_existing_table_path(parquet_path, run_dir)
    if parquet_rel is not None:
        return {
            "path": parquet_rel,
            "format": "parquet",
            "fallback": False,
            "error": "",
            "rows": 0,
            "source": "legacy_scan",
        }

    csv_rel = _normalize_existing_table_path(csv_path, run_dir)
    if csv_rel is not None:
        return {
            "path": csv_rel,
            "format": "csv",
            "fallback": False,
            "error": "",
            "rows": 0,
            "source": "legacy_scan",
        }
    return None


def discover_table_artifacts(
    run_dir: Path,
    artifact_keys: Iterable[str] = DEFAULT_TABLE_ARTIFACT_KEYS,
) -> Dict[str, Dict[str, Any]]:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = _safe_read_json_dict(run_dir / "logs" / "tables_write_manifest.json")
    manifest_artifacts = manifest.get("artifacts", {})
    if not isinstance(manifest_artifacts, dict):
        manifest_artifacts = {}

    discovered: Dict[str, Dict[str, Any]] = {}
    for artifact_key in artifact_keys:
        manifest_entry = manifest_artifacts.get(artifact_key)
        if isinstance(manifest_entry, dict):
            normalized = _normalize_manifest_entry(manifest_entry, run_dir)
            if normalized is not None:
                discovered[artifact_key] = normalized
                continue

        legacy_entry = _legacy_scan_entry(run_dir, artifact_key)
        if legacy_entry is not None:
            discovered[artifact_key] = legacy_entry

    return discovered
