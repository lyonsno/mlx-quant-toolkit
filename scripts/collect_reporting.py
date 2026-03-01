from __future__ import annotations

from typing import Any, Dict


def build_index_report_data(
    expected_shards,
    scanned_shards,
    weight_map,
    observed_tensor_names,
    extra_safetensors_files_on_disk,
    index_metadata,
):
    expected_set = set(expected_shards)
    scanned_set = set(scanned_shards)
    map_keys = set(weight_map.keys()) if weight_map is not None else set()
    observed_set = set(observed_tensor_names)
    extra_on_disk_set = set(extra_safetensors_files_on_disk)

    report: Dict[str, Any] = {
        "expected_shards": sorted(expected_set),
        "scanned_shards": sorted(scanned_set),
        "missing_shards": sorted(expected_set - scanned_set),
        "extra_scanned_shards": sorted(scanned_set - expected_set),
        "missing_tensors": sorted(map_keys - observed_set),
        "extra_tensors": sorted(observed_set - map_keys),
        "extra_safetensors_files_on_disk": sorted(extra_on_disk_set),
    }
    if index_metadata:
        report["index_metadata"] = index_metadata
    return report


def build_index_summary(
    index_used_for_scan,
    index_parsed,
    index_path,
    strict_index,
    expected_shards,
    scanned_shards,
    missing_shards_report,
    extra_scanned_shards,
    missing_tensors,
    extra_tensors,
    extra_on_disk,
    index_metadata,
):
    index_active = bool(index_used_for_scan)
    summary: Dict[str, Any] = {
        "active": index_active,
        "parsed": bool(index_parsed),
        "used_for_scan": index_active,
        "index_path": str(index_path) if index_path is not None else None,
        "strict_index": bool(strict_index),
        "expected_shards_count": int(len(expected_shards)) if index_active else 0,
        "scanned_shards_count": int(len(scanned_shards)) if index_active else 0,
        "missing_shards_count": int(len(missing_shards_report)) if index_active else 0,
        "extra_scanned_shards_count": int(len(extra_scanned_shards)) if index_active else 0,
        "missing_tensors_count": int(len(missing_tensors)) if index_active else 0,
        "extra_tensors_count": int(len(extra_tensors)) if index_active else 0,
        "extra_safetensors_files_on_disk_count": int(len(extra_on_disk)) if index_active else 0,
    }
    if index_metadata:
        summary["index_metadata"] = index_metadata
    return summary
