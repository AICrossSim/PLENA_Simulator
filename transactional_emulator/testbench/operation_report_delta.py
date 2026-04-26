#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


LIST_FIELDS = (
    "vram_value_tiles",
    "mram_value_tiles",
    "hbm_value_tiles",
    "fpram_fp_fragments",
)

MAP_FIELDS = (
    "value_tile_slice_refs",
    "scatter_group_backing_slice_refs",
    "fp_fragment_value_refs",
)

MAP_FIELD_ALIASES = {
    "value_tile_tensor_refs": "value_tile_slice_refs",
    "scatter_group_backing_tile_refs": "scatter_group_backing_slice_refs",
}


def _split_top_level_csv(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    bracket_depth = 0
    paren_depth = 0

    for char in text:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1

        if char == "," and bracket_depth == 0 and paren_depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_csv(raw: str) -> List[str]:
    text = raw.strip()
    if not text or text == "(empty)":
        return []
    return _split_top_level_csv(text)


def parse_operation_report(text: str) -> List[Dict[str, object]]:
    snapshots: List[Dict[str, object]] = []
    current: Dict[str, object] | None = None
    current_map: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            current_map = None
            continue

        if line.startswith("op["):
            current_map = None
            index_end = line.find("]:")
            if index_end < 0:
                continue
            op_index = int(line[3:index_end])
            label = line[index_end + 2 :].strip()
            current = {
                "index": op_index,
                "label": label,
            }
            for field in LIST_FIELDS:
                current[field] = []
            for field in MAP_FIELDS:
                current[field] = {}
            snapshots.append(current)
            continue

        if current is None:
            continue

        stripped = line.strip()
        if stripped.endswith(":"):
            map_name = MAP_FIELD_ALIASES.get(stripped[:-1], stripped[:-1])
            if map_name in MAP_FIELDS:
                current_map = map_name
                continue
            current_map = None
            continue

        if current_map and line.startswith("    "):
            entry = stripped
            if entry == "(empty)":
                continue
            key, _, values = entry.partition(":")
            current[current_map][key.strip()] = _parse_csv(values)  # type: ignore[index]
            continue

        current_map = None
        for field in LIST_FIELDS:
            prefix = f"{field}:"
            if stripped.startswith(prefix):
                current[field] = _parse_csv(stripped[len(prefix) :])
                break

    return snapshots


def _list_delta(prev_items: List[str], curr_items: List[str]) -> List[str]:
    prev_set = set(prev_items)
    curr_set = set(curr_items)
    added = sorted(curr_set - prev_set)
    removed = sorted(prev_set - curr_set)
    lines: List[str] = []
    if added:
        lines.append(f"    + {', '.join(added)}")
    if removed:
        lines.append(f"    - {', '.join(removed)}")
    return lines


def _map_delta(prev_map: Dict[str, List[str]], curr_map: Dict[str, List[str]]) -> List[str]:
    lines: List[str] = []
    all_keys = sorted(set(prev_map) | set(curr_map))
    for key in all_keys:
        prev_values = set(prev_map.get(key, []))
        curr_values = set(curr_map.get(key, []))
        added = sorted(curr_values - prev_values)
        removed = sorted(prev_values - curr_values)
        if not added and not removed:
            continue
        lines.append(f"    {key}:")
        if added:
            lines.append(f"      + {', '.join(added)}")
        if removed:
            lines.append(f"      - {', '.join(removed)}")
    return lines


def _value_tile_target_labels(snapshot: Dict[str, object], value_tile_id: str) -> List[str]:
    labels: set[str] = set()
    for field in ("value_tile_slice_refs", "scatter_group_backing_slice_refs"):
        mapping = snapshot.get(field, {})
        if isinstance(mapping, dict):
            labels.update(mapping.get(value_tile_id, []))
    return sorted(labels)


def build_delta_report(snapshots: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    previous: Dict[str, object] | None = None

    for snapshot in snapshots:
        header = f"op[{snapshot['index']}]: {snapshot['label']}"
        lines.append(header)

        if previous is None:
            lines.append("  initial_state:")
            for field in LIST_FIELDS:
                items = snapshot[field]  # type: ignore[index]
                if items:
                    lines.append(f"    {field}: {', '.join(items)}")
            for field in MAP_FIELDS:
                mapping = snapshot[field]  # type: ignore[index]
                if mapping:
                    lines.append(f"    {field}:")
                    for key in sorted(mapping):
                        values = mapping[key]
                        if values:
                            lines.append(f"      {key}: {', '.join(values)}")
            if len(lines) > 0 and lines[-1] == header:
                lines.append("  no_state")
            lines.append("")
            previous = snapshot
            continue

        changed = False
        changed_value_tiles: set[str] = set()
        for field in LIST_FIELDS:
            prev_items = previous[field]  # type: ignore[index]
            curr_items = snapshot[field]  # type: ignore[index]
            delta_lines = _list_delta(
                prev_items,
                curr_items,
            )
            if delta_lines:
                changed = True
                lines.append(f"  {field}:")
                lines.extend(delta_lines)
                changed_value_tiles.update(set(prev_items) ^ set(curr_items))

        for field in MAP_FIELDS:
            prev_map = previous[field]  # type: ignore[index]
            curr_map = snapshot[field]  # type: ignore[index]
            delta_lines = _map_delta(
                prev_map,
                curr_map,
            )
            if delta_lines:
                changed = True
                lines.append(f"  {field}:")
                lines.extend(delta_lines)
                changed_value_tiles.update(set(prev_map) ^ set(curr_map))
                for key in set(prev_map) & set(curr_map):
                    if set(prev_map.get(key, [])) != set(curr_map.get(key, [])):
                        changed_value_tiles.add(key)

        value_tile_like = sorted(tile_id for tile_id in changed_value_tiles if tile_id.startswith("value_tile."))
        if value_tile_like:
            lines.append("  changed_value_tile_targets:")
            for value_tile_id in value_tile_like:
                previous_targets = _value_tile_target_labels(previous, value_tile_id)
                current_targets = _value_tile_target_labels(snapshot, value_tile_id)
                if current_targets:
                    lines.append(f"    {value_tile_id}: {', '.join(current_targets)}")
                elif previous_targets:
                    lines.append(f"    {value_tile_id}: (released) | prev={', '.join(previous_targets)}")
                else:
                    lines.append(f"    {value_tile_id}: (no slice refs)")

        if not changed:
            lines.append("  no_change")
        lines.append("")
        previous = snapshot

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an operation report into a step-by-step delta-only report."
    )
    parser.add_argument("input", type=Path, help="Path to the original operation report.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Optional output path. Defaults to <input_stem>_delta.txt",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem}_delta.txt")

    snapshots = parse_operation_report(input_path.read_text(encoding="utf-8"))
    delta_report = build_delta_report(snapshots)
    output_path.write_text(delta_report, encoding="utf-8")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
