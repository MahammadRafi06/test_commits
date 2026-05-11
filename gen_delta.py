"""
Generate a delta JSON comparing two versions of an engine JSON
(args or envs) produced by any gen_*_args.py / gen_*_envs.py script.

Usage:
    python gen_delta.py --old OLD.json --new NEW.json [--out DELTA.json]

What is reported:
  added     — keys present in NEW but not in OLD
  removed   — keys present in OLD but not in NEW
  changed   — keys present in both but with differing field values
              (ui / aic / value are excluded from comparison — they are
               user-managed metadata, not engine-driven)
  unchanged — count only (not written to keep the file small)
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

_META_KEYS: frozenset[str] = frozenset({"engine", "version", "date", "source"})
_SKIP_FIELDS: frozenset[str] = frozenset({"ui", "aic", "value"})


def compute_delta(old_data: dict, new_data: dict) -> dict:
    old_entries = {k: v for k, v in old_data.items() if k not in _META_KEYS}
    new_entries = {k: v for k, v in new_data.items() if k not in _META_KEYS}

    old_keys = set(old_entries)
    new_keys = set(new_entries)

    added   = {k: new_entries[k] for k in sorted(new_keys - old_keys)}
    removed = {k: old_entries[k] for k in sorted(old_keys - new_keys)}

    changed: dict[str, Any] = {}
    unchanged = 0
    for k in sorted(old_keys & new_keys):
        diff: dict[str, Any] = {}
        for f in set(old_entries[k]) | set(new_entries[k]):
            if f in _SKIP_FIELDS:
                continue
            ov = old_entries[k].get(f)
            nv = new_entries[k].get(f)
            if ov != nv:
                diff[f] = {"old": ov, "new": nv}
        if diff:
            changed[k] = {"diff": diff}
        else:
            unchanged += 1

    return {
        "engine":      new_data.get("engine", "unknown"),
        "old_version": old_data.get("version", "unknown"),
        "new_version": new_data.get("version", "unknown"),
        "date": date.today().isoformat(),
        "summary": {
            "total_old": len(old_entries),
            "total_new": len(new_entries),
            "added":     len(added),
            "removed":   len(removed),
            "changed":   len(changed),
            "unchanged": unchanged,
        },
        "added":   added,
        "removed": removed,
        "changed": changed,
    }


def _print_delta(delta: dict, out_path: str) -> None:
    s = delta["summary"]
    print(
        f"Written {out_path!r}  —  "
        f"{delta['engine']} {delta['old_version']} → {delta['new_version']}"
    )
    print(
        f"  added: {s['added']}  removed: {s['removed']}  "
        f"changed: {s['changed']}  unchanged: {s['unchanged']}"
    )
    for label, section in (("Added", "added"), ("Removed", "removed"), ("Changed", "changed")):
        entries = delta[section]
        if not entries:
            continue
        keys = list(entries)
        print(f"\n  {label} ({len(keys)}):")
        for k in keys[:10]:
            if section == "changed":
                fields = ", ".join(entries[k]["diff"])
                print(f"    {'~'} {k}  [{fields}]")
            else:
                marker = "+" if section == "added" else "-"
                print(f"    {marker} {k}")
        if len(keys) > 10:
            print(f"    ... and {len(keys) - 10} more")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--old", required=True, help="Path to old version JSON")
    cli.add_argument("--new", required=True, help="Path to new version JSON")
    cli.add_argument("--out", help="Output delta JSON path (auto-named if omitted)")
    ns = cli.parse_args()

    with open(ns.old) as f:
        old_data = json.load(f)
    with open(ns.new) as f:
        new_data = json.load(f)

    delta = compute_delta(old_data, new_data)

    out_path = ns.out or (
        f"delta__{Path(ns.old).stem}__{Path(ns.new).stem}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(delta, fh, indent=2)

    _print_delta(delta, out_path)
