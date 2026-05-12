"""
Generate Dynamo env-only JSON by scanning backend/common Dynamo source.

This complements ``gen_dynamo_wrapper_args.py``. Wrapper ArgGroups capture
env-backed CLI arguments; this scanner captures Dynamo env vars read directly
from source without a CLI flag.

Usage:
    python dynamo/gen_dynamo_envs.py --engine vllm --out dynamo_envs.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


TARGET_PREFIXES = ("DYN_", "DYNAMO_")
META_KEYS = {"engine", "version", "date", "source"}

_GETENV_RE = re.compile(
    r'os\.(?:getenv|environ\.get)\s*\(\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*'
    r'(?:,\s*([^)\n]+?))?\s*\)',
    re.MULTILINE,
)
_ENVIRON_ITEM_RE = re.compile(
    r'os\.environ\s*\[\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*\]',
)
_STRING_ENV_RE = re.compile(r'["\']((?:DYN|DYNAMO)_[A-Z0-9_]+)["\']')


def _parse_default(raw: str | None) -> Any:
    if not raw:
        return None
    value = raw.strip().rstrip(",").strip()
    if value in {"None", ""}:
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    for quote in ('"', "'"):
        if value.startswith(quote) and value.endswith(quote) and len(value) >= 2:
            return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return None


def _type_from_default(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, (list, tuple)):
        return "List[str]"
    return "str"


def _preceding_comment(source: str, match_start: int) -> str:
    line_start = source.rfind("\n", 0, match_start) + 1
    comments: list[str] = []
    pos = line_start - 1
    while pos > 0:
        prev_end = pos
        prev_start = source.rfind("\n", 0, prev_end) + 1
        prev = source[prev_start:prev_end].strip()
        if prev.startswith("#"):
            text = prev.lstrip("# ").strip()
            if text and not re.match(r"^=+$|^-+$", text):
                comments.insert(0, text)
            pos = prev_start - 1
        else:
            break
    return " ".join(comments)


def _inline_comment(source: str, match_start: int) -> str:
    line_end = source.find("\n", match_start)
    if line_end == -1:
        line_end = len(source)
    line = source[match_start:line_end]
    if "#" in line:
        return line[line.index("#") + 1:].strip()
    return ""


def _engine_module(engine: str) -> str:
    if engine == "tensorrt_llm":
        return "dynamo.trtllm"
    return f"dynamo.{engine}"


def _source_roots(engine: str) -> list[Path]:
    import dynamo.common as common

    roots = [Path(common.__file__).parent]
    module = importlib.import_module(_engine_module(engine))
    roots.append(Path(module.__file__).parent)
    return roots


def _skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if "tests" in parts or "__pycache__" in parts:
        return True
    # These config ArgGroups are captured as wrapper args. Scanning the whole
    # directory would also pull in frontend/router-only envs for worker images.
    return "configuration" in parts


def _category(root: Path, file_path: Path) -> str:
    rel_parts = file_path.relative_to(root).parts
    if root.name == "common":
        if rel_parts:
            return f"dynamo_common_{rel_parts[0].replace('.py', '')}"
        return "dynamo_common"
    return f"dynamo_{root.name}"


def _is_target(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TARGET_PREFIXES)


def _record(
    found: dict[str, dict[str, Any]],
    name: str,
    *,
    default: Any,
    description: str,
    category: str,
    source_file: str,
) -> None:
    if not _is_target(name):
        return
    if name not in found:
        found[name] = {
            "category": category,
            "type": _type_from_default(default),
            "default_value": default,
            "value": None,
            "description": description,
            "source_file": source_file,
            "ui": "less_frequent",
            "aic": False,
        }
        return

    existing = found[name]
    if existing["default_value"] is None and default is not None:
        existing["default_value"] = default
        existing["type"] = _type_from_default(default)
    if not existing["description"] and description:
        existing["description"] = description


def build_json(engine: str) -> dict[str, Any]:
    found: dict[str, dict[str, Any]] = {}

    for root in _source_roots(engine):
        for py_file in sorted(root.rglob("*.py")):
            if _skip_path(py_file):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not any(prefix in source for prefix in TARGET_PREFIXES):
                continue

            rel = str(py_file.relative_to(root))
            category = _category(root, py_file)

            for match in _GETENV_RE.finditer(source):
                default = _parse_default(match.group(2))
                description = (
                    _preceding_comment(source, match.start())
                    or _inline_comment(source, match.start())
                )
                _record(
                    found,
                    match.group(1),
                    default=default,
                    description=description,
                    category=category,
                    source_file=rel,
                )

            for match in _ENVIRON_ITEM_RE.finditer(source):
                _record(
                    found,
                    match.group(1),
                    default=None,
                    description=_preceding_comment(source, match.start()),
                    category=category,
                    source_file=rel,
                )

            for match in _STRING_ENV_RE.finditer(source):
                _record(
                    found,
                    match.group(1),
                    default=None,
                    description=_preceding_comment(source, match.start()),
                    category=category,
                    source_file=rel,
                )

    return {
        "engine": engine,
        "version": "dynamo-env-scan",
        "date": date.today().isoformat(),
        "source": f"source-scanned from Dynamo common + {_engine_module(engine)}",
        **dict(sorted(found.items())),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--engine", choices=["vllm", "sglang", "tensorrt_llm"], required=True)
    cli.add_argument("--out", type=Path, required=True)
    ns = cli.parse_args()

    data = build_json(ns.engine)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(data, indent=2) + "\n")
    count = len([k for k in data if k not in META_KEYS])
    print(f"Written {ns.out!s} — {ns.engine} Dynamo env scan — {count} vars")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        sys.exit(f"Failed to import Dynamo modules: {exc}")
