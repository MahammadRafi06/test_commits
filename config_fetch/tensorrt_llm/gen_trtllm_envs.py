"""
Generate trtllm_envs.json by scanning the tensorrt_llm package source for
environment variable usages (os.getenv / os.environ.get patterns) with
TRTLLM_* or TLLM_* prefixes, plus StrEnum classes that enumerate env var keys.

TensorRT-LLM has no single centralised env-vars file (unlike vLLM's envs.py or
SGLang's environ.py), so this script uses a best-effort source scan.

Usage:
    python gen_trtllm_envs.py [--out trtllm_envs.json]

Requirements:
    Run inside a Python environment that has TensorRT-LLM installed.
    No GPU required.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Import tensorrt_llm
# ---------------------------------------------------------------------------
try:
    import tensorrt_llm as _trtllm
    TRTLLM_VERSION: str = _trtllm.__version__
except ImportError:
    sys.exit("tensorrt_llm is not installed in this environment.")

_PKG_DIR = Path(_trtllm.__file__).parent

# Only collect vars with these prefixes (filters out CUDA_*, NCCL_*, etc.)
_TARGET_PREFIXES = ("TRTLLM_", "TLLM_")


# ---------------------------------------------------------------------------
# Category assignment by file path
# ---------------------------------------------------------------------------
_PATH_CATEGORIES: list[tuple[str, str]] = [
    # (directory segment, category key) — first match wins
    ("commands",      "serve"),
    ("executor",      "executor"),
    ("llmapi",        "llmapi"),
    ("_torch",        "pytorch_backend"),
    ("bindings",      "runtime"),
    ("serve",         "server"),
    ("builder",       "builder"),
    ("quantization",  "quantization"),
    ("models",        "models"),
    ("profiler",      "profiling"),
    ("tools",         "tools"),
    ("_utils",        "utils"),
    ("runtime",       "runtime"),
]


def _path_to_category(rel_path: str) -> str:
    parts = Path(rel_path).parts
    for seg, cat in _PATH_CATEGORIES:
        if seg in parts:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Matches: os.getenv("VAR", ...) or os.environ.get("VAR", ...)
_GETENV_RE = re.compile(
    r'os\.(?:getenv|environ\.get)\s*\(\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*'
    r'(?:,\s*([^)\n]+?))?\s*\)',
    re.MULTILINE,
)

# Matches: os.environ["VAR"] used as a read (in conditions / assignments)
_ENVIRON_ITEM_RE = re.compile(
    r'os\.environ\s*\[\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*\]',
)

# StrEnum class body: class FooEnvs(StrEnum): ...
_STRENUM_CLASS_RE = re.compile(
    r'class\s+\w+\s*\([^)]*StrEnum[^)]*\)\s*:\s*\n((?:[ \t]+[^\n]*\n)*)',
    re.MULTILINE,
)
_STRENUM_MEMBER_RE = re.compile(
    r'^\s*([A-Z][A-Z0-9_]+)\s*=\s*["\']([A-Z][A-Z0-9_]+)["\']',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Default-value parser
# ---------------------------------------------------------------------------
def _parse_default(raw: str | None) -> Any:
    if not raw:
        return None
    s = raw.strip().rstrip(",").strip()
    if s in ("None", ""):
        return None
    if s == "True":
        return True
    if s == "False":
        return False
    for q in ('"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2:
            return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return None  # complex expression — leave as None


def _type_from_default(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, (list, tuple)):
        inner = "int" if default and isinstance(default[0], int) and not isinstance(default[0], bool) else "str"
        return f"List[{inner}]"
    return "str"


# ---------------------------------------------------------------------------
# Comment extraction
# ---------------------------------------------------------------------------
def _preceding_comment(source: str, match_start: int) -> str:
    """Collect # comment lines immediately before the match line."""
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
    """Get trailing # comment on the same line as the match."""
    line_end = source.find("\n", match_start)
    if line_end == -1:
        line_end = len(source)
    line = source[match_start:line_end]
    if "#" in line:
        return line[line.index("#") + 1:].strip()
    return ""


# ---------------------------------------------------------------------------
# Source scanner
# ---------------------------------------------------------------------------
def _scan_sources() -> dict[str, dict]:
    found: dict[str, dict] = {}

    for py_file in sorted(_PKG_DIR.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Skip files that clearly can't contain our vars
        if not any(p in source for p in _TARGET_PREFIXES):
            continue

        rel = str(py_file.relative_to(_PKG_DIR))
        cat = _path_to_category(rel)

        # os.getenv / os.environ.get
        for m in _GETENV_RE.finditer(source):
            name = m.group(1)
            if not any(name.startswith(p) for p in _TARGET_PREFIXES):
                continue
            default = _parse_default(m.group(2))
            desc = (
                _preceding_comment(source, m.start())
                or _inline_comment(source, m.start())
            )
            if name not in found:
                found[name] = {
                    "default": default,
                    "description": desc,
                    "category": cat,
                }
            else:
                # Keep the first non-None default and first description found
                if found[name]["default"] is None and default is not None:
                    found[name]["default"] = default
                if not found[name]["description"] and desc:
                    found[name]["description"] = desc

        # os.environ["VAR"] direct access (usually writes, so no default)
        for m in _ENVIRON_ITEM_RE.finditer(source):
            name = m.group(1)
            if not any(name.startswith(p) for p in _TARGET_PREFIXES):
                continue
            if name not in found:
                found[name] = {
                    "default": None,
                    "description": _preceding_comment(source, m.start()),
                    "category": cat,
                }

    return found


def _scan_strenum_classes() -> dict[str, dict]:
    """Collect env var keys defined as members of StrEnum subclasses."""
    found: dict[str, dict] = {}

    for py_file in sorted(_PKG_DIR.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "StrEnum" not in source:
            continue

        rel = str(py_file.relative_to(_PKG_DIR))
        cat = _path_to_category(rel)

        for cls_m in _STRENUM_CLASS_RE.finditer(source):
            body = cls_m.group(1)
            for mem_m in _STRENUM_MEMBER_RE.finditer(body):
                var_name = mem_m.group(2)  # the string value, e.g. "TLLM_SPAWN_..."
                if not any(var_name.startswith(p) for p in _TARGET_PREFIXES):
                    continue
                if var_name not in found:
                    found[var_name] = {
                        "default": None,
                        "description": "",
                        "category": cat,
                    }

    return found


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for TRT-LLM deployment
# ---------------------------------------------------------------------------
# Move vars between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
]

_UI_ADVANCED = [
    "TRTLLM_DISAGG_ROLE",
    "TRTLLM_DISAGG_DEPLOYMENT_ID",
    "TLLM_DISAGG_INSTANCE_IDX",
    "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT",
    "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP",
    "TRTLLM_KV_TRANSFER_NUM_THREADS",
    "TRTLLM_NIXL_NUM_THREADS",
    "TRTLLM_USE_PY_NIXL_KVCACHE",
    "TLLM_MULTIMODAL_DISAGGREGATED",
    "TRTLLM_ENABLE_PYAV",
    "TRTLLM_MEDIA_STORAGE_PATH",
    "TRTLLM_NO_USAGE_STATS",
]


# ---------------------------------------------------------------------------
# Build JSON
# ---------------------------------------------------------------------------
def build_json() -> dict:
    # StrEnum constants first (weaker — no defaults), then overwrite with
    # scanned getenv calls (stronger — have defaults and context)
    all_vars = {**_scan_strenum_classes(), **_scan_sources()}

    vars_: dict[str, dict] = {}
    for name in sorted(all_vars):
        info = all_vars[name]
        default = info["default"]
        record: dict[str, Any] = {
            "category": info["category"],
            "type": _type_from_default(default),
            "default_value": default,
            "value": None,
            "description": info["description"],
            "ui": False,
            "aic": False,
        }
        vars_[name] = record

    ordered = {}
    for k in _UI_PRIMARY:
        if k in vars_:
            vars_[k]["ui"] = "primary"
            ordered[k] = vars_[k]
    for k in _UI_ADVANCED:
        if k in vars_:
            vars_[k]["ui"] = "advanced"
            ordered[k] = vars_[k]
    for k, v in vars_.items():
        if k not in ordered:
            v["ui"] = "less_frequent"
            ordered[k] = v
    return {
        "engine": "tensorrt_llm",
        "version": TRTLLM_VERSION,
        "date": date.today().isoformat(),
        "source": "source-scanned from tensorrt_llm package (os.getenv + StrEnum patterns)",
        **ordered,
    }


# ---------------------------------------------------------------------------
# Delta — compare two versions of this JSON
# ---------------------------------------------------------------------------
_DELTA_META   = frozenset({"engine", "version", "date", "source"})
_DELTA_SKIP   = frozenset({"ui", "aic", "value"})


def _compute_delta(old_data: dict, new_data: dict) -> dict:
    old_e = {k: v for k, v in old_data.items() if k not in _DELTA_META}
    new_e = {k: v for k, v in new_data.items() if k not in _DELTA_META}
    added   = {k: new_e[k] for k in sorted(set(new_e) - set(old_e))}
    removed = {k: old_e[k] for k in sorted(set(old_e) - set(new_e))}
    changed: dict = {}
    unchanged = 0
    for k in sorted(set(old_e) & set(new_e)):
        diff = {
            f: {"old": old_e[k].get(f), "new": new_e[k].get(f)}
            for f in set(old_e[k]) | set(new_e[k])
            if f not in _DELTA_SKIP and old_e[k].get(f) != new_e[k].get(f)
        }
        if diff:
            changed[k] = {"diff": diff}
        else:
            unchanged += 1
    return {
        "engine": new_data.get("engine"), "old_version": old_data.get("version"),
        "new_version": new_data.get("version"), "date": date.today().isoformat(),
        "summary": {"total_old": len(old_e), "total_new": len(new_e),
                    "added": len(added), "removed": len(removed),
                    "changed": len(changed), "unchanged": unchanged},
        "added": added, "removed": removed, "changed": changed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--out",
        default="trtllm_envs.json",
        help="Output file path (default: trtllm_envs.json)",
    )
    cli.add_argument("--prev", default=None, metavar="OLD_JSON",
                     help="Previous version JSON to diff against (produces a delta file)")
    ns = cli.parse_args()

    data = build_json()

    with open(ns.out, "w") as fh:
        json.dump(data, fh, indent=2)

    meta_keys = {"engine", "version", "date", "source"}
    vars_ = {k: v for k, v in data.items() if k not in meta_keys}
    cats: dict[str, int] = {}
    for rec in vars_.values():
        cats[rec["category"]] = cats.get(rec["category"], 0) + 1
    print(
        f"Written {ns.out!r}  —  tensorrt_llm {data['version']}  "
        f"—  {len(cats)} categories  {len(vars_)} total vars"
    )
    for cat, count in sorted(cats.items()):
        print(f"  {cat:25s}  {count} vars")

    if ns.prev:
        with open(ns.prev) as fh:
            old_data = json.load(fh)
        delta = _compute_delta(old_data, data)
        delta_out = str(Path(ns.out).parent / f"delta__{Path(ns.prev).stem}__{Path(ns.out).stem}.json")
        with open(delta_out, "w") as fh:
            json.dump(delta, fh, indent=2)
        s = delta["summary"]
        print(f"\nDelta  {delta_out!r}  —  {delta['old_version']} → {delta['new_version']}")
        print(f"  added: {s['added']}  removed: {s['removed']}  changed: {s['changed']}  unchanged: {s['unchanged']}")
