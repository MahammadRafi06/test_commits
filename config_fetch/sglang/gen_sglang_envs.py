"""
Generate sglang_envs.json by introspecting sglang.srt.environ.Envs and
parsing its source to extract descriptions and categories.

Usage:
    python gen_sglang_envs.py [--out sglang_envs.json]

Requirements:
    Run inside a Python environment that has the target SGLang version installed.
    No GPU required.
"""

import argparse
import inspect
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Import sglang
# ---------------------------------------------------------------------------
try:
    import sglang
    try:
        SGLANG_VERSION: str = sglang.__version__
    except AttributeError:
        from importlib.metadata import version
        SGLANG_VERSION = version("sglang")
except ImportError:
    sys.exit("sglang is not installed in this environment.")

import sglang.srt.environ as _environ_module
from sglang.srt.environ import Envs, EnvBool, EnvField


# ---------------------------------------------------------------------------
# Map source section header text → category key
# ---------------------------------------------------------------------------
_SECTION_OVERRIDES: dict[str, str] = {
    "Model & File Download":                          "model",
    "Logging Options":                                "logging",
    "SGLang CI":                                      "ci",
    "Constrained Decoding (Grammar)":                 "grammar",
    "Test & Debug":                                   "debug",
    "Scheduler: memory leak test":                    "scheduler",
    "Scheduler: new token ratio hyperparameters":     "scheduler",
    "Scheduler: recv interval":                       "scheduler",
    "Scheduler: others:":                             "scheduler",
    "Scheduler: others":                              "scheduler",
    "PD Disaggregation (runtime)":                    "disaggregation",
    "Test: pd-disaggregation":                        "disaggregation",
    "Model Parallel":                                 "distributed",
    "Tool Calling":                                   "tool_calling",
    "Hi-Cache":                                       "kv_cache",
    "AMD & ROCm":                                     "rocm",
    "MPS (Apple Silicon)":                            "mps",
    "NPU":                                            "npu",
    "MTHREADS & MUSA":                                "musa",
    "Quantization":                                   "quantization",
    "Flashinfer":                                     "flashinfer",
    "Triton":                                         "triton",
    "Torch Compile":                                  "compilation",
    "EPLB":                                           "eplb",
    "TBO":                                            "tbo",
    "DeepGemm":                                       "deepgemm",
    "DeepSeek MHA Optimization":                      "deepseek",
    "DeepEP":                                         "deepep",
    "NIXL-EP":                                        "nixl",
    "NSA Backend":                                    "nsa",
    "sgl-kernel":                                     "kernel",
    "Flash Attention":                                "attention",
    "Kernels":                                        "kernels",
    "Deterministic inference":                        "engine",
    "RoPE cache configuration":                       "attention",
    "Overlap Spec V2":                                "speculative",
    "Spec Config":                                    "speculative",
    "VLM":                                            "multimodal",
    "VLM Item CUDA IPC Transport":                    "multimodal",
    "Mamba":                                          "mamba",
    "Unified Radix Tree":                             "cache",
    "Breakable CUDA Graph":                           "engine",
    "Release & Resume Memory":                        "engine",
    "Sparse Embeddings":                              "engine",
    "Logits processor":                               "engine",
    "Tool-Call behavior":                             "tool_calling",
    "Ngram":                                          "speculative",
    "Warmup":                                         "engine",
    "HTTP Server":                                    "networking",
    "HTTP/2 Server":                                  "networking",
    "Health Check":                                   "networking",
    "Encoder gRPC":                                   "networking",
    "Native gRPC server (internal, not yet user-facing)": "networking",
    "External models":                                "external",
    "Numa":                                           "system",
    "Metrics":                                        "observability",
    "TokenizerManager":                               "engine",
    "Symmetric Memory":                               "engine",
    "Aiter":                                          "rocm",
    "EPD":                                            "disaggregation",
    "Elastic EP Backup Port":                         "networking",
    "Sglang Cache Dir":                               "cache",
    "Plugin system":                                  "plugins",
    "Mooncake Store":                                 "kv_cache",
    "Mooncake KV Transfer":                           "kv_cache",
}

# Prefix-based fallbacks for long section names
_SECTION_PREFIXES: list[tuple[str, str]] = [
    ("Tokenizer",   "engine"),
    ("Scheduler",   "scheduler"),
    ("Test:",       "debug"),
    ("VLM",         "multimodal"),
]


def _section_to_key(comment_text: str) -> str:
    known = _known_section_key(comment_text)
    if known is not None:
        return known
    text = comment_text.rstrip(":")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _known_section_key(comment_text: str) -> str | None:
    text = comment_text.rstrip(":")
    if text in _SECTION_OVERRIDES:
        return _SECTION_OVERRIDES[text]
    # Original text (with trailing colon) lookup
    if comment_text in _SECTION_OVERRIDES:
        return _SECTION_OVERRIDES[comment_text]
    for prefix, key in _SECTION_PREFIXES:
        if text.startswith(prefix):
            return key
    return None


def _next_significant_line(lines: list[str], start_idx: int) -> str:
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# ---------------------------------------------------------------------------
# Parse source to extract per-variable descriptions and categories
# ---------------------------------------------------------------------------
def _parse_envs_source(source: str) -> dict[str, dict]:
    """
    Walk the Envs class body and for each variable:
      - category: derived from the section-header comment that follows a blank line
      - description: comments immediately preceding the variable (no blank line between)
        plus any trailing inline comment on the assignment line
    """
    m = re.search(r"class Envs:\s*\n(.*?)(?=\nclass |\Z)", source, re.DOTALL)
    if not m:
        return {}
    body = m.group(1)
    lines = body.splitlines()

    results: dict[str, dict] = {}
    current_section = "general"
    pending_comments: list[str] = []
    prev_was_blank = True  # treat start of class body like after a blank line

    for idx, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            prev_was_blank = True
            pending_comments = []
            continue

        if stripped.startswith("#"):
            comment_text = stripped.lstrip("# ").strip()
            # Skip formatter directives
            if comment_text in ("fmt: off", "fmt: on"):
                continue
            if re.match(r"^=+$", comment_text):
                continue

            if prev_was_blank:
                # Section header: updates current category, not a description
                section_key = _known_section_key(comment_text)
                next_line = _next_significant_line(lines, idx)
                if section_key is not None:
                    current_section = section_key
                    pending_comments = []
                elif re.match(r"[A-Z][A-Z0-9_]+\s*=", next_line):
                    pending_comments.append(comment_text)
                else:
                    current_section = _section_to_key(comment_text)
                    pending_comments = []
            else:
                # Description comment for the next variable
                pending_comments.append(comment_text)
            prev_was_blank = False
            continue

        # Variable assignment: VAR_NAME = EnvType(...)
        assign_m = re.match(r"([A-Z][A-Z0-9_]+)\s*=\s*", stripped)
        if assign_m:
            var_name = assign_m.group(1)

            # Inline comment on the same line (e.g. `EnvFloat(-1)  # in seconds`)
            inline_m = re.search(r"#\s*(.+)$", stripped)
            inline = inline_m.group(1).strip() if inline_m else ""

            desc = " ".join(pending_comments)
            if inline and not desc:
                desc = inline

            results[var_name] = {
                "description": desc,
                "category": current_section,
            }
            pending_comments = []
            prev_was_blank = False
            continue

        prev_was_blank = False

    return results


# ---------------------------------------------------------------------------
# Infer type label from EnvField subclass
# ---------------------------------------------------------------------------
def _type_label(field: EnvField) -> str:
    cls = type(field).__name__
    if isinstance(field, EnvBool) or cls == "EnvBool":
        return "bool"
    if "Int" in cls:
        return "int"
    if "Float" in cls:
        return "float"
    if "Tuple" in cls or "List" in cls:
        return "List[str]"
    # Infer from default value as fallback
    dv = field.default
    if isinstance(dv, bool):
        return "bool"
    if isinstance(dv, int):
        return "int"
    if isinstance(dv, float):
        return "float"
    if isinstance(dv, (list, tuple)):
        return "List[str]"
    return "str"


# ---------------------------------------------------------------------------
# Serialise default value to JSON-safe type
# ---------------------------------------------------------------------------
def _serialise(val: Any) -> Any:
    if isinstance(val, str):
        try:
            home = str(Path.home())
        except Exception:
            home = ""
        if home and (val == home or val.startswith(home + os.sep)):
            return "~" + val[len(home):]
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialise(v) for v in val]
    return str(val)


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for SGLang deployment
# ---------------------------------------------------------------------------
# Move vars between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
]

_UI_ADVANCED = [
    "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
    "SGLANG_DISAGGREGATION_QUEUE_SIZE",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
    "SGLANG_DISAGG_STAGING_BUFFER",
    "SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE",
    "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE",
    "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "SGLANG_MM_FEATURE_CACHE_MB",
    "SGLANG_FORWARD_UNKNOWN_TOOLS",
    "SGLANG_TOOL_STRICT_LEVEL",
    "SGLANG_LOG_MS", "SGLANG_LOG_REQUEST_HEADERS",
]


# ---------------------------------------------------------------------------
# Build the full JSON structure
# ---------------------------------------------------------------------------
def build_json() -> dict:
    source = inspect.getsource(_environ_module)
    meta = _parse_envs_source(source)

    vars_: dict[str, dict] = {}

    for attr_name, field in vars(Envs).items():
        if not isinstance(field, EnvField):
            continue

        info = meta.get(attr_name, {"description": "", "category": "general"})
        record: dict[str, Any] = {
            "category": info["category"],
            "type": _type_label(field),
            "default_value": _serialise(field.default),
            "value": None,
            "description": info["description"],
            "ui": False,
            "aic": False,
        }
        vars_[attr_name] = record

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
        "engine": "sglang",
        "version": SGLANG_VERSION,
        "date": date.today().isoformat(),
        "source": "introspected from sglang.srt.environ.Envs",
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
        default="sglang_envs.json",
        help="Output file path (default: sglang_envs.json)",
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
        f"Written {ns.out!r}  —  sglang {data['version']}  "
        f"—  {len(cats)} categories  {len(vars_)} total vars"
    )
    for cat, count in sorted(cats.items()):
        print(f"  {cat:30s}  {count} vars")

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
