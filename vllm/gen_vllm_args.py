"""
Generate vllm_args.json by introspecting the vLLM argparse parser.

Usage:
    python gen_vllm_args.py [--out vllm_args.json]

Requirements:
    Run inside a Python environment that has the target vLLM version installed.
    No GPU required — DeviceConfig is patched to bypass hardware detection.
"""

import argparse
import json
import sys
from datetime import date, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Patch DeviceConfig before vllm.config is imported so GPU detection is
# bypassed.  This lets the script run on CPU-only machines.
# ---------------------------------------------------------------------------
def _patch_device_config() -> None:
    try:
        import vllm.config.device as _dev
        _dev.DeviceConfig.__post_init__ = lambda self: setattr(
            self, "device_type", "cpu"
        )
    except Exception:
        pass  # older vllm layout — not needed


_patch_device_config()

# Force vllm's arg_utils to populate help/description strings.
# Its NEEDS_HELP flag is True only when --help or mkdocs is in argv.
_real_argv = sys.argv[:]
sys.argv = ["mkdocs"]  # triggers NEEDS_HELP = True inside arg_utils

# ---------------------------------------------------------------------------
# Now safe to import vllm
# ---------------------------------------------------------------------------
try:
    import vllm
    VLLM_VERSION: str = vllm.__version__
except ImportError:
    sys.argv = _real_argv
    sys.exit("vllm is not installed in this environment.")

from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402

sys.argv = _real_argv  # restore immediately after import

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser as _ParserCls
except ImportError:
    _ParserCls = argparse.ArgumentParser  # type: ignore[misc]

import re as _re


_MODULE_KEY_OVERRIDES: dict[str, str] = {
    # Compound words or abbreviations that snake_case mis-splits
    "MultiModalConfig": "multimodal",
    "LoRAConfig": "lora",
    "VllmConfig": "vllm",
}


def _config_class_to_module_key(title: str) -> str:
    """Convert 'StructuredOutputsConfig' → 'structured_outputs', etc."""
    if title in _MODULE_KEY_OVERRIDES:
        return _MODULE_KEY_OVERRIDES[title]
    if title in ("options", "optional arguments"):
        return "engine"
    name = title[: -len("Config")] if title.endswith("Config") else title
    # CamelCase → snake_case
    s = _re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = _re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Helper: normalise a default value so it is JSON-serialisable
# ---------------------------------------------------------------------------
def _serialise_default(val, _depth: int = 0) -> object:
    """Recursively convert any default value to a JSON-safe type."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialise_default(v, _depth + 1) for v in val]
    if isinstance(val, (set, frozenset)):
        return sorted(str(_serialise_default(v, _depth + 1)) for v in val)
    if isinstance(val, dict):
        return {str(k): _serialise_default(v, _depth + 1) for k, v in val.items()}
    # Guard against infinite recursion on deeply nested configs
    if _depth > 6:
        return str(val)
    # dataclass instance (including pydantic dataclasses)
    try:
        import dataclasses as _dc
        if _dc.is_dataclass(val) and not isinstance(val, type):
            return {
                f.name: _serialise_default(getattr(val, f.name, None), _depth + 1)
                for f in _dc.fields(val)
            }
    except Exception:
        pass
    # Enum
    try:
        return _serialise_default(val.value, _depth + 1)
    except AttributeError:
        pass
    # Fallback: string repr (always JSON-safe)
    return str(val)


# ---------------------------------------------------------------------------
# Infer exact Python type string from an argparse action
# ---------------------------------------------------------------------------
def _infer_arg_type(action: argparse.Action) -> str:
    nargs = getattr(action, 'nargs', None)
    is_list = nargs in ('+', '*') or (isinstance(nargs, int) and nargs > 1)
    atype = action.type
    if atype is int:
        base = "int"
    elif atype is float:
        base = "float"
    elif atype is str:
        base = "str"
    elif atype is None:
        dv = action.default
        if isinstance(dv, bool):
            base = "bool"
        elif isinstance(dv, int):
            base = "int"
        elif isinstance(dv, float):
            base = "float"
        else:
            base = "str"
    else:
        tname = getattr(atype, '__name__', '') or ''
        if 'int' in tname.lower():
            base = "int"
        elif 'float' in tname.lower():
            base = "float"
        else:
            base = "str"
    return f"List[{base}]" if is_list else base


# ---------------------------------------------------------------------------
# Build the JSON record for a single argparse action
# ---------------------------------------------------------------------------
def _action_to_record(action: argparse.Action) -> dict:
    dest: str = action.dest
    default = _serialise_default(action.default)
    choices = list(action.choices) if action.choices is not None else None
    opts: list[str] = action.option_strings

    # --- boolean flag (BooleanOptionalAction → --foo / --no-foo) ---
    if isinstance(action, argparse.BooleanOptionalAction):
        true_arg = next((o for o in opts if not o.startswith("--no-")), opts[0])
        false_arg = next((o for o in opts if o.startswith("--no-")), f"--no-{opts[0][2:]}")
        return {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else False,
            "value": None,
            "true_arg": true_arg,
            "false_arg": false_arg,
            "description": (action.help or "").replace("%%", "%").strip(),
        }

    # --- store_true (old-style boolean, no --no- counterpart) ---
    if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        flag = opts[0] if opts else f"--{dest.replace('_', '-')}"
        return {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else False,
            "value": None,
            "true_arg": flag,
            "false_arg": None,
            "description": (action.help or "").replace("%%", "%").strip(),
        }

    type_str = _infer_arg_type(action)

    # --- choice ---
    if choices:
        if all(str(c).lstrip('-').isdigit() for c in choices):
            type_str = "int"
        return {
            "type": type_str,
            "flag": False,
            "arg": opts[0] if opts else f"--{dest.replace('_', '-')}",
            "default_value": default,
            "value": None,
            "choices": choices,
            "description": (action.help or "").replace("%%", "%").strip(),
        }

    # --- plain value ---
    return {
        "type": type_str,
        "flag": False,
        "arg": opts[0] if opts else f"--{dest.replace('_', '-')}",
        "default_value": default,
        "value": None,
        "description": (action.help or "").replace("%%", "%").strip(),
    }


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for vLLM deployment
# ---------------------------------------------------------------------------
# Move args between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
    "model", "dtype", "tensor_parallel_size", "pipeline_parallel_size",
    "gpu_memory_utilization", "max_model_len",
    "host", "port", "served_model_name", "api_key",
    "trust_remote_code", "quantization", "kv_cache_dtype",
    "max_num_seqs", "enable_prefix_caching",
]

_UI_ADVANCED = [
    "max_num_batched_tokens", "enable_chunked_prefill", "scheduling_policy",
    "load_format", "tokenizer", "chat_template", "chat_template_content_format",
    "speculative_config", "num_speculative_tokens",
    "enable_lora", "max_loras", "lora_modules",
    "limit_mm_per_prompt", "distributed_executor_backend",
    "enforce_eager", "max_seq_len_to_capture",
    "rope_scaling", "rope_theta",
    "disable_log_requests", "uvicorn_log_level",
]


# ---------------------------------------------------------------------------
# Build the parser and walk its groups
# ---------------------------------------------------------------------------
def build_json() -> dict:
    parser = _ParserCls(add_help=False)
    make_arg_parser(parser)

    args: dict[str, dict] = {}

    for group in parser._action_groups:  # noqa: SLF001
        actions = [a for a in group._group_actions if a.dest != "help"]  # noqa: SLF001
        if not actions:
            continue

        title: str = group.title or "engine"
        module_key = _config_class_to_module_key(title)
        config_class = title if module_key != "engine" else "EngineArgs"

        for action in actions:
            record = _action_to_record(action)
            record["module"] = module_key
            record["config_class"] = config_class
            record["ui"] = False
            record["aic"] = False
            # Last group wins for duplicate dest names
            args[action.dest] = record

    ordered = {}
    for k in _UI_PRIMARY:
        if k in args:
            args[k]["ui"] = "primary"
            ordered[k] = args[k]
    for k in _UI_ADVANCED:
        if k in args:
            args[k]["ui"] = "advanced"
            ordered[k] = args[k]
    for k, v in args.items():
        if k not in ordered:
            v["ui"] = "less_frequent"
            ordered[k] = v
    return {
        "engine": "vllm",
        "version": VLLM_VERSION,
        "date": date.today().isoformat(),
        "source": "introspected from vllm.entrypoints.openai.cli_args.make_arg_parser (vllm serve)",
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
        default="vllm_args.json",
        help="Output file path (default: vllm_args.json)",
    )
    cli.add_argument(
        "--prev",
        default=None,
        metavar="OLD_JSON",
        help="Previous version JSON to diff against (produces a delta file)",
    )
    ns = cli.parse_args()

    data = build_json()

    with open(ns.out, "w") as fh:
        json.dump(data, fh, indent=2)

    meta_keys = {"engine", "version", "date", "source"}
    args = {k: v for k, v in data.items() if k not in meta_keys}
    modules: dict[str, int] = {}
    for rec in args.values():
        modules[rec["module"]] = modules.get(rec["module"], 0) + 1
    print(f"Written {ns.out!r}  —  vllm {data['version']}  "
          f"—  {len(modules)} modules  {len(args)} total args")
    for mod, count in sorted(modules.items()):
        print(f"  {mod:22s}  {count} args")

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
