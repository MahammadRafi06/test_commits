"""
Generate sglang_args.json by introspecting SGLang's ServerArgs argparse parser.

Usage:
    python gen_sglang_args.py [--out sglang_args.json]

Requirements:
    Run inside a Python environment that has the target SGLang version installed.
    No GPU required.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Import sglang
# ---------------------------------------------------------------------------
try:
    import sglang
    SGLANG_VERSION: str = sglang.__version__
except ImportError:
    sys.exit("sglang is not installed in this environment.")

from sglang.srt.server_args import ServerArgs


# ---------------------------------------------------------------------------
# Helper: normalise a default value so it is JSON-serialisable
# ---------------------------------------------------------------------------
def _serialise_default(val: Any, _depth: int = 0) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialise_default(v, _depth + 1) for v in val]
    if isinstance(val, (set, frozenset)):
        return sorted(str(_serialise_default(v, _depth + 1)) for v in val)
    if isinstance(val, dict):
        return {str(k): _serialise_default(v, _depth + 1) for k, v in val.items()}
    if _depth > 6:
        return str(val)
    try:
        import dataclasses as _dc
        if _dc.is_dataclass(val) and not isinstance(val, type):
            return {
                f.name: _serialise_default(getattr(val, f.name, None), _depth + 1)
                for f in _dc.fields(val)
            }
    except Exception:
        pass
    try:
        return _serialise_default(val.value, _depth + 1)
    except AttributeError:
        pass
    return str(val)


# ---------------------------------------------------------------------------
# Build JSON record for a single argparse action
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


def _action_to_record(action: argparse.Action) -> dict:
    dest: str = action.dest
    default = _serialise_default(action.default)
    choices = list(action.choices) if action.choices is not None else None
    opts: list[str] = action.option_strings

    # bool: BooleanOptionalAction (--foo / --no-foo)
    if isinstance(action, argparse.BooleanOptionalAction):
        true_arg = next((o for o in opts if not o.startswith("--no-")), opts[0])
        false_arg = next(
            (o for o in opts if o.startswith("--no-")),
            f"--no-{opts[0][2:]}",
        )
        return {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else False,
            "value": None,
            "true_arg": true_arg,
            "false_arg": false_arg,
            "description": (action.help or "").replace("%%", "%").strip(),
        }

    # bool: store_true
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

    # choice (constrained string/int)
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

    # plain value
    return {
        "type": type_str,
        "flag": False,
        "arg": opts[0] if opts else f"--{dest.replace('_', '-')}",
        "default_value": default,
        "value": None,
        "description": (action.help or "").replace("%%", "%").strip(),
    }


# ---------------------------------------------------------------------------
# Map argparse group title to a module key
# ---------------------------------------------------------------------------
_GROUP_TITLE_MAP: dict[str, str] = {
    "options": "server",
    "optional arguments": "server",
    "positional arguments": "server",
}


def _title_to_module_key(title: str) -> str:
    lower = title.lower()
    if lower in _GROUP_TITLE_MAP:
        return _GROUP_TITLE_MAP[lower]
    return lower.replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for SGLang deployment
# ---------------------------------------------------------------------------
# Move args between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
    "model_path", "dtype", "tp_size", "dp_size",
    "context_length", "mem_fraction_static",
    "host", "port", "served_model_name", "api_key",
    "trust_remote_code", "quantization", "kv_cache_dtype",
    "max_running_requests", "disable_radix_cache",
]

_UI_ADVANCED = [
    "tokenizer_path", "chat_template_path",
    "schedule_policy", "max_total_tokens",
    "speculative_algorithm", "speculative_draft_model_path",
    "num_speculative_steps",
    "enable_lora", "lora_paths", "max_loras_per_batch",
    "tokenizer_mode", "random_seed",
    "attention_backend", "sampling_backend",
    "enable_cache_report", "chunked_prefill_size",
    "max_prefill_tokens", "decode_log_interval",
    "enable_double_sparsity", "triton_attention_reduce_in_fp32",
]


# ---------------------------------------------------------------------------
# Build the full JSON structure
# ---------------------------------------------------------------------------
def build_json() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)

    args: dict[str, dict] = {}

    for group in parser._action_groups:  # noqa: SLF001
        actions = [a for a in group._group_actions if a.dest != "help"]  # noqa: SLF001
        if not actions:
            continue

        title: str = group.title or "server"
        module_key = _title_to_module_key(title)
        config_class = title if module_key != "server" else "ServerArgs"

        for action in actions:
            record = _action_to_record(action)
            record["module"] = module_key
            record["config_class"] = config_class
            record["ui"] = False
            record["aic"] = False
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
        "engine": "sglang",
        "version": SGLANG_VERSION,
        "date": date.today().isoformat(),
        "source": "introspected from sglang.srt.server_args.ServerArgs.add_cli_args",
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
        default="sglang_args.json",
        help="Output file path (default: sglang_args.json)",
    )
    cli.add_argument("--prev", default=None, metavar="OLD_JSON",
                     help="Previous version JSON to diff against (produces a delta file)")
    ns = cli.parse_args()

    data = build_json()

    with open(ns.out, "w") as fh:
        json.dump(data, fh, indent=2)

    meta_keys = {"engine", "version", "date", "source"}
    args = {k: v for k, v in data.items() if k not in meta_keys}
    modules: dict[str, int] = {}
    for rec in args.values():
        modules[rec["module"]] = modules.get(rec["module"], 0) + 1
    print(
        f"Written {ns.out!r}  —  sglang {data['version']}  "
        f"—  {len(modules)} modules  {len(args)} total args"
    )
    for mod, count in sorted(modules.items()):
        print(f"  {mod:25s}  {count} args")

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
