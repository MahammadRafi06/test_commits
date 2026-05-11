"""
Generate trtllm_args.json by introspecting:
  1. The trtllm-serve Click CLI command  → module "serve"
  2. BaseLlmArgs (common to all backends) → module "base"
  3. TrtLlmArgs (TensorRT-only fields)   → module "tensorrt"
  4. TorchLlmArgs (PyTorch-only fields)  → module "pytorch"

Usage:
    python gen_trtllm_args.py [--out trtllm_args.json]

Requirements:
    Run inside a Python environment that has TensorRT-LLM installed.
    No GPU required for the introspection itself.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin


# ---------------------------------------------------------------------------
# Import tensorrt_llm
# ---------------------------------------------------------------------------
try:
    import tensorrt_llm as _trtllm
    TRTLLM_VERSION: str = _trtllm.__version__
except ImportError:
    sys.exit("tensorrt_llm is not installed in this environment.")

import click
from tensorrt_llm.commands.serve import serve as _serve_cmd
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs, TorchLlmArgs, TrtLlmArgs

try:
    from pydantic.fields import PydanticUndefined as _Undefined
except ImportError:
    _Undefined = None  # fallback — treat everything as having no default


# ---------------------------------------------------------------------------
# Serialise helper
# ---------------------------------------------------------------------------
def _serialise(val: Any, _depth: int = 0) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialise(v, _depth + 1) for v in val]
    if isinstance(val, (set, frozenset)):
        return sorted(str(_serialise(v, _depth + 1)) for v in val)
    if isinstance(val, dict):
        return {str(k): _serialise(v, _depth + 1) for k, v in val.items()}
    if _depth > 6:
        return str(val)
    try:
        import dataclasses as _dc
        if _dc.is_dataclass(val) and not isinstance(val, type):
            return {
                f.name: _serialise(getattr(val, f.name, None), _depth + 1)
                for f in _dc.fields(val)
            }
    except Exception:
        pass
    try:
        return _serialise(val.value, _depth + 1)
    except AttributeError:
        pass
    return str(val)


# ---------------------------------------------------------------------------
# Click introspection
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r":tag:`([^`]+)`")

_CLICK_TYPE_MAP: dict[str, str] = {
    "INTEGER": "int", "INT": "int",
    "FLOAT": "float",
    "BOOLEAN": "bool",
    "TEXT": "str", "STRING": "str",
    "PATH": "path", "FILENAME": "path",
}


def _strip_tag(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _extract_status(text: str) -> str | None:
    m = _TAG_RE.search(text)
    return m.group(1) if m else None


def _click_type_str(param: click.Parameter) -> str:
    if param.is_flag:
        return "bool"
    if isinstance(param.type, click.Choice):
        choices = param.type.choices
        if choices and all(str(c).lstrip('-').isdigit() for c in choices):
            return "int"
        return "str"
    type_name = getattr(param.type, 'name', 'TEXT').upper()
    return _CLICK_TYPE_MAP.get(type_name, "str")


def _click_param_to_record(param: click.Parameter) -> dict:
    # Positional argument (e.g. MODEL)
    if isinstance(param, click.Argument):
        return {
            "type": "str",
            "flag": False,
            "arg": param.name,
            "default_value": None,
            "value": None,
            "required": True,
            "description": "",
        }

    # click.Option
    opts: list[str] = list(param.opts)
    raw_help = param.help or ""
    description = _strip_tag(raw_help)
    status = _extract_status(raw_help)
    default = _serialise(param.default)
    type_str = _click_type_str(param)

    # Boolean flags
    if param.is_flag:
        neg = [o for o in opts if o.lstrip("-").startswith("no-")]
        pos = [o for o in opts if not o.lstrip("-").startswith("no-")]
        if neg and pos:
            rec = {
                "type": "bool",
                "flag": True,
                "default_value": bool(param.default)
                if param.default is not None else False,
                "value": None,
                "true_arg": pos[0],
                "false_arg": neg[0],
                "description": description,
            }
        else:
            rec = {
                "type": "bool",
                "flag": True,
                "default_value": bool(param.default)
                if param.default is not None else False,
                "value": None,
                "true_arg": opts[0] if opts else f"--{param.name}",
                "false_arg": None,
                "description": description,
            }
        if status:
            rec["status"] = status
        return rec

    # Choice (click.Choice or subclass like ChoiceWithAlias)
    if isinstance(param.type, click.Choice):
        rec = {
            "type": type_str,
            "flag": False,
            "arg": opts[0] if opts else f"--{param.name}",
            "default_value": default,
            "value": None,
            "choices": list(param.type.choices),
            "description": description,
        }
        if status:
            rec["status"] = status
        return rec

    # Generic value (may have multiple=True)
    rec = {
        "type": type_str if not getattr(param, "multiple", False) else f"List[{type_str}]",
        "flag": False,
        "arg": opts[0] if opts else f"--{param.name}",
        "default_value": default,
        "value": None,
        "description": description,
    }
    if status:
        rec["status"] = status
    return rec


# ---------------------------------------------------------------------------
# Pydantic introspection
# ---------------------------------------------------------------------------
def _core_type(annotation: Any) -> Any:
    """Unwrap Annotated[...] and Optional[X] to reach the core type."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    return annotation


def _annotation_to_type(annotation: Any) -> str:
    """Map a Python type annotation to a validation type string."""
    core = _core_type(annotation)
    if core is bool:
        return "bool"
    if core is int:
        return "int"
    if core is float:
        return "float"
    if core is str:
        return "str"
    origin = get_origin(core)
    if origin is list:
        inner_args = get_args(core)
        inner = _annotation_to_type(inner_args[0]) if inner_args else "str"
        return f"List[{inner}]"
    if origin is dict:
        return "Dict[str, Any]"
    if origin is Literal:
        lit_vals = get_args(core)
        if lit_vals and isinstance(lit_vals[0], bool):
            return "bool"
        if lit_vals and isinstance(lit_vals[0], int):
            return "int"
        return "str"
    name = getattr(core, '__name__', None)
    if name:
        if name.lower() in ('str', 'string'):
            return "str"
        if name.lower() in ('int', 'integer'):
            return "int"
        if name.lower() == 'float':
            return "float"
        if name.lower() == 'bool':
            return "bool"
    return "str"


def _pydantic_field_to_record(field_info: Any) -> dict:
    core = _core_type(field_info.annotation)

    # Choices from Literal
    choices = None
    if get_origin(core) is Literal:
        choices = [str(a) for a in get_args(core)]

    type_label = _annotation_to_type(field_info.annotation)

    # Default value
    default = field_info.default
    is_undef = (_Undefined is not None and default is _Undefined)
    if is_undef:
        if field_info.default_factory is not None:
            try:
                default = field_info.default_factory()
            except Exception:
                default = None
        else:
            default = None

    # Status from json_schema_extra
    status = None
    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict):
        status = extra.get("status")

    rec: dict[str, Any] = {
        "type": type_label,
        "default_value": _serialise(default),
        "value": None,
        "description": field_info.description or "",
    }
    if choices:
        rec["choices"] = choices
    if status:
        rec["status"] = status
    return rec


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for TRT-LLM deployment
# ---------------------------------------------------------------------------
# Move args between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
    "model", "dtype", "tp_size", "pp_size",
    "max_beam_width", "host", "port",
    "served_model_name", "api_key",
    "max_batch_size", "max_num_tokens",
    "kv_cache_free_gpu_memory_fraction",
    "trust_remote_code", "quantization", "backend",
]

_UI_ADVANCED = [
    "kv_cache_type", "enable_chunked_context",
    "multi_block_mode", "tokenizer",
    "speculative_decoding_mode", "num_draft_tokens",
    "lora_dir", "max_lora_rank",
    "max_seq_len", "enable_prefix_caching",
    "request_timeout", "max_queue_size",
    "executor_type", "scheduling_policy",
    "num_postprocess_workers", "batching_type",
    "enable_kv_cache_reuse", "max_tokens_in_paged_kv_cache",
    "enable_streaming", "pipeline_parallel_size",
]


# ---------------------------------------------------------------------------
# Build JSON
# ---------------------------------------------------------------------------
def build_json() -> dict:
    args: dict[str, dict] = {}

    # 1. trtllm-serve Click CLI params
    for param in _serve_cmd.params:
        record = _click_param_to_record(param)
        record["module"] = "serve"
        record["config_class"] = "ServeCommand"
        record["ui"] = False
        record["aic"] = False
        args[param.name] = record

    # 2. BaseLlmArgs — fields common to all backends
    base_names: set[str] = set(BaseLlmArgs.model_fields)
    for name, fi in BaseLlmArgs.model_fields.items():
        record = _pydantic_field_to_record(fi)
        record["module"] = "base"
        record["config_class"] = "BaseLlmArgs"
        record["ui"] = False
        record["aic"] = False
        args[name] = record

    # 3. TrtLlmArgs — TensorRT-backend-only fields
    for name, fi in TrtLlmArgs.model_fields.items():
        if name not in base_names:
            record = _pydantic_field_to_record(fi)
            record["module"] = "tensorrt"
            record["config_class"] = "TrtLlmArgs"
            record["ui"] = False
            record["aic"] = False
            args[name] = record

    # 4. TorchLlmArgs — PyTorch-backend-only fields
    for name, fi in TorchLlmArgs.model_fields.items():
        if name not in base_names:
            record = _pydantic_field_to_record(fi)
            record["module"] = "pytorch"
            record["config_class"] = "TorchLlmArgs"
            record["ui"] = False
            record["aic"] = False
            args[name] = record

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
        "engine": "tensorrt_llm",
        "version": TRTLLM_VERSION,
        "date": date.today().isoformat(),
        "source": (
            "tensorrt_llm.commands.serve (Click) + "
            "tensorrt_llm.llmapi.llm_args (Pydantic)"
        ),
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
        default="trtllm_args.json",
        help="Output file path (default: trtllm_args.json)",
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
        f"Written {ns.out!r}  —  tensorrt_llm {data['version']}  "
        f"—  {len(modules)} modules  {len(args)} total args"
    )
    for mod, count in sorted(modules.items()):
        print(f"  {mod:10s}  {count} args")

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
