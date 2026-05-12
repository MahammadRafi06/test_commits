"""
Generate Dynamo wrapper args JSON for a backend runtime image.

Run this inside an NVIDIA Dynamo runtime image. It introspects the Dynamo
ArgGroups used by ``python -m dynamo.<backend>``. This captures wrapper-level
DYN_* / DYN_VLLM_* / DYN_TRTLLM_* / DYN_SGL_* settings that are not part of the
native engine's own parser.

Usage:
    python dynamo/gen_dynamo_wrapper_args.py --engine vllm --out wrapper.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


_ENV_RE = re.compile(r"\benv var:\s*([A-Z][A-Z0-9_]*)\b", re.IGNORECASE)
_DEPRECATED_RE = re.compile(r"\bdeprecating flag:\s*(--[A-Za-z0-9][\w-]*)")


def _serialise(value: Any, _depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise(v, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(_serialise(v, _depth + 1)) for v in value)
    if isinstance(value, dict):
        return {str(k): _serialise(v, _depth + 1) for k, v in value.items()}
    if _depth > 6:
        return str(value)
    try:
        return _serialise(value.value, _depth + 1)
    except AttributeError:
        return str(value)


def _clean_help(help_text: str) -> tuple[str, str | None, str | None]:
    text = (help_text or "").replace("%%", "%").strip()
    env_match = _ENV_RE.search(text)
    deprecated_match = _DEPRECATED_RE.search(text)

    env_var = env_match.group(1) if env_match else None
    deprecated_flag = deprecated_match.group(1) if deprecated_match else None

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("env var:"):
            continue
        if stripped.lower().startswith("deprecating flag:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip(), env_var, deprecated_flag


def _list_like(action: argparse.Action) -> bool:
    nargs = getattr(action, "nargs", None)
    if nargs in ("+", "*") or (isinstance(nargs, int) and nargs != 1):
        return True
    multi_action_types = tuple(
        cls
        for cls in (
            getattr(argparse, "_AppendAction", None),
            getattr(argparse, "_ExtendAction", None),
        )
        if cls is not None
    )
    return bool(multi_action_types and isinstance(action, multi_action_types))


def _infer_type(action: argparse.Action) -> str:
    if isinstance(action, argparse.BooleanOptionalAction):
        return "bool"
    if isinstance(
        action,
        (
            getattr(argparse, "_StoreTrueAction"),
            getattr(argparse, "_StoreFalseAction"),
        ),
    ):
        return "bool"

    is_list = _list_like(action)
    atype = getattr(action, "type", None)
    if atype is int:
        base = "int"
    elif atype is float:
        base = "float"
    elif atype is str:
        base = "str"
    elif atype is None:
        default = getattr(action, "default", None)
        if isinstance(default, bool):
            base = "bool"
        elif isinstance(default, int):
            base = "int"
        elif isinstance(default, float):
            base = "float"
        else:
            base = "str"
    else:
        name = getattr(atype, "__name__", "").lower()
        if "int" in name:
            base = "int"
        elif "float" in name:
            base = "float"
        else:
            base = "str"
    return f"List[{base}]" if is_list else base


def _canonical_option(action: argparse.Action) -> str:
    opts = [opt for opt in action.option_strings if opt.startswith("--")]
    pos = [opt for opt in opts if not opt.startswith("--no-")]
    return (pos or opts or [action.dest])[0]


def _group_to_module(title: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if key.startswith("dynamo_"):
        key = key[len("dynamo_"):]
    for suffix in ("_options", "_args", "_arguments"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]

    key = {
        "trt_llm": "trtllm",
        "tensorrt_llm": "trtllm",
        "tensorrtllm": "trtllm",
    }.get(key, key)

    if key in {"options", "args", "arguments"}:
        return "dynamo"
    if key == "runtime":
        return "dynamo_runtime"
    if key:
        return f"dynamo_{key}"
    return "dynamo"


def _action_to_record(action: argparse.Action, module: str, title: str) -> dict[str, Any]:
    description, env_var, deprecated_flag = _clean_help(action.help or "")
    choices = list(action.choices) if action.choices is not None else None
    default = _serialise(action.default)

    if isinstance(action, argparse.BooleanOptionalAction):
        true_arg = _canonical_option(action)
        false_arg = next(
            (opt for opt in action.option_strings if opt.startswith("--no-")),
            f"--no-{true_arg[2:]}" if true_arg.startswith("--") else None,
        )
        record: dict[str, Any] = {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else False,
            "value": None,
            "true_arg": true_arg,
            "false_arg": false_arg,
            "description": description,
        }
    elif isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
        record = {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else False,
            "value": None,
            "true_arg": _canonical_option(action),
            "false_arg": None,
            "description": description,
        }
    elif isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
        false_arg = next(
            (opt for opt in action.option_strings if opt.startswith("--no-")),
            _canonical_option(action),
        )
        record = {
            "type": "bool",
            "flag": True,
            "default_value": default if isinstance(default, bool) else True,
            "value": None,
            "true_arg": None,
            "false_arg": false_arg,
            "description": description,
        }
    else:
        record = {
            "type": _infer_type(action),
            "flag": False,
            "arg": _canonical_option(action),
            "default_value": default,
            "value": None,
            "description": description,
        }
        if _list_like(action):
            record["multiple"] = True

    if choices:
        record["choices"] = _serialise(choices)
    if env_var:
        record["env_var"] = env_var
    if deprecated_flag:
        record["deprecated_alias"] = deprecated_flag
    if "[Deprecated]" in description or "DEPRECATED" in description:
        record["status"] = "deprecated"

    record["module"] = module
    record["config_class"] = title
    record["ui"] = "less_frequent"
    record["aic"] = False
    return record


def _build_parser(engine: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )

    from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup

    DynamoRuntimeArgGroup().add_arguments(parser)

    if engine == "vllm":
        from dynamo.vllm.backend_args import DynamoVllmArgGroup

        DynamoVllmArgGroup().add_arguments(parser)
    elif engine == "sglang":
        from dynamo.sglang.backend_args import DynamoSGLangArgGroup

        DynamoSGLangArgGroup().add_arguments(parser)
    elif engine == "tensorrt_llm":
        from dynamo.trtllm.backend_args import DynamoTrtllmArgGroup

        DynamoTrtllmArgGroup().add_arguments(parser)
    else:
        raise ValueError(f"unsupported engine: {engine}")

    return parser


def build_json(engine: str) -> dict[str, Any]:
    parser = _build_parser(engine)
    records: dict[str, dict[str, Any]] = {}
    module_name = "trtllm" if engine == "tensorrt_llm" else engine

    for group in parser._action_groups:  # noqa: SLF001
        title = group.title or "Dynamo Options"
        module = _group_to_module(title)
        for action in group._group_actions:  # noqa: SLF001
            if action.dest in {"help", "version"}:
                continue
            if isinstance(action, argparse._VersionAction):  # noqa: SLF001
                continue
            if not action.option_strings:
                continue
            records[action.dest] = _action_to_record(action, module, title)

    return {
        "engine": engine,
        "version": "dynamo-wrapper",
        "date": date.today().isoformat(),
        "source": f"introspected from python -m dynamo.{module_name} wrapper ArgGroups",
        **records,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--engine", choices=["vllm", "sglang", "tensorrt_llm"], required=True)
    cli.add_argument("--out", type=Path, required=True)
    ns = cli.parse_args()

    data = build_json(ns.engine)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(data, indent=2) + "\n")

    count = len([k for k in data if k not in {"engine", "version", "date", "source"}])
    print(f"Written {ns.out!s} — {ns.engine} Dynamo wrapper — {count} args")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        sys.exit(f"Failed to import Dynamo wrapper modules: {exc}")
