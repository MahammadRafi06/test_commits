"""
Merge Dynamo wrapper args into native engine args/envs JSONs.

The wrapper JSON should be produced by ``gen_dynamo_wrapper_args.py`` inside the
same Dynamo runtime image. Wrapper records intentionally take precedence when a
destination name overlaps with native engine records, because Kubernetes runs
``python -m dynamo.<backend>``, not the engine's standalone serve command.

Usage:
    python dynamo/merge_dynamo_wrapper.py \
        --wrapper-json /out/vllm_dynamo_wrapper.json \
        --dynamo-envs-json /out/vllm_dynamo_envs.json \
        --args-json /out/vllm_args.json \
        --envs-json /out/vllm_envs.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


META_KEYS = {"engine", "version", "date", "source"}


def _load(path: Path) -> OrderedDict[str, Any]:
    return json.loads(path.read_text(), object_pairs_hook=OrderedDict)


def _write(path: Path, data: OrderedDict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _wrapper_records(wrapper: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict((k, v) for k, v in wrapper.items() if k not in META_KEYS)


def _merge_args(
    args_data: OrderedDict[str, Any],
    wrapper_data: OrderedDict[str, Any],
) -> OrderedDict[str, Any]:
    wrapper = _wrapper_records(wrapper_data)
    out = OrderedDict((k, args_data[k]) for k in args_data if k in META_KEYS)

    source = out.get("source") or ""
    if "Dynamo wrapper ArgGroups" not in source:
        out["source"] = (source + " + Dynamo wrapper ArgGroups").strip(" +")

    for key, rec in wrapper.items():
        out[key] = rec
    for key, rec in args_data.items():
        if key not in META_KEYS and key not in wrapper:
            out[key] = rec
    return out


def _env_record_from_arg(arg_key: str, arg_rec: dict[str, Any]) -> OrderedDict[str, Any] | None:
    env_var = arg_rec.get("env_var")
    if not env_var:
        return None

    record: OrderedDict[str, Any] = OrderedDict()
    record["category"] = arg_rec.get("module", "dynamo")
    record["type"] = "bool" if arg_rec.get("flag") else _env_type(arg_rec.get("type"))
    record["default_value"] = arg_rec.get("default_value")
    record["value"] = None
    record["description"] = arg_rec.get("description", "")
    if arg_rec.get("flag"):
        record["true_arg"] = arg_rec.get("true_arg")
        record["false_arg"] = arg_rec.get("false_arg")
    else:
        record["arg"] = arg_rec.get("arg")
    record["arg_key"] = arg_key
    record["ui"] = arg_rec.get("ui", "less_frequent")
    record["aic"] = arg_rec.get("aic", False)

    for key in (
        "choices",
        "multiple",
        "status",
        "deprecated_alias",
    ):
        if key in arg_rec:
            record[key] = arg_rec[key]
    return record


def _env_type(type_label: Any) -> str:
    if isinstance(type_label, str) and type_label.startswith("List["):
        return "List[str]"
    if type_label in {"int", "float", "bool"}:
        return str(type_label)
    return "str"


def _merge_envs(
    envs_data: OrderedDict[str, Any],
    wrapper_data: OrderedDict[str, Any],
    scanned_envs_data: OrderedDict[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    wrapper_envs: OrderedDict[str, Any] = OrderedDict()
    for key, rec in _wrapper_records(wrapper_data).items():
        env_rec = _env_record_from_arg(key, rec)
        if env_rec is not None:
            wrapper_envs[rec["env_var"]] = env_rec

    out = OrderedDict((k, envs_data[k]) for k in envs_data if k in META_KEYS)
    source = out.get("source") or ""
    if "Dynamo wrapper ArgGroups" not in source:
        out["source"] = (source + " + Dynamo wrapper ArgGroups").strip(" +")

    for key, rec in wrapper_envs.items():
        out[key] = rec
    if scanned_envs_data:
        for key, rec in scanned_envs_data.items():
            if key not in META_KEYS and key not in out:
                out[key] = rec
    for key, rec in envs_data.items():
        if key not in META_KEYS and key not in out:
            out[key] = rec
    return out


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--wrapper-json", type=Path, required=True)
    cli.add_argument("--dynamo-envs-json", type=Path)
    cli.add_argument("--args-json", type=Path, required=True)
    cli.add_argument("--envs-json", type=Path, required=True)
    ns = cli.parse_args()

    wrapper_data = _load(ns.wrapper_json)
    scanned_envs_data = _load(ns.dynamo_envs_json) if ns.dynamo_envs_json else None
    args_data = _load(ns.args_json)
    envs_data = _load(ns.envs_json)

    _write(ns.args_json, _merge_args(args_data, wrapper_data))
    _write(ns.envs_json, _merge_envs(envs_data, wrapper_data, scanned_envs_data))

    wrapper_count = len(_wrapper_records(wrapper_data))
    env_count = sum(
        1 for rec in _wrapper_records(wrapper_data).values() if rec.get("env_var")
    )
    scanned_count = (
        len([k for k in scanned_envs_data if k not in META_KEYS])
        if scanned_envs_data
        else 0
    )
    print(
        f"Merged {wrapper_count} Dynamo wrapper args, {env_count} wrapper env vars, "
        f"and {scanned_count} scanned env vars "
        f"into {ns.args_json} / {ns.envs_json}"
    )


if __name__ == "__main__":
    main()
