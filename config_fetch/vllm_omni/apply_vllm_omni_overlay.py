"""Apply the vLLM-Omni multimodal overlay to generated vLLM configs.

The Dynamo Omni recipes still launch ``python -m dynamo.vllm``. This overlay
keeps the base config image-derived from vLLM, then marks the backend as
``vllm_omni`` and guarantees the Omni/multimodal knobs used by the recipes are
present in the generated args/envs JSONs.

Usage:
    python vllm_omni/apply_vllm_omni_overlay.py \
        --args-json /out/vllm_omni_args.json \
        --envs-json /out/vllm_omni_envs.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


META_KEYS = {"engine", "version", "date", "source"}
MODULE = "vllm_omni"
CONFIG_CLASS = "VllmOmniMultimodalConfig"


def _load(path: Path) -> OrderedDict[str, Any]:
    return json.loads(path.read_text(), object_pairs_hook=OrderedDict)


def _write(path: Path, data: OrderedDict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _arg_record(
    *,
    type_: str,
    flag: bool,
    default: Any,
    description: str,
    arg: str | None = None,
    true_arg: str | None = None,
    false_arg: str | None = None,
    env_var: str | None = None,
    choices: list[Any] | None = None,
    ui: str = "advanced",
) -> OrderedDict[str, Any]:
    rec: OrderedDict[str, Any] = OrderedDict()
    rec["type"] = type_
    rec["flag"] = flag
    if flag:
        rec["true_arg"] = true_arg
        rec["false_arg"] = false_arg
    else:
        rec["arg"] = arg
    rec["default_value"] = default
    rec["value"] = None
    rec["description"] = description
    if env_var:
        rec["env_var"] = env_var
    if choices:
        rec["choices"] = choices
    rec["module"] = MODULE
    rec["config_class"] = CONFIG_CLASS
    rec["ui"] = ui
    rec["aic"] = False
    return rec


def _env_record(
    *,
    type_: str,
    default: Any,
    description: str,
    arg_key: str | None = None,
    arg: str | None = None,
    true_arg: str | None = None,
    false_arg: str | None = None,
    choices: list[Any] | None = None,
    ui: str = "advanced",
) -> OrderedDict[str, Any]:
    rec: OrderedDict[str, Any] = OrderedDict()
    rec["category"] = MODULE
    rec["type"] = type_
    rec["default_value"] = default
    rec["value"] = None
    rec["description"] = description
    if arg:
        rec["arg"] = arg
    if true_arg is not None or false_arg is not None:
        rec["true_arg"] = true_arg
        rec["false_arg"] = false_arg
    if arg_key:
        rec["arg_key"] = arg_key
    rec["ui"] = ui
    rec["aic"] = False
    if choices:
        rec["choices"] = choices
    return rec


ARG_OVERLAY: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict(
    [
        (
            "enable_multimodal",
            _arg_record(
                type_="bool",
                flag=True,
                true_arg="--enable-multimodal",
                false_arg=None,
                default=False,
                description=(
                    "Enable Dynamo/vLLM multimodal request handling for image, "
                    "video, or audio-capable models."
                ),
                ui="primary",
            ),
        ),
        (
            "media_io_kwargs",
            _arg_record(
                type_="str",
                flag=False,
                arg="--media-io-kwargs",
                default={},
                description=(
                    "JSON object with media loader options by modality, for example "
                    '{"video": {"num_frames": 512, "fps": 1}}.'
                ),
                ui="primary",
            ),
        ),
        (
            "video_pruning_rate",
            _arg_record(
                type_="float",
                flag=False,
                arg="--video-pruning-rate",
                default=None,
                description=(
                    "Video token pruning rate for Omni/video models. Values are in "
                    "the range [0, 1)."
                ),
                ui="advanced",
            ),
        ),
        (
            "multimodal_embedding_cache_capacity_gb",
            _arg_record(
                type_="float",
                flag=False,
                arg="--multimodal-embedding-cache-capacity-gb",
                env_var="DYN_MULTIMODAL_EMBEDDING_CACHE_GB",
                default=None,
                description="Capacity in GB for the multimodal embedding cache.",
                ui="primary",
            ),
        ),
        (
            "dyn_tool_call_parser",
            _arg_record(
                type_="str",
                flag=False,
                arg="--dyn-tool-call-parser",
                env_var="DYN_TOOL_CALL_PARSER",
                default=None,
                description=(
                    "Dynamo tool-call parser override used by Omni models such as "
                    "Nemotron Nano Omni."
                ),
                choices=["nemotron_nano", "deepseek_v4", "llama3_json", "none"],
                ui="advanced",
            ),
        ),
        (
            "dyn_reasoning_parser",
            _arg_record(
                type_="str",
                flag=False,
                arg="--dyn-reasoning-parser",
                env_var="DYN_REASONING_PARSER",
                default=None,
                description=(
                    "Dynamo reasoning parser override used by chain-of-thought "
                    "Omni models."
                ),
                choices=["nemotron_nano", "deepseek_v4", "none"],
                ui="advanced",
            ),
        ),
    ]
)


ENV_ONLY_OVERLAY: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict(
    [
        (
            "DYN_MM_VIDEO_NUM_FRAMES",
            _env_record(
                type_="int",
                default=None,
                description=(
                    "Frontend/worker agreement knob for the maximum number of "
                    "video frames used by multimodal preprocessing."
                ),
                ui="primary",
            ),
        ),
        (
            "DYN_VLLM_EMBEDDING_TRANSFER_MODE",
            _env_record(
                type_="str",
                default=None,
                description=(
                    "Dynamo vLLM multimodal embedding transfer mode. Recipes use "
                    "nixl-write when moving embeddings through NIXL."
                ),
                choices=["nixl-write"],
                ui="advanced",
            ),
        ),
    ]
)


def _promote_existing_arg(existing: OrderedDict[str, Any], overlay: OrderedDict[str, Any]) -> None:
    existing["module"] = overlay["module"]
    existing["config_class"] = overlay["config_class"]
    existing["ui"] = overlay["ui"]
    existing["aic"] = overlay["aic"]
    existing["description"] = existing.get("description") or overlay["description"]
    if overlay.get("env_var") and not existing.get("env_var"):
        existing["env_var"] = overlay["env_var"]


def _env_record_from_arg(arg_key: str, arg_rec: OrderedDict[str, Any]) -> OrderedDict[str, Any] | None:
    env_var = arg_rec.get("env_var")
    if not env_var:
        return None
    if arg_rec.get("flag"):
        return _env_record(
            type_=arg_rec.get("type", "bool"),
            default=arg_rec.get("default_value"),
            description=arg_rec.get("description", ""),
            arg_key=arg_key,
            true_arg=arg_rec.get("true_arg"),
            false_arg=arg_rec.get("false_arg"),
            choices=arg_rec.get("choices"),
            ui=arg_rec.get("ui", "advanced"),
        )
    return _env_record(
        type_=arg_rec.get("type", "str"),
        default=arg_rec.get("default_value"),
        description=arg_rec.get("description", ""),
        arg_key=arg_key,
        arg=arg_rec.get("arg"),
        choices=arg_rec.get("choices"),
        ui=arg_rec.get("ui", "advanced"),
    )


def apply_overlay(args_data: OrderedDict[str, Any], envs_data: OrderedDict[str, Any]) -> None:
    args_data["engine"] = "vllm_omni"
    envs_data["engine"] = "vllm_omni"

    for data in (args_data, envs_data):
        source = data.get("source") or ""
        if "vLLM Omni multimodal overlay" not in source:
            data["source"] = (source + " + vLLM Omni multimodal overlay").strip(" +")

    for key, overlay in ARG_OVERLAY.items():
        if key in args_data:
            _promote_existing_arg(args_data[key], overlay)
        else:
            args_data[key] = overlay

        env_rec = _env_record_from_arg(key, args_data[key])
        if env_rec and args_data[key]["env_var"] not in envs_data:
            envs_data[args_data[key]["env_var"]] = env_rec

    for env_var, record in ENV_ONLY_OVERLAY.items():
        if env_var not in envs_data:
            envs_data[env_var] = record


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--args-json", type=Path, required=True)
    cli.add_argument("--envs-json", type=Path, required=True)
    ns = cli.parse_args()

    args_data = _load(ns.args_json)
    envs_data = _load(ns.envs_json)
    apply_overlay(args_data, envs_data)
    _write(ns.args_json, args_data)
    _write(ns.envs_json, envs_data)

    args_count = len([key for key in args_data if key not in META_KEYS])
    envs_count = len([key for key in envs_data if key not in META_KEYS])
    print(
        f"Applied vLLM Omni overlay to {ns.args_json} / {ns.envs_json} "
        f"({args_count} args, {envs_count} envs)"
    )


if __name__ == "__main__":
    main()
