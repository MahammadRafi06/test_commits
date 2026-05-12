"""
Apply Dynamo-specific SGLang wrapper args/envs to generated SGLang configs.

The native SGLang generators intentionally introspect SGLang itself. Dynamo
runtime images launch SGLang through ``python -m dynamo.sglang``, which adds a
small wrapper argument layer backed by DYN_* environment variables. Run this
after generating configs inside an NVIDIA Dynamo SGLang runtime image.

Usage:
    python apply_dynamo_overlay.py \
        --args-json ../dynamo_runtime/sglang/sglang_args_0.5.10.post1.json \
        --envs-json ../dynamo_runtime/sglang/sglang_envs_0.5.10.post1.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


META_KEYS = {"engine", "version", "date", "source"}

DYNAMO_SGLANG_OVERLAY: list[dict[str, Any]] = [
    {
        "key": "endpoint",
        "arg": "--endpoint",
        "env_var": "DYN_ENDPOINT",
        "type": "str",
        "default_value": None,
        "default_note": "Auto-generated",
        "description": "Dynamo endpoint in dyn://namespace.component.endpoint format.",
        "ui": "less_frequent",
    },
    {
        "key": "use_sglang_tokenizer",
        "arg": "--use-sglang-tokenizer",
        "env_var": "DYN_SGL_USE_TOKENIZER",
        "type": "bool",
        "default_value": False,
        "description": (
            "[Deprecated] Use --dyn-chat-processor sglang on the frontend instead."
        ),
        "status": "deprecated",
        "ui": "less_frequent",
    },
    {
        "key": "dyn_tool_call_parser",
        "arg": "--dyn-tool-call-parser",
        "env_var": "DYN_TOOL_CALL_PARSER",
        "type": "str",
        "default_value": None,
        "description": "Tool call parser (overrides SGLang's --tool-call-parser).",
        "ui": "advanced",
    },
    {
        "key": "dyn_reasoning_parser",
        "arg": "--dyn-reasoning-parser",
        "env_var": "DYN_REASONING_PARSER",
        "type": "str",
        "default_value": None,
        "description": "Reasoning parser for chain-of-thought models.",
        "ui": "advanced",
    },
    {
        "key": "custom_jinja_template",
        "arg": "--custom-jinja-template",
        "env_var": "DYN_CUSTOM_JINJA_TEMPLATE",
        "type": "path",
        "default_value": None,
        "description": "Custom chat template path.",
        "conflicts_with": ["use_sglang_tokenizer"],
        "ui": "advanced",
    },
    {
        "key": "embedding_worker",
        "arg": "--embedding-worker",
        "env_var": "DYN_SGL_EMBEDDING_WORKER",
        "type": "bool",
        "default_value": False,
        "description": "Run as embedding worker (also sets SGLang's --is-embedding).",
        "native_effects": ["--is-embedding"],
        "ui": "advanced",
    },
    {
        "key": "multimodal_encode_worker",
        "arg": "--multimodal-encode-worker",
        "env_var": "DYN_SGL_MULTIMODAL_ENCODE_WORKER",
        "type": "bool",
        "default_value": False,
        "description": "Run as multimodal encode worker (frontend-facing).",
        "ui": "advanced",
    },
    {
        "key": "multimodal_worker",
        "arg": "--multimodal-worker",
        "env_var": "DYN_SGL_MULTIMODAL_WORKER",
        "type": "bool",
        "default_value": False,
        "description": "Run as multimodal LLM worker.",
        "ui": "advanced",
    },
    {
        "key": "image_diffusion_worker",
        "arg": "--image-diffusion-worker",
        "env_var": "DYN_SGL_IMAGE_DIFFUSION_WORKER",
        "type": "bool",
        "default_value": False,
        "description": "Run as image diffusion worker.",
        "ui": "advanced",
    },
    {
        "key": "video_generation_worker",
        "arg": "--video-generation-worker",
        "env_var": "DYN_SGL_VIDEO_GENERATION_WORKER",
        "type": "bool",
        "default_value": False,
        "description": "Run as video generation worker.",
        "ui": "advanced",
    },
    {
        "key": "disagg_config",
        "arg": "--disagg-config",
        "env_var": "DYN_SGL_DISAGG_CONFIG",
        "type": "path",
        "default_value": None,
        "description": "Path to YAML disaggregation config file.",
        "requires": ["disagg_config_key"],
        "ui": "advanced",
    },
    {
        "key": "disagg_config_key",
        "arg": "--disagg-config-key",
        "env_var": "DYN_SGL_DISAGG_CONFIG_KEY",
        "type": "str",
        "default_value": None,
        "description": (
            "Key to select from disaggregation config, for example prefill or decode."
        ),
        "requires": ["disagg_config"],
        "ui": "advanced",
    },
]


def _load_json(path: Path) -> OrderedDict[str, Any]:
    return json.loads(path.read_text(), object_pairs_hook=OrderedDict)


def _write_json(path: Path, data: OrderedDict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _with_overlay(
    data: OrderedDict[str, Any],
    overlay: OrderedDict[str, Any],
) -> OrderedDict[str, Any]:
    out = OrderedDict((k, data[k]) for k in data if k in META_KEYS)
    source = out.get("source") or ""
    if "Dynamo wrapper overlay" not in source:
        out["source"] = (source + " + Dynamo wrapper overlay").strip(" +")

    for key, rec in overlay.items():
        out[key] = rec
    for key, rec in data.items():
        if key not in META_KEYS and key not in overlay:
            out[key] = rec
    return out


def _copy_extra_fields(item: dict[str, Any], rec: OrderedDict[str, Any]) -> None:
    for key in (
        "status",
        "default_note",
        "conflicts_with",
        "requires",
        "native_effects",
    ):
        if key in item:
            rec[key] = item[key]


def _args_overlay() -> OrderedDict[str, Any]:
    overlay: OrderedDict[str, Any] = OrderedDict()
    for item in DYNAMO_SGLANG_OVERLAY:
        is_bool = item["type"] == "bool"
        rec: OrderedDict[str, Any] = OrderedDict()
        rec["type"] = item["type"]
        rec["flag"] = is_bool
        if is_bool:
            rec["default_value"] = item["default_value"]
            rec["value"] = None
            rec["true_arg"] = item["arg"]
            rec["false_arg"] = None
        else:
            rec["arg"] = item["arg"]
            rec["default_value"] = item["default_value"]
            rec["value"] = None
        rec["env_var"] = item["env_var"]
        rec["description"] = item["description"]
        rec["module"] = "dynamo"
        rec["config_class"] = "DynamoSGLangArgs"
        rec["ui"] = item["ui"]
        rec["aic"] = False
        _copy_extra_fields(item, rec)
        overlay[item["key"]] = rec
    return overlay


def _envs_overlay() -> OrderedDict[str, Any]:
    overlay: OrderedDict[str, Any] = OrderedDict()
    for item in DYNAMO_SGLANG_OVERLAY:
        rec: OrderedDict[str, Any] = OrderedDict()
        rec["category"] = "dynamo"
        rec["type"] = "bool" if item["type"] == "bool" else "str"
        rec["default_value"] = item["default_value"]
        rec["value"] = None
        rec["description"] = item["description"]
        rec["arg"] = item["arg"]
        rec["ui"] = item["ui"]
        rec["aic"] = False
        _copy_extra_fields(item, rec)
        overlay[item["env_var"]] = rec
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--envs-json", type=Path, required=True)
    ns = parser.parse_args()

    args_data = _load_json(ns.args_json)
    envs_data = _load_json(ns.envs_json)

    _write_json(ns.args_json, _with_overlay(args_data, _args_overlay()))
    _write_json(ns.envs_json, _with_overlay(envs_data, _envs_overlay()))

    print(
        f"Applied {len(DYNAMO_SGLANG_OVERLAY)} Dynamo SGLang overlay records "
        f"to {ns.args_json} and {ns.envs_json}"
    )


if __name__ == "__main__":
    main()
