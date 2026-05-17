"""Apply product-facing UI metadata to generated Dynamo runtime JSONs.

This script never removes records. It rewrites the ``ui`` tier, adds stable
``display_name`` labels, and orders records by frequency-of-use buckets for
each engine catalog. Prefill/decode files are intentionally left to callers to
create and curate separately.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


META_KEYS = {"engine", "version", "date", "source"}
TIERS = ("primary", "advanced", "less_frequent")
DISPLAY_ACRONYMS = {
    "aic": "AIC",
    "api": "API",
    "cpu": "CPU",
    "cuda": "CUDA",
    "gdr": "GDR",
    "gpu": "GPU",
    "ib": "IB",
    "kv": "KV",
    "lora": "LoRA",
    "mm": "MM",
    "nccl": "NCCL",
    "nixl": "NIXL",
    "nvls": "NVLS",
    "ssl": "SSL",
    "tls": "TLS",
    "tp": "TP",
    "trt": "TRT",
    "ucx": "UCX",
    "vllm": "vLLM",
}


BASE_RULES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("vllm", "args"): {
        "primary": [
            "enable_lora",
            "lora_modules",
            "tool_call_parser",
            "reasoning_parser",
        ],
        "advanced": [
            "kv_transfer_config",
            "kv_events_config",
            "enable_prefix_caching",
            "chat_template",
            "chat_template_content_format",
            "max_loras",
            "max_lora_rank",
            "lora_dtype",
            "limit_mm_per_prompt",
            "media_io_kwargs",
            "video_pruning_rate",
            "speculative_config",
            "enable_prompt_embeds",
            "enable_log_requests",
            "uvicorn_log_level",
        ],
    },
    ("vllm", "envs"): {
        "primary": [],
        "advanced": [
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
            "VLLM_IMAGE_FETCH_TIMEOUT",
            "VLLM_VIDEO_FETCH_TIMEOUT",
            "VLLM_AUDIO_FETCH_TIMEOUT",
            "VLLM_MEDIA_FETCH_MAX_RETRIES",
            "VLLM_MEDIA_LOADING_THREAD_COUNT",
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "VLLM_NIXL_SIDE_CHANNEL_PORT",
            "VLLM_MOONCAKE_BOOTSTRAP_PORT",
            "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES",
            "VLLM_LOGGING_LEVEL",
            "VLLM_ENGINE_READY_TIMEOUT_S",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        ],
    },
    ("vllm_omni", "args"): {
        "primary": [
            "enable_multimodal",
            "media_io_kwargs",
            "multimodal_embedding_cache_capacity_gb",
        ],
        "advanced": [
            "kv_transfer_config",
            "kv_events_config",
            "video_pruning_rate",
            "enable_lora",
            "lora_modules",
            "max_loras",
            "limit_mm_per_prompt",
            "dyn_tool_call_parser",
            "dyn_reasoning_parser",
            "tool_call_parser",
            "reasoning_parser",
            "chat_template",
            "chat_template_content_format",
            "enable_mm_embeds",
            "mm_encoder_only",
            "mm_tensor_ipc",
            "disable_chunked_mm_input",
        ],
    },
    ("vllm_omni", "envs"): {
        "primary": [
            "DYN_MULTIMODAL_EMBEDDING_CACHE_GB",
            "DYN_MM_VIDEO_NUM_FRAMES",
        ],
        "advanced": [
            "DYN_VLLM_EMBEDDING_TRANSFER_MODE",
            "DYN_TOOL_CALL_PARSER",
            "DYN_REASONING_PARSER",
            "VLLM_IMAGE_FETCH_TIMEOUT",
            "VLLM_VIDEO_FETCH_TIMEOUT",
            "VLLM_AUDIO_FETCH_TIMEOUT",
            "VLLM_MEDIA_FETCH_MAX_RETRIES",
            "VLLM_MEDIA_LOADING_THREAD_COUNT",
            "VLLM_MM_HASHER_ALGORITHM",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "VLLM_NIXL_SIDE_CHANNEL_PORT",
        ],
    },
    ("sglang", "args"): {
        "primary": [
            "enable_lora",
            "lora_paths",
            "enable_multimodal",
        ],
        "advanced": [
            "embedding_worker",
            "multimodal_encode_worker",
            "multimodal_worker",
            "image_diffusion_worker",
            "video_generation_worker",
            "disagg_config",
            "disagg_config_key",
            "dyn_tool_call_parser",
            "dyn_reasoning_parser",
            "custom_jinja_template",
            "kv_events_config",
            "max_loras_per_batch",
            "lora_backend",
            "chat_template",
            "reasoning_parser",
            "tool_call_parser",
            "tool_server",
            "enable_cache_report",
        ],
    },
    ("sglang", "envs"): {
        "primary": [
            "DYN_SGL_EMBEDDING_WORKER",
            "DYN_SGL_MULTIMODAL_ENCODE_WORKER",
            "DYN_SGL_MULTIMODAL_WORKER",
        ],
        "advanced": [
            "DYN_SGL_IMAGE_DIFFUSION_WORKER",
            "DYN_SGL_VIDEO_GENERATION_WORKER",
            "DYN_SGL_DISAGG_CONFIG",
            "DYN_SGL_DISAGG_CONFIG_KEY",
            "DYN_TOOL_CALL_PARSER",
            "DYN_REASONING_PARSER",
            "DYN_CUSTOM_JINJA_TEMPLATE",
            "SGLANG_FORWARD_UNKNOWN_TOOLS",
            "SGLANG_TOOL_STRICT_LEVEL",
            "SGLANG_VLM_CACHE_SIZE_MB",
            "SGLANG_MM_FEATURE_CACHE_MB",
        ],
    },
    ("tensorrt_llm", "args"): {
        "primary": [
            "backend",
            "enable_lora",
            "lora_config",
        ],
        "advanced": [
            "server_role",
            "disagg_cluster_uri",
            "kv_connector_config",
            "cache_transceiver_config",
            "kv_cache_config",
            "guided_decoding_backend",
            "reasoning_parser",
            "tool_parser",
            "custom_tokenizer",
            "media_io_kwargs",
            "video_pruning_rate",
            "speculative_config",
            "batching_type",
            "scheduler_config",
            "decoding_config",
        ],
    },
    ("tensorrt_llm", "envs"): {
        "primary": [],
        "advanced": [
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
        ],
    },
    ("frontend", "args"): {
        "primary": [
            "router_mode",
            "router_min_initial_workers",
            "serve_indexer",
            "enforce_disagg",
            "router_kv_events",
            "router_prefill_load_model",
            "dyn_chat_processor",
        ],
        "advanced": [
            "request_plane",
            "event_plane",
            "enable_anthropic_api",
            "strip_anthropic_preamble",
            "enable_streaming_tool_dispatch",
            "enable_streaming_reasoning_dispatch",
            "exclude_tools_when_tool_choice_none",
            "dyn_debug_perf",
            "dyn_preprocess_workers",
            "router_replica_sync",
            "router_track_active_blocks",
            "router_track_output_blocks",
            "router_assume_kv_reuse",
            "router_track_prefill_tokens",
            "router_queue_policy",
            "router_event_threads",
        ],
    },
    ("frontend", "envs"): {
        "primary": [
            "DYN_ROUTER_MODE",
            "DYN_ROUTER_MIN_INITIAL_WORKERS",
            "DYN_SERVE_INDEXER",
            "DYN_ENFORCE_DISAGG",
            "DYN_ROUTER_KV_EVENTS",
            "DYN_ROUTER_PREFILL_LOAD_MODEL",
            "DYN_CHAT_PROCESSOR",
        ],
        "advanced": [
            "DYN_REQUEST_PLANE",
            "DYN_EVENT_PLANE",
            "DYN_ENABLE_ANTHROPIC_API",
            "DYN_ENABLE_STREAMING_TOOL_DISPATCH",
            "DYN_ENABLE_STREAMING_REASONING_DISPATCH",
            "DYN_DEBUG_PERF",
            "DYN_PREPROCESS_WORKERS",
            "DYN_ROUTER_REPLICA_SYNC",
            "DYN_ROUTER_QUEUE_POLICY",
        ],
    },
}


def _load(path: Path) -> OrderedDict[str, Any]:
    return json.loads(path.read_text(), object_pairs_hook=OrderedDict)


def _write(path: Path, data: OrderedDict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _display_name(key: str) -> str:
    words = [word for word in key.replace("-", "_").split("_") if word]
    return " ".join(DISPLAY_ACRONYMS.get(word.lower(), word.capitalize()) for word in words)


def _kind(path: Path) -> str | None:
    if "_args_" in path.name:
        return "args"
    if "_envs_" in path.name:
        return "envs"
    return None


def _is_manual_role_catalog(path: Path) -> bool:
    return "_prefill_" in path.name or "_decode_" in path.name


def _rule_for(engine: str, kind: str) -> dict[str, list[str]]:
    return BASE_RULES.get((engine, kind), {})


def _apply_ui(
    data: OrderedDict[str, Any],
    rules: dict[str, list[str]],
) -> OrderedDict[str, Any]:
    primary = [key for key in rules.get("primary", []) if key in data]
    advanced = [key for key in rules.get("advanced", []) if key in data and key not in primary]
    primary_set = set(primary)
    advanced_set = set(advanced)

    for key, record in data.items():
        if key in META_KEYS:
            continue
        if not isinstance(record, dict):
            continue
        if key in primary_set:
            record["ui"] = "primary"
        elif key in advanced_set:
            record["ui"] = "advanced"
        else:
            record["ui"] = "less_frequent"
    record["display_name"] = str(record.get("display_name") or record.get("display_value") or _display_name(key))

    out: OrderedDict[str, Any] = OrderedDict()
    for key in data:
        if key in META_KEYS:
            out[key] = data[key]
    for key in primary:
        out[key] = data[key]
    for key in advanced:
        out[key] = data[key]
    for key, record in data.items():
        if key in META_KEYS or key in primary_set or key in advanced_set:
            continue
        out[key] = record
    return out


def _iter_base_catalogs(root: Path, selected_engines: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob("*/*.json")):
        if _is_manual_role_catalog(path):
            continue
        kind = _kind(path)
        if kind is None:
            continue
        data = _load(path)
        engine = str(data.get("engine") or "")
        if selected_engines and engine not in selected_engines:
            continue
        paths.append(path)
    return paths


def rebucket(root: Path, selected_engines: set[str] | None = None) -> None:
    for path in _iter_base_catalogs(root, selected_engines):
        kind = _kind(path)
        if kind is None:
            continue

        data = _load(path)
        engine = str(data.get("engine") or "")
        base_rules = _rule_for(engine, kind)
        base_view = _apply_ui(data, base_rules)
        _write(path, base_view)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "dynamo_runtime",
        help="dynamo_runtime root containing component subdirectories.",
    )
    cli.add_argument(
        "--engine",
        action="append",
        help="Only rebucket this engine. Repeatable.",
    )
    ns = cli.parse_args()

    selected = set(ns.engine) if ns.engine else None
    rebucket(ns.root, selected)


if __name__ == "__main__":
    main()
