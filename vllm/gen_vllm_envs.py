"""
Generate vllm_envs.json by introspecting vllm.envs at runtime and
parsing its source to extract descriptions and choices.

Usage:
    python gen_vllm_envs.py [--out vllm_envs.json]

Requirements:
    Run inside a Python environment that has the target vLLM version installed.
    No GPU required.
"""

import argparse
import inspect
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Import vllm
# ---------------------------------------------------------------------------
try:
    import vllm
    VLLM_VERSION: str = vllm.__version__
except ImportError:
    sys.exit("vllm is not installed in this environment.")

import vllm.envs as _envs


# ---------------------------------------------------------------------------
# Category assignment by name prefix / keyword
# ---------------------------------------------------------------------------
_CATEGORIES: list[tuple[str, list[str]]] = [
    # Order matters — first match wins
    ("build",         ["MAX_JOBS", "NVCC_THREADS", "CMAKE_BUILD_TYPE", "VERBOSE",
                       "VLLM_USE_PRECOMPILED", "VLLM_SKIP_PRECOMPILED",
                       "VLLM_DOCKER_BUILD_CONTEXT", "VLLM_TARGET_DEVICE",
                       "VLLM_MAIN_CUDA_VERSION", "VLLM_FLOAT32_MATMUL_PRECISION",
                       "VLLM_BATCH_INVARIANT"]),
    ("cache",         ["VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT",
                       "VLLM_ASSETS_CACHE", "VLLM_ASSETS_CACHE_MODEL_CLEAN",
                       "VLLM_XLA_CACHE_PATH", "VLLM_MODEL_REDIRECT_PATH",
                       "VLLM_TUNED_CONFIG_FOLDER",
                       "VLLM_COMPILE_CACHE_SAVE_FORMAT",
                       "VLLM_USE_SIMPLE_KV_OFFLOAD",
                       "VLLM_WEIGHT_OFFLOADING_"]),
    ("networking",    ["VLLM_HOST_IP", "VLLM_PORT", "VLLM_RPC_BASE_PATH",
                       "VLLM_LOOPBACK_IP", "VLLM_RINGBUFFER_WARNING_INTERVAL",
                       "VLLM_API_KEY", "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
                       "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
                       "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
                       "VLLM_SERVER_DEV_MODE", "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
                       "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
                       "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE",
                       "VLLM_ALLOW_INSECURE_SERIALIZATION",
                       "VLLM_ENABLE_RESPONSES_API_STORE",
                       "VLLM_GPT_OSS_", "VLLM_TOOL_"]),
    ("storage",       ["S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
                       "S3_ENDPOINT_URL", "VLLM_CI_USE_S3"]),
    ("gpu",           ["CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LOCAL_RANK",
                       "VLLM_NCCL_SO_PATH", "VLLM_NCCL_INCLUDE_PATH",
                       "LD_LIBRARY_PATH", "VLLM_CUDART_SO_PATH",
                       "VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE",
                       "VLLM_USE_MODELSCOPE", "VLLM_SKIP_P2P_CHECK",
                       "VLLM_ALLREDUCE_", "VLLM_USE_NCCL_SYMM_MEM",
                       "VLLM_DISABLE_PYNCCL",
                       "VLLM_CUDA_COMPATIBILITY_", "VLLM_ENABLE_CUDA_COMPATIBILITY",
                       "VLLM_XPU_"]),
    ("rocm",          ["VLLM_ROCM_"]),
    ("engine",        ["VLLM_ENGINE_ITERATION_TIMEOUT_S",
                       "VLLM_ENGINE_READY_TIMEOUT_S",
                       "VLLM_PP_LAYER_PARTITION",
                       "VLLM_USE_FLASHINFER_SAMPLER",
                       "VLLM_WORKER_MULTIPROC_METHOD",
                       "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
                       "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
                       "VLLM_ENABLE_CUDAGRAPH_GC", "VLLM_KV_CACHE_LAYOUT",
                       "VLLM_ENABLE_V1_MULTIPROCESSING",
                       "VLLM_USE_V2_MODEL_RUNNER",
                       "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE",
                       "VLLM_MSGPACK_ZERO_COPY_THRESHOLD",
                       "VLLM_MQ_MAX_CHUNK_BYTES_MB",
                       "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
                       "VLLM_PROCESS_NAME_PREFIX",
                       "VLLM_DISABLE_LOG_LOGO",
                       "VLLM_SLEEP_WHEN_IDLE",
                       "VLLM_RPC_TIMEOUT", "VLLM_RPC_",
                       "VLLM_V1_", "VLLM_ENABLE_V1_",
                       "VLLM_DISABLE_REQUEST_ID_",
                       "VLLM_ENABLE_FLA_",
                       "VLLM_MAX_N_SEQUENCES",
                       "VLLM_MULTI_STREAM_GEMM_",
                       "VLLM_SSM_",
                       "VLLM_SYSTEM_START_",
                       "VLLM_USE_LAYERNAME",
                       "VLLM_USE_OINK_OPS"]),
    ("data_parallel", ["VLLM_DP_", "VLLM_RANDOMIZE_DP_DUMMY_INPUTS"]),
    ("compilation",   ["VLLM_USE_STANDALONE_COMPILE", "VLLM_ENABLE_PREGRAD_PASSES",
                       "VLLM_PATTERN_MATCH_DEBUG", "VLLM_DEBUG_DUMP_PATH",
                       "VLLM_USE_AOT_COMPILE", "VLLM_USE_BYTECODE_HOOK",
                       "VLLM_FORCE_AOT_LOAD", "VLLM_USE_MEGA_AOT_ARTIFACT",
                       "VLLM_DISABLE_COMPILE_CACHE",
                       "VLLM_ENABLE_INDUCTOR_", "VLLM_DISABLED_KERNELS",
                       "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
                       "VLLM_USE_FBGEMM"]),
    ("ray",           ["VLLM_USE_RAY_", "VLLM_RAY_"]),
    ("flashinfer",    ["VLLM_FLASHINFER_", "VLLM_HAS_FLASHINFER_CUBIN",
                       "VLLM_USE_FLASHINFER_MOE_"]),
    ("moe",           ["VLLM_MOE_", "VLLM_FUSED_MOE_", "VLLM_DEEPEP_",
                       "VLLM_DEEP_GEMM_", "VLLM_USE_DEEP_GEMM",
                       "VLLM_MAX_TOKENS_PER_EXPERT_",
                       "VLLM_ENABLE_MOE_DP_CHUNK", "VLLM_DBO_COMM_SMS",
                       "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING",
                       "VLLM_MLA_DISABLE", "VLLM_DEEPEPLL_",
                       "VLLM_SHARED_EXPERTS_STREAM_",
                       "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
                       "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
                       "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
                       "VLLM_ELASTIC_EP_", "VLLM_HUMMING_MOE_"]),
    ("quantization",  ["VLLM_MARLIN_", "VLLM_NVFP4_", "VLLM_MXFP4_",
                       "K_SCALE_CONSTANT", "Q_SCALE_CONSTANT", "V_SCALE_CONSTANT",
                       "VLLM_USE_TRITON_AWQ", "VLLM_USE_NVFP4_CT_EMULATIONS",
                       "VLLM_TEST_FORCE_FP8_MARLIN", "VLLM_COMPUTE_NANS_IN_LOGITS",
                       "VLLM_HUMMING_"]),
    ("kv_transfer",   ["VLLM_KV_EVENTS_", "VLLM_NIXL_", "VLLM_MOONCAKE_",
                       "VLLM_MORIIO_"]),
    ("lora",          ["VLLM_LORA_"]),
    ("cpu",           ["VLLM_CPU_", "VLLM_ZENTORCH_"]),
    ("multimodal",    ["VLLM_IMAGE_", "VLLM_VIDEO_", "VLLM_AUDIO_",
                       "VLLM_MEDIA_", "VLLM_MM_", "VLLM_MAX_AUDIO_",
                       "VLLM_VIDEO_LOADER_BACKEND", "VLLM_MEDIA_CONNECTOR"]),
    ("tpu_xla",       ["VLLM_XLA_", "VLLM_TPU_"]),
    ("structured_outputs", ["VLLM_XGRAMMAR_CACHE_MB", "VLLM_V1_USE_OUTLINES_CACHE",
                            "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT"]),
    ("profiling",     ["VLLM_NVTX_", "VLLM_CUSTOM_SCOPES_",
                       "VLLM_DEBUG_MFU_METRICS", "VLLM_DEBUG_WORKSPACE",
                       "VLLM_GC_DEBUG", "VLLM_MEMORY_PROFILER_"]),
    ("telemetry",     ["VLLM_USAGE_", "VLLM_NO_USAGE_STATS",
                       "VLLM_DO_NOT_TRACK", "DO_NOT_TRACK"]),
    ("logging",       ["VLLM_LOGGING_", "VLLM_LOG_", "VLLM_CONFIGURE_LOGGING",
                       "VLLM_TRACE_FUNCTION", "NO_COLOR"]),
    ("debug",         ["VLLM_TEST_", "VLLM_CI_"]),
    ("plugins",       ["VLLM_PLUGINS"]),
]


def _categorise(name: str) -> str:
    for cat, patterns in _CATEGORIES:
        for pat in patterns:
            if name == pat or name.startswith(pat):
                return cat
    return "general"


# ---------------------------------------------------------------------------
# Parse source to extract per-variable comments and choices
# ---------------------------------------------------------------------------
def _parse_source(source: str) -> dict[str, dict]:
    """
    Walk the environment_variables dict block in the source and for each key:
      - collect the comment lines immediately preceding it
      - detect env_with_choices calls to get choices list
    Returns {name: {"description": str, "choices": list|None}}
    """
    # Isolate the dict body
    m = re.search(
        r"environment_variables\s*:\s*dict\[.*?\]\s*=\s*\{(.+)",
        source,
        re.DOTALL,
    )
    if not m:
        return {}
    body = m.group(1)

    results: dict[str, dict] = {}
    lines = body.splitlines()
    pending_comments: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Accumulate comment lines (ignore section dividers like ===)
        if stripped.startswith("#"):
            comment_text = stripped.lstrip("# ").strip()
            if not re.match(r"^=+", comment_text):
                pending_comments.append(comment_text)
            i += 1
            continue

        # Detect a key entry: "VAR_NAME":
        key_match = re.match(r'\s*"([A-Z][A-Z0-9_]*)"\s*:', stripped)
        if key_match:
            name = key_match.group(1)

            # Collect the rest of this entry (until the next top-level key,
            # comment block, or closing brace).
            # Stop before comment lines so they remain available as descriptions
            # for the next entry.
            entry_lines = [stripped]
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if re.match(r'"[A-Z][A-Z0-9_]*"\s*:', next_stripped):
                    break
                if next_stripped.startswith("#"):
                    break
                if next_stripped in ("}", ""):
                    break
                entry_lines.append(next_stripped)
                j += 1
            entry_src = " ".join(entry_lines)

            # Extract choices from env_with_choices("NAME", default, [c1, c2, ...])
            choices = None
            ch_match = re.search(
                r'env_with_choices\s*\([^,]+,\s*[^,]+,\s*\[([^\]]+)\]',
                entry_src
            )
            if ch_match:
                raw = ch_match.group(1)
                choices = [
                    c.strip().strip('"').strip("'")
                    for c in raw.split(",")
                    if c.strip().strip('"\'')
                ]

            results[name] = {
                "description": " ".join(pending_comments),
                "choices": choices,
            }
            pending_comments = []
            i = j
            continue

        # Non-comment, non-key line resets pending comments
        if stripped and not stripped.startswith(("#", "}")):
            pending_comments = []
        i += 1

    return results


# ---------------------------------------------------------------------------
# Infer JSON type label and default by calling each lambda
# ---------------------------------------------------------------------------
def _get_default(name: str) -> Any:
    """Call the lambda for name, return (value, type_label)."""
    fn = _envs.environment_variables.get(name)
    if fn is None:
        return None, "value"

    # Temporarily clear the var so we get the hardcoded default, not a live value
    import os
    old = os.environ.pop(name, None)
    try:
        val = fn()
    except Exception:
        val = None
    finally:
        if old is not None:
            os.environ[name] = old

    if isinstance(val, bool):
        type_label = "bool"
    elif isinstance(val, int):
        type_label = "int"
    elif isinstance(val, float):
        type_label = "float"
    elif isinstance(val, list):
        inner = "int" if val and isinstance(val[0], int) and not isinstance(val[0], bool) else "str"
        type_label = f"List[{inner}]"
    else:
        type_label = "str"

    return val, type_label


# ---------------------------------------------------------------------------
# Most-frequently-used ordering for vLLM deployment
# ---------------------------------------------------------------------------
# Move vars between buckets to tune the UI experience.
# primary   → always visible in main form (max 15)
# advanced  → collapsible "Advanced" section (max 20)
# everything else → "less_frequent" (search / show-all only)
_UI_PRIMARY = [
    "CUDA_VISIBLE_DEVICES", "LOCAL_RANK",
    "VLLM_HOST_IP", "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_LOGGING_LEVEL", "VLLM_CONFIGURE_LOGGING",
    "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT",
    "VLLM_API_KEY", "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S", "VLLM_USE_MODELSCOPE",
    "VLLM_TARGET_DEVICE",
]

_UI_ADVANCED = [
    "VLLM_NCCL_SO_PATH", "VLLM_SKIP_P2P_CHECK",
    "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_DISABLE_LOG_LOGO",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "VLLM_PP_LAYER_PARTITION",
    "VLLM_ASSETS_CACHE", "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "VLLM_DISABLE_PYNCCL",
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", "VLLM_PORT",
    "VLLM_USE_V2_MODEL_RUNNER", "VLLM_RPC_TIMEOUT",
    "VLLM_MODEL_REDIRECT_PATH", "VLLM_MAIN_CUDA_VERSION",
    "VLLM_FLOAT32_MATMUL_PRECISION", "VLLM_USE_RAY_COMPILED_DAG",
    "VLLM_NCCL_INCLUDE_PATH", "VLLM_ENGINE_READY_TIMEOUT_S",
]


# ---------------------------------------------------------------------------
# Build the full JSON structure
# ---------------------------------------------------------------------------
def build_json() -> dict:
    source = inspect.getsource(_envs)
    meta = _parse_source(source)

    vars_: dict[str, dict] = {}

    for name in _envs.environment_variables:
        default, type_label = _get_default(name)
        info = meta.get(name, {"description": "", "choices": None})
        choices = info["choices"]

        if choices:
            type_label = "int" if all(str(c).lstrip('-').isdigit() for c in choices) else "str"

        record: dict[str, Any] = {
            "category": _categorise(name),
            "type": type_label,
            "default_value": _serialise(default),
            "value": None,
            "description": info["description"],
            "ui": False,
            "aic": False,
        }
        if choices:
            record["choices"] = choices

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
        "engine": "vllm",
        "version": VLLM_VERSION,
        "date": date.today().isoformat(),
        "source": "introspected from vllm.envs.environment_variables",
        **ordered,
    }


def _serialise(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialise(v) for v in val]
    return str(val)


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
    cli.add_argument("--out", default="vllm_envs.json",
                     help="Output file path (default: vllm_envs.json)")
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
    print(f"Written {ns.out!r}  —  vllm {data['version']}  "
          f"—  {len(cats)} categories  {len(vars_)} total vars")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:20s}  {count} vars")

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
