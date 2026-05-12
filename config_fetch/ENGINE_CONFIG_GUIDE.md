# Dynamo Runtime Configuration Capture Guide

This folder generates versioned JSON catalogs for the arguments and environment
variables exposed by NVIDIA Dynamo runtime images. The catalogs are meant to be
complete machine-readable inventories. A profile UI, AIC, Helm, the operator,
and K8s patches can decide what to show or set later.

## Capture Contract

The generator contract is:

1. Capture every arg/env the selected runtime image exposes through the supported
   discovery paths.
2. Do not drop settings because they are rare, advanced, AIC-owned, operator
   managed, or ugly to show in a UI.
3. Keep generated records flat and structurally consistent across engines.
4. Keep `value` as `null`; the generated files are catalogs, not profiles.
5. Use `ui` and `aic` only as metadata. They must not affect whether a record is
   captured.
6. Do not emit legacy frontend-only fields such as `emit`, `managed_by`, or
   `reason`.

In product terms: these JSON files are the source catalog. The profile creation
experience should render a small curated subset, but search/show-all and AIC
still need the full catalog available.

## What "Everything" Means

For each runtime image, "everything" means all configuration discovered from:

| Source | Captured as | Notes |
| --- | --- | --- |
| Native engine CLI parser | args JSON | vLLM, SGLang, and TensorRT-LLM launch surfaces. |
| Native engine env registry/source | envs JSON | vLLM and SGLang have registries; TRT-LLM is source-scanned. |
| Dynamo backend wrapper ArgGroups | args JSON plus env records | Captures Dynamo `DYN_*` args added on top of the native engine. |
| Dynamo backend/common env-only source scan | envs JSON | Captures `DYN_*` / `DYNAMO_*` vars read directly without a CLI arg. |
| Dynamo frontend `--help` | frontend args/envs JSON | Captures frontend/router/AIC help output from `python -m dynamo.frontend --help`. |
| vLLM-Omni overlay | vllm_omni args/envs JSON | Adds multimodal knobs not discoverable as a separate backend parser. |

For backend engines, the conceptual source of truth is the Dynamo backend CLI:
`python -m dynamo.<backend> --help` (`dynamo.vllm`, `dynamo.sglang`, or
`dynamo.trtllm`). That help surface is expected to be the union of:

1. `Dynamo Runtime Options`: common Dynamo settings such as namespace,
   discovery, request/event plane, endpoint, parser, and chat-template knobs.
2. `Dynamo <Backend> Options`: backend-specific Dynamo settings such as
   disaggregation mode, tokenizer behavior, KV transfer, worker mode, headless
   mode, multimodal mode, or ModelExpress knobs.
3. Native engine options: the upstream engine parser/options accepted by the
   backend (`vLLM Engine Options`, SGLang server args, or TRT-LLM serve/API
   args).

The generator keeps those pieces separate while parsing because the native
engines expose richer structured metadata through their own parser/registry
APIs. The final JSON is still the merged Dynamo backend surface. The runner also
stores the raw backend `--help` output next to generated backend catalogs so a
new Dynamo release can be checked against the authoritative CLI text.

Platform-level configuration that is not read by the engine/Dynamo image source
is outside these engine catalogs. Examples include K8s resource placement,
secrets, model artifact mounts, and site-wide transport defaults such as NCCL,
UCX, EFA, or NVIDIA container runtime envs when they are injected by Helm,
operator defaults, or strategic merge patches. Keep those in the platform/profile
catalogs, not in engine arg/env catalogs, unless a runtime image actually exposes
or reads them.

## Supported Components

The Docker runner currently supports:

| Component | Image | Output directory |
| --- | --- | --- |
| `vllm` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>` | `config_fetch/dynamo_runtime/vllm/` |
| `vllm_omni` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>` by default | `config_fetch/dynamo_runtime/vllm_omni/` |
| `sglang` | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:<tag>` | `config_fetch/dynamo_runtime/sglang/` |
| `tensorrt_llm` | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:<tag>` | `config_fetch/dynamo_runtime/tensorrt_llm/` |
| `frontend` | vLLM runtime image by default | `config_fetch/dynamo_runtime/frontend/` |

`vllm-omni` is accepted as an alias for `vllm_omni`; `trtllm` is accepted as an
alias for `tensorrt_llm`.

## Running the Capture

Run from the repo root on a machine with Docker and NVIDIA Container Toolkit:

```bash
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1
```

Generate one component:

```bash
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine sglang
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm_omni
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine tensorrt_llm
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine frontend
```

The capture wrapper pulls the unique selected images first, then runs
`generate_dynamo_runtime_configs.sh` to start the respective containers, execute
the per-engine scripts, validate the results, and copy the JSON/help artifacts
back into the local output folder. If images are already present locally, skip
pulling:

```bash
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --skip-pull
```

Useful overrides:

```bash
# Change Docker flags passed before the image name.
DYNAMO_DOCKER_FLAGS="--gpus all --network host --rm" \
  config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1

# Use a different image for frontend capture.
DYNAMO_FRONTEND_IMAGE=nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1 \
  config_fetch/capture_dynamo_runtime_configs.sh --engine frontend

# Use a custom Omni image if NVIDIA publishes one later.
DYNAMO_VLLM_OMNI_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-omni-runtime:1.1.1 \
  config_fetch/capture_dynamo_runtime_configs.sh --engine vllm_omni
```

The runner validates every generated file before copying it into
`config_fetch/dynamo_runtime/<component>/`.

## Generated Files

Backend outputs are versioned by the runtime package version discovered inside
the image:

```text
config_fetch/dynamo_runtime/vllm/vllm_args_<version>.json
config_fetch/dynamo_runtime/vllm/vllm_envs_<version>.json
config_fetch/dynamo_runtime/sglang/sglang_args_<version>.json
config_fetch/dynamo_runtime/sglang/sglang_envs_<version>.json
config_fetch/dynamo_runtime/tensorrt_llm/trtllm_args_<version>.json
config_fetch/dynamo_runtime/tensorrt_llm/trtllm_envs_<version>.json
config_fetch/dynamo_runtime/vllm_omni/vllm_omni_args_<version>.json
config_fetch/dynamo_runtime/vllm_omni/vllm_omni_envs_<version>.json
config_fetch/dynamo_runtime/<backend>/<prefix>_dynamo_help_<version>.txt
```

Frontend outputs include the flat catalogs plus raw help and metadata:

```text
config_fetch/dynamo_runtime/frontend/frontend_args_<version>.json
config_fetch/dynamo_runtime/frontend/frontend_envs_<version>.json
config_fetch/dynamo_runtime/frontend/frontend_help_<version>.txt
config_fetch/dynamo_runtime/frontend/frontend_meta_<version>.json
```

Temporary intermediate files such as `*_dynamo_wrapper.json` and
`*_dynamo_envs.json` are kept only inside the runner's temporary output
directory. Commit the merged final JSON files and the raw `*_dynamo_help_*.txt`
files, because the help text is the audit trail for the authoritative Dynamo
backend CLI.

After the merged files are copied, `rebucket_runtime_configs.py` rewrites only
the `ui` tier metadata and creates role-specific backend views:

```text
config_fetch/dynamo_runtime/vllm/vllm_prefill_args_<version>.json
config_fetch/dynamo_runtime/vllm/vllm_prefill_envs_<version>.json
config_fetch/dynamo_runtime/vllm/vllm_decode_args_<version>.json
config_fetch/dynamo_runtime/vllm/vllm_decode_envs_<version>.json
```

The same pattern is used for `sglang`, `tensorrt_llm`, and `vllm_omni`.
Role-specific files contain the same records as the base catalog; only `ui`
tiers and ordering are changed.

## JSON Structure: Args

Every args file is a flat JSON object. Top-level metadata keys are followed by
argument records keyed by canonical setting name.

```json
{
  "engine": "sglang",
  "version": "0.5.10.post1",
  "date": "2026-05-11",
  "source": "introspected from ... + Dynamo wrapper ArgGroups",

  "disagg_config": {
    "type": "str",
    "flag": false,
    "arg": "--disagg-config",
    "default_value": null,
    "value": null,
    "description": "Path to YAML disaggregation config file.",
    "env_var": "DYN_SGL_DISAGG_CONFIG",
    "module": "dynamo_sglang",
    "config_class": "Dynamo SGLang Options",
    "ui": "less_frequent",
    "aic": false
  }
}
```

Required args fields:

| Field | Meaning |
| --- | --- |
| `type` | Validation type: `str`, `int`, `float`, `bool`, `List[str]`, `List[int]`, `path`, etc. |
| `flag` | `true` for boolean flags, `false` for value args. |
| `arg` | CLI option for value args. |
| `true_arg` / `false_arg` | CLI option forms for boolean flags. At least one is present for flags. |
| `default_value` | JSON-safe default from the image, or `null` if unknown/required. |
| `value` | Always `null` in generated catalogs. |
| `description` | Help text or source comment when available. |
| `module` | Logical source group. |
| `config_class` | Source class/group name. |
| `ui` | `primary`, `advanced`, or `less_frequent`. Metadata only. |
| `aic` | `true` means profile UI should hide this because AIC owns it. Metadata only. |

Optional args fields include `choices`, `multiple`, `required`, `status`,
`env_var`, `deprecated_alias`, `config_key`, and TRT-LLM Pydantic metadata.

## JSON Structure: Envs

Every envs file is also flat. Top-level metadata keys are followed by env records
keyed by environment variable name.

```json
{
  "engine": "sglang",
  "version": "0.5.10.post1",
  "date": "2026-05-11",
  "source": "introspected from ... + Dynamo wrapper ArgGroups",

  "DYN_SGL_DISAGG_CONFIG": {
    "category": "dynamo_sglang",
    "type": "str",
    "default_value": null,
    "value": null,
    "description": "Path to YAML disaggregation config file.",
    "arg": "--disagg-config",
    "arg_key": "disagg_config",
    "ui": "less_frequent",
    "aic": false
  }
}
```

Required env fields:

| Field | Meaning |
| --- | --- |
| `category` | Logical source group. |
| `type` | Validation type. |
| `default_value` | Default from the image/source, or `null`. |
| `value` | Always `null` in generated catalogs. |
| `description` | Help text or source comment when available. |
| `ui` | `primary`, `advanced`, or `less_frequent`. Metadata only. |
| `aic` | `true` means profile UI should hide this because AIC owns it. Metadata only. |

Optional env fields include `choices`, `arg`, `true_arg`, `false_arg`, `arg_key`,
`multiple`, `status`, `deprecated_alias`, and `source_file`.

## Capture Pipeline Per Component

### vLLM

Inside the Dynamo vLLM image, the runner executes:

```bash
python -m dynamo.vllm --help > /out/vllm_dynamo_help.txt
python vllm/gen_vllm_args.py --out /out/vllm_args.json
python vllm/gen_vllm_envs.py --out /out/vllm_envs.json
python dynamo/gen_dynamo_wrapper_args.py --engine vllm --out /out/vllm_dynamo_wrapper.json
python dynamo/gen_dynamo_envs.py --engine vllm --out /out/vllm_dynamo_envs.json
python dynamo/merge_dynamo_wrapper.py ...
```

The final catalog contains native vLLM args/envs plus Dynamo wrapper args/envs
and Dynamo env-only vars.

### SGLang

SGLang follows the same native plus Dynamo merge flow. This is where Dynamo
entries such as `DYN_ENDPOINT`, `DYN_SGL_DISAGG_CONFIG`,
`DYN_SGL_EMBEDDING_WORKER`, `DYN_TOOL_CALL_PARSER`, and
`DYN_REASONING_PARSER` are captured when exposed by the image.

### TensorRT-LLM

TRT-LLM args are collected from both the Click serve command and LLM API
Pydantic models. Env vars are best-effort source-scanned because TRT-LLM does
not expose one central env registry. Dynamo wrapper and Dynamo env-only records
are merged afterward.

### vLLM-Omni

Dynamo Omni recipes currently launch through `python -m dynamo.vllm`, so
`vllm_omni` starts from the vLLM capture and applies
`vllm_omni/apply_vllm_omni_overlay.py`. The overlay adds or promotes
multimodal settings such as:

```text
enable_multimodal
media_io_kwargs
video_pruning_rate
multimodal_embedding_cache_capacity_gb
DYN_MM_VIDEO_NUM_FRAMES
DYN_VLLM_EMBEDDING_TRANSFER_MODE
```

If a future image exposes a dedicated Omni backend/parser, replace the overlay
with direct introspection and keep the same flat output structure.

### Frontend

Frontend capture runs:

```bash
python frontend/gen_frontend_args.py \
  --out /out/frontend_args.json \
  --env-out /out/frontend_envs.json \
  --help-out /out/frontend_help.txt \
  --meta-out /out/frontend_meta.json
```

The script captures `python -m dynamo.frontend --help`, parses frontend,
KV-router, and AIC help sections, and emits engine-style flat args/envs JSON.
The raw help text is stored next to the generated files so future parser changes
can be audited.

## UI and Profile Use

The catalogs intentionally include far more than a user should see on the first
screen. Product/UI code should apply a separate view policy:

1. Hide `aic: true` settings from the user.
2. Show `ui: primary` in the main profile form.
3. Put `ui: advanced` behind an advanced section.
4. Keep `ui: less_frequent` searchable or available in "show all".
5. Never delete records from the catalog to simplify the UI.

Model choice, TP/PP/DP, token budgets, GPU/memory sizing, ports/endpoints,
secrets, and placement are usually supplied by AIC, Helm/operator defaults, or
K8s patches. Keep them in the catalog if the image exposes them, but the profile
UI can hide or ignore them.

Prefill and decode worker files are derived views over the same catalog. They do
not remove records; they only promote the role-relevant knobs into
`primary`/`advanced` and leave everything else as `less_frequent`.

Regenerate bucket views without recapturing images:

```bash
python config_fetch/rebucket_runtime_configs.py --root config_fetch/dynamo_runtime
```

## Maintenance Workflow

For a new Dynamo release:

1. Run the Docker capture for the new tag.
2. Review the printed arg/env counts. A large unexpected drop usually means the
   image command failed, a parser import changed, or a scan path needs updating.
3. Inspect generated `source` fields to confirm native, Dynamo backend
   ArgGroups, and Dynamo env-only sources were merged.
4. Inspect the raw `*_dynamo_help_*.txt` files. They should still show the
   expected group model: `Dynamo Runtime Options`, backend-specific Dynamo
   options, and native engine options.
5. Check that no unsupported frontend fields leaked in:

   ```bash
   rg '"emit"|"managed_by"|"reason"' config_fetch/dynamo_runtime
   ```

6. Compare old/new JSONs for added, removed, and changed records. UI bucket
   changes can be reviewed separately because `ui`, `aic`, and `value` are not
   engine behavior.
7. If new Dynamo backend args appear, they should be captured automatically by
   `gen_dynamo_wrapper_args.py`.
8. If new direct `DYN_*` env reads appear, they should be captured automatically
   by `gen_dynamo_envs.py` unless the source moved outside the scanned roots.
9. Commit the updated JSON/help artifacts and any parser/overlay changes
   together.

## Adding a New Backend

To support another backend:

1. Add `config_fetch/<backend>/gen_<backend>_args.py`.
2. Add `config_fetch/<backend>/gen_<backend>_envs.py`.
3. Emit the same flat args/env structure and required fields.
4. Add image selection and container commands to
   `generate_dynamo_runtime_configs.sh`.
5. If Dynamo adds wrapper ArgGroups for the backend, extend
   `dynamo/gen_dynamo_wrapper_args.py`.
6. If Dynamo source reads backend-specific env vars directly, extend
   `dynamo/gen_dynamo_envs.py`.
7. Add validation/copy logic in the runner.

The default should always be exhaustive capture first, product filtering later.
