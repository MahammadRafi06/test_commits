# Engine Configuration JSON — Build Guide

A system for introspecting LLM inference engine configuration at runtime and producing structured JSON files that drive a configuration UI, an AI configurator (AIC), and version-diff tooling.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [How the Generators Work](#2-how-the-generators-work)
3. [Running the Generators](#3-running-the-generators)
4. [JSON Schema — Args](#4-json-schema--args)
5. [JSON Schema — Envs](#5-json-schema--envs)
6. [Delta JSON Schema](#6-delta-json-schema)
7. [UI Tier System](#7-ui-tier-system)
8. [Building the UI](#8-building-the-ui)
9. [AIC Integration](#9-aic-integration)
10. [Extending to a New Engine](#10-extending-to-a-new-engine)

---

## 1. Project Structure

```
test_commits/
├── gen_delta.py                  # Standalone version-diff tool (any engine)
│
├── vllm/
│   ├── gen_vllm_args.py          # Generates vLLM CLI args JSON
│   ├── gen_vllm_envs.py          # Generates vLLM env vars JSON
│   ├── vllm_args_0.20.1.json
│   └── vllm_envs_0.20.1.json
│
├── sglang/
│   ├── gen_sglang_args.py        # Generates SGLang CLI args JSON
│   ├── gen_sglang_envs.py        # Generates SGLang env vars JSON
│   ├── sglang_args.json
│   └── sglang_envs.json
│
└── tensorrt_llm/
    ├── gen_trtllm_args.py        # Generates TRT-LLM CLI args JSON
    ├── gen_trtllm_envs.py        # Generates TRT-LLM env vars JSON
    └── (run after installing TRT-LLM)
```

---

## 2. How the Generators Work

Each generator **introspects a live installed package** rather than parsing source code. This means the output is always accurate to whatever version is installed in the active Python environment.

| Generator | Introspection source |
|-----------|----------------------|
| `gen_vllm_args.py` | `vllm.entrypoints.openai.cli_args.make_arg_parser()` — the actual argparse parser vLLM uses for `vllm serve` |
| `gen_vllm_envs.py` | `vllm.envs.environment_variables` — dict of lambdas, each called with the env var cleared to get the true default |
| `gen_sglang_args.py` | `sglang.srt.server_args.ServerArgs.add_cli_args()` — argparse |
| `gen_sglang_envs.py` | `vars(sglang.srt.environ.Envs)` — descriptor instances of `EnvField` subclasses |
| `gen_trtllm_args.py` | `tensorrt_llm.commands.serve` (Click command) + `BaseLlmArgs / TrtLlmArgs / TorchLlmArgs` (Pydantic v2) |
| `gen_trtllm_envs.py` | Source scan of the installed `tensorrt_llm` package for `os.getenv()` and `StrEnum` patterns |

### Type inference

Types are derived from the introspection source and written as Python type strings for direct use in validation:

| Inferred from | Type string |
|---------------|-------------|
| argparse `action.type is int` | `"int"` |
| argparse `action.type is float` | `"float"` |
| argparse `BooleanOptionalAction` or `store_true` | `"bool"` |
| argparse `nargs='+'` or `'*'` | `"List[int]"` / `"List[str]"` |
| Pydantic `bool` annotation | `"bool"` |
| Pydantic `List[str]` annotation | `"List[str]"` |
| Pydantic `Literal[...]` | `"str"` or `"int"` based on literal values |
| Click `param.type.name == "INTEGER"` | `"int"` |
| Click `click.Path` | `"path"` |
| SGLang `EnvBool` | `"bool"` |
| SGLang `EnvInt` / `EnvFloat` / `EnvTuple` | `"int"` / `"float"` / `"List[str]"` |
| Default value is `bool` / `int` / `float` / `list` | corresponding type (fallback) |

---

## 3. Running the Generators

Each generator requires the target engine installed in the active Python environment. No GPU is required.

### vLLM

```bash
# Create and activate a venv with vLLM installed
python gen_vllm_args.py --out vllm_args_0.20.1.json
python gen_vllm_envs.py --out vllm_envs_0.20.1.json
```

### SGLang

```bash
python gen_sglang_args.py --out sglang_args.json
python gen_sglang_envs.py --out sglang_envs.json
```

### TRT-LLM (requires Python 3.12 + NVIDIA PyPI index)

```bash
pip install --pre tensorrt-llm==1.3.0rc14 \
    --extra-index-url https://pypi.nvidia.com/
python gen_trtllm_args.py --out trtllm_args_1.3.0rc14.json
python gen_trtllm_envs.py --out trtllm_envs_1.3.0rc14.json
```

### Generating a delta against a previous version

Pass `--prev` with the path to the previous version's JSON. A delta file is written alongside the main output automatically.

```bash
# Inline — delta is produced as a side effect
python gen_vllm_args.py \
    --out vllm_args_0.21.0.json \
    --prev vllm_args_0.20.1.json
# → writes delta__vllm_args_0.20.1__vllm_args_0.21.0.json

# Standalone tool — works on any two engine JSONs
python gen_delta.py \
    --old vllm_args_0.20.1.json \
    --new vllm_args_0.21.0.json
```

---

## 4. JSON Schema — Args

Top-level keys are either metadata (`engine`, `version`, `date`, `source`) or arg names. Every arg name maps directly to a record object.

```jsonc
{
  "engine": "vllm",           // engine identifier
  "version": "0.20.1",        // engine version from package
  "date": "2026-05-11",       // date generated
  "source": "introspected from ...",

  "model": {
    // ── Core fields ──────────────────────────────────────────────────────
    "type":          "str",           // Python type: str | int | float | bool | List[str] | path | …
    "default_value": null,            // default as a JSON-native value; null = required or unknown
    "value":         null,            // slot for the user's chosen value (always null in stored file)
    "description":   "Name or path of the Hugging Face model to use.",

    // ── CLI shape ─────────────────────────────────────────────────────────
    "flag":          false,           // true = boolean flag, no value required
    "arg":           "--model",       // CLI option string; absent when flag=true (use true_arg/false_arg)

    // ── Boolean flag shape (flag: true) ───────────────────────────────────
    // "true_arg":   "--enable-prefix-caching",
    // "false_arg":  "--no-enable-prefix-caching",  // null for store_true style

    // ── Choice shape (when choices array is present) ──────────────────────
    // "choices": ["auto", "half", "float16", "bfloat16", "float32"],

    // ── Origin metadata ───────────────────────────────────────────────────
    "module":        "model",         // logical grouping derived from config class
    "config_class":  "ModelConfig",   // exact class that owns this field

    // ── UI control ────────────────────────────────────────────────────────
    "ui":            "primary",       // primary | advanced | less_frequent
    "aic":           false,           // true = owned by AI configurator, hide from user

    // ── TRT-LLM only ─────────────────────────────────────────────────────
    // "status": "beta"               // stability tag from :tag:`beta` or Field(status=…)
  }
}
```

### Complete field reference

| Field | Always present | Description |
|-------|---------------|-------------|
| `type` | ✓ | Python type string for input validation |
| `default_value` | ✓ | Default; `null` when required or not determinable |
| `value` | ✓ | Always `null` in stored file — set by UI/AIC at runtime |
| `description` | ✓ | Help text from the engine source |
| `flag` | ✓ | `true` for boolean CLI flags |
| `arg` | when `flag=false` | The `--option` string to pass on the CLI |
| `true_arg` | when `flag=true` | The enabling form, e.g. `--enable-prefix-caching` |
| `false_arg` | when `flag=true` | The disabling form; `null` for store_true-only flags |
| `choices` | optional | Allowed values; validate against this list |
| `multiple` | optional | `true` when the arg can be repeated (argparse nargs) |
| `module` | ✓ | Logical module/backend grouping |
| `config_class` | ✓ | Source config class name |
| `ui` | ✓ | Display tier: `primary` / `advanced` / `less_frequent` |
| `aic` | ✓ | `true` = AI configurator manages this, hide from user |
| `status` | TRT-LLM | `"beta"` / `"prototype"` stability label |
| `required` | TRT-LLM positional | `true` for required positional args |

---

## 5. JSON Schema — Envs

Same flat top-level structure. Each env var name maps to a record.

```jsonc
{
  "engine": "vllm",
  "version": "0.20.1",
  "date": "2026-05-11",
  "source": "introspected from vllm.envs.environment_variables",

  "VLLM_HOST_IP": {
    "category":      "networking",    // logical grouping
    "type":          "str",           // Python type for validation
    "default_value": "",              // default value (always a scalar or list)
    "value":         null,            // user-set value slot
    "description":   "IP address of the current node in distributed setups.",
    "ui":            "primary",       // primary | advanced | less_frequent
    "aic":           false,           // true = AIC-managed, hide from user

    // ── When choices exist ────────────────────────────────────────────────
    // "choices": ["spawn", "fork", "forkserver"]
  }
}
```

### Field reference

| Field | Description |
|-------|-------------|
| `category` | Functional grouping: `networking`, `gpu`, `logging`, `cache`, `engine`, etc. |
| `type` | Python type string — same set as args |
| `default_value` | Default when the var is unset |
| `value` | `null` in stored file; written by UI/AIC at runtime |
| `description` | Extracted from source comments preceding each `os.getenv()` call |
| `ui` | Display tier |
| `aic` | AIC ownership flag |
| `choices` | Optional constrained value list |

---

## 6. Delta JSON Schema

Produced by `gen_delta.py` or the `--prev` flag on any generator.

```jsonc
{
  "engine":      "vllm",
  "old_version": "0.20.1",
  "new_version": "0.21.0",
  "date":        "2026-05-11",

  "summary": {
    "total_old": 240,   // total args/envs in old version
    "total_new": 249,   // total args/envs in new version
    "added":     9,
    "removed":   1,
    "changed":   4,
    "unchanged": 235    // count only — not written to keep file small
  },

  // Full record from new version for each added key
  "added": {
    "new_arg": { ...full record... }
  },

  // Full record from old version for each removed key
  "removed": {
    "old_arg": { ...full record... }
  },

  // Only the fields that changed, with old and new values
  "changed": {
    "dtype": {
      "diff": {
        "choices": {
          "old": ["auto", "half", "float16"],
          "new":  ["auto", "bfloat16", "float", "float16", "float32", "half"]
        }
      }
    },
    "max_num_seqs": {
      "diff": {
        "default_value": { "old": 64, "new": 256 }
      }
    }
  }
}
```

**Fields excluded from diff:** `ui`, `aic`, `value` — these are user-managed metadata, not engine-driven, so they should never appear as "changed" between versions.

---

## 7. UI Tier System

### The two source-of-truth lists

Every generator contains two explicit Python lists that control both **display tier** and **sort order**:

```python
# In gen_vllm_args.py (example)

_UI_PRIMARY = [          # ← max 15 — shown immediately in the main form
    "model", "dtype", "tensor_parallel_size", "pipeline_parallel_size",
    "gpu_memory_utilization", "max_model_len",
    "host", "port", "served_model_name", "api_key",
    "trust_remote_code", "quantization", "kv_cache_dtype",
    "max_num_seqs", "enable_prefix_caching",
]

_UI_ADVANCED = [         # ← max 20 — shown in collapsible "Advanced" section
    "max_num_batched_tokens", "enable_chunked_prefill", "scheduling_policy",
    "load_format", "tokenizer", "chat_template", ...
]

# Everything else → ui = "less_frequent" (search / show-all only)
```

**To move an arg between tiers:** edit one list, remove from the other, regenerate. Nothing else changes.

### Tier assignment logic

```python
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
```

The JSON is written in this order, so iteration order == display order with no extra sorting needed.

### The `aic` flag

When `aic: true`, the arg/env is managed programmatically by the AI configurator and must be **completely hidden** from the user — not even visible in search or show-all.

```python
# Setting aic=true for an arg (done manually or by AIC after generation)
data["max_num_tokens"]["aic"] = True
```

---

## 8. Building the UI

### Display logic (pseudocode)

```
for each arg/env in JSON:
    if aic == true:
        skip entirely  ← AIC owns it

    if ui == "primary":
        render in main form

    elif ui == "advanced":
        render inside <Accordion title="Advanced">

    elif ui == "less_frequent":
        only show when user searches or clicks "Show all"
```

### Widget mapping by `type`

| `type` value | Widget | Notes |
|-------------|--------|-------|
| `"bool"` + `flag: true` | Toggle / Switch | Use `true_arg` / `false_arg` for the CLI string; show `false_arg: null` as a hint that it cannot be disabled |
| `"bool"` | Checkbox | |
| `"int"` | Number input | Integer step |
| `"float"` | Number input | Decimal step (e.g. 0.01 for `gpu_memory_utilization`) |
| `"str"` + `choices` | Select / Dropdown | Populate options from `choices` array |
| `"str"` | Text input | |
| `"List[str]"` | Tag / chip input | Each tag is one list item |
| `"List[int]"` | Comma-separated number input | Parse to `[int, int, …]` |
| `"path"` | File/directory picker or text input with path hint | |

### Grouping in the Advanced section

Use `module` (args) or `category` (envs) as section headers inside the Advanced accordion. Items sharing the same module/category appear together.

```
▼ Advanced
  ┌─ model ──────────────────────────────────┐
  │  load_format        [dropdown]           │
  │  tokenizer          [text]               │
  │  chat_template      [text]               │
  └──────────────────────────────────────────┘
  ┌─ scheduler ──────────────────────────────┐
  │  max_num_batched_tokens  [number]        │
  │  enable_chunked_prefill  [toggle]        │
  └──────────────────────────────────────────┘
```

### Search

Search should span all args/envs regardless of tier, but exclude `aic: true` items. Match against both the key name and the `description` field. Show results grouped by tier so the user can tell at a glance whether a result is primary, advanced, or rarely used.

```
🔍  "lora"
────────────────────────────────────
  primary      enable_lora              Enable LoRA adapters
  advanced     max_loras                Max simultaneous LoRA adapters
               lora_modules             LoRA module specifications
  less_freq    max_loras_per_batch      …
               max_cpu_loras            …
```

### Validation

Use the `type` field to validate user input before generating the CLI command or env var string:

```js
function validate(record, userInput) {
  if (record.choices) {
    return record.choices.includes(userInput)
      ? null
      : `Must be one of: ${record.choices.join(', ')}`;
  }
  switch (record.type) {
    case 'int':   return /^-?\d+$/.test(userInput) ? null : 'Must be an integer';
    case 'float': return /^-?\d+(\.\d+)?$/.test(userInput) ? null : 'Must be a number';
    case 'bool':  return null;  // toggle handles this
    case 'path':  return userInput.length > 0 ? null : 'Path required';
    default:      return null;  // str — any value is valid
  }
}
```

### Generating the CLI command from user-set values

```js
function buildCliArgs(argsJson) {
  const meta = new Set(['engine','version','date','source']);
  const parts = [];

  for (const [name, rec] of Object.entries(argsJson)) {
    if (meta.has(name)) continue;
    if (rec.aic || rec.value === null) continue;  // skip AIC-owned and unset

    if (rec.flag) {
      // Boolean flag
      parts.push(rec.value ? rec.true_arg : rec.false_arg ?? '');
    } else if (Array.isArray(rec.value)) {
      // List arg — append flag multiple times
      for (const item of rec.value) parts.push(`${rec.arg} ${item}`);
    } else {
      parts.push(`${rec.arg} ${rec.value}`);
    }
  }
  return parts.filter(Boolean).join(' \\\n  ');
}
```

### Generating env var exports

```js
function buildEnvExports(envsJson) {
  const meta = new Set(['engine','version','date','source']);
  const lines = [];

  for (const [name, rec] of Object.entries(envsJson)) {
    if (meta.has(name)) continue;
    if (rec.aic || rec.value === null) continue;
    lines.push(`export ${name}="${rec.value}"`);
  }
  return lines.join('\n');
}
```

### Showing delta to users

When a new engine version is available, load the delta JSON and surface changes in the UI:

```
⚠️  vLLM 0.21.0 — 9 new args, 1 removed, 4 changed

  New in 0.21.0
  ┌─ uvicorn_log_level  [str]  "info"  ──────────────────────────────┐
  │  Log level for the Uvicorn HTTP server.                          │
  └──────────────────────────────────────────────────────────────────┘

  Changed
  ┌─ dtype  choices changed ─────────────────────────────────────────┐
  │  Old: auto | half | float16                                      │
  │  New: auto | bfloat16 | float | float16 | float32 | half        │
  └──────────────────────────────────────────────────────────────────┘

  Removed
  ┌─ old_arg  was: str, default null ────────────────────────────────┐
  │  Remove from any saved configs that reference it.                │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 9. AIC Integration

The AI configurator reads and writes the `value` field on each record. It respects the following contract:

| `aic` | `ui` | Behaviour |
|-------|------|-----------|
| `false` | `primary` | User sets it; AIC may suggest but not override |
| `false` | `advanced` / `less_frequent` | User may set; AIC fills if not set by user |
| `true` | any | AIC sets programmatically; **never shown to user** |

### Workflow

1. UI loads the JSON — renders user-visible args/envs based on `ui` tier, hides `aic: true` items.
2. User configures visible fields → `value` slots are populated.
3. AIC runs → reads user-set values, fills `aic: true` fields (e.g. `max_num_tokens`, token budgets, infra knobs), and optionally suggests values for `aic: false` fields the user left empty.
4. CLI command / env exports are generated from all records where `value != null`.

### Marking a field as AIC-owned

After generating the JSON, patch the relevant records before serving to the frontend:

```python
AIC_OWNED_ARGS = {
    "vllm": ["max_num_tokens", "max_num_batched_tokens", "distributed_executor_backend"],
    "sglang": ["max_total_tokens", "schedule_policy"],
}

for key in AIC_OWNED_ARGS.get(engine, []):
    if key in data:
        data[key]["aic"] = True
```

---

## 10. Extending to a New Engine

1. **Create `gen_<engine>_args.py`** — introspect the engine's CLI parser (argparse, Click, Typer, etc.) and emit the flat JSON structure with `type`, `flag`, `arg`, `module`, `config_class`, `ui`, `aic` fields.

2. **Create `gen_<engine>_envs.py`** — introspect environment variable definitions and emit `type`, `category`, `default_value`, `ui`, `aic` fields.

3. **Define `_UI_PRIMARY` and `_UI_ADVANCED` lists** inside each generator — these are the only two things to tune based on user feedback.

4. **Add `_compute_delta` and `--prev`** — copy the identical delta block from any existing generator.

5. The delta tool `gen_delta.py` works on the output without any changes.

### Minimum required fields per record

```
type, default_value, value, description, ui, aic
```

For args, additionally: `flag`, and either `arg` (non-flag) or `true_arg` (flag).  
For envs, additionally: `category`.
