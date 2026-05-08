# test_commits

## Dynamo frontend config

`frontend_config.py` manages the Dynamo frontend server config.

The source file named `frontend.json` is argparse help text. The generated
server config is `frontend_args.json`, which is the single source of truth for
both CLI args and env vars.

Regenerate the config from the argparse help text:

```bash
python3 frontend_config.py generate \
  --input ./frontend.json \
  --output ./frontend_args.json
```

Users should only edit the `value` fields in `frontend_args.json`. A `null`
value means the setting is not overridden and will not be emitted.

The config is grouped first by subsystem, then by stability bucket:

```json
{
  "frontend": {
    "ga": {},
    "exp": {}
  },
  "kv_router": {
    "ga": {},
    "exp": {}
  },
  "aic": {
    "ga": {},
    "exp": {}
  }
}
```

The containers map to argparse help sections:

- `frontend`: `Dynamo Frontend Options`
- `kv_router`: `KV Router Options`
- `aic`: `AIC Perf Model Options`

Deprecated options are not generated. If an option block is marked
`[Deprecated]`, the setting and its env var are omitted entirely. Deprecated
alias spellings listed as `deprecating flag:` are also stripped, so the config
only exposes canonical args.

Some settings are managed by Dynamo/operator defaults and must not be sent in
the final generated env/args, even if a user provides a value. These settings
are marked with `emit: false`:

```json
{
  "type": "value",
  "arg": "--http-port",
  "env_var": "DYN_HTTP_PORT",
  "default_value": "8000",
  "value": null,
  "emit": false,
  "managed_by": "dynamo",
  "reason": "Injected by frontend defaults to match the generated service port."
}
```

Currently auto-managed frontend settings are:

- `frontend.ga.namespace`
- `frontend.ga.http_port`
- `frontend.ga.namespace_prefix`
- `frontend.ga.discovery_backend`

Supported setting types:

```json
{
  "type": "value",
  "arg": "--http-port",
  "env_var": "DYN_HTTP_PORT",
  "default_value": "8000",
  "value": null
}
```

```json
{
  "type": "choice",
  "arg": "--router-mode",
  "env_var": "DYN_ROUTER_MODE",
  "default_value": "round-robin",
  "value": null,
  "choices": [
    "round-robin",
    "random",
    "power-of-two",
    "kv",
    "direct",
    "least-loaded",
    "device-aware-weighted"
  ]
}
```

```json
{
  "type": "boolean_flag",
  "env_var": "DYN_ROUTER_REPLICA_SYNC",
  "default_value": false,
  "value": null,
  "true_arg": "--router-replica-sync",
  "false_arg": "--no-router-replica-sync"
}
```

```json
{
  "type": "flag",
  "arg": "--some-positive-only-flag",
  "env_var": "",
  "default_value": false,
  "value": null
}
```

Use `flag` for a standalone CLI flag that has no matching `--no-*` argument.
When `value` is `true`, the parser emits the flag. When `value` is `false` or
`null`, it emits no CLI arg. If `env_var` is an empty string, no env entry is
emitted. Current Dynamo frontend flags with `--no-*` variants are represented as
`boolean_flag`.

Read the config and emit Kubernetes-style env entries plus a CLI args list:

```bash
python3 frontend_config.py emit --config ./frontend_args.json
```

Test temporary values without editing the file:

```bash
python3 frontend_config.py emit \
  --config ./frontend_args.json \
  --set frontend.ga.namespace=my-dgd \
  --set frontend.ga.http_port=9000 \
  --set frontend.ga.discovery_backend=file \
  --set frontend.ga.router_mode=kv \
  --set frontend.ga.serve_indexer=true \
  --set frontend.exp.dyn_debug_perf=true \
  --set kv_router.ga.router_replica_sync=true \
  --set kv_router.ga.router_queue_policy=wspt \
  --set kv_router.exp.router_prefill_load_model=aic \
  --set kv_router.exp.use_remote_indexer=true
```

Because `frontend.ga.namespace`, `frontend.ga.http_port`, and
`frontend.ga.discovery_backend` have `emit: false`, those temporary values are
accepted but ignored in the emitted arrays.

Example output:

```json
{
  "env_array": [
    { "name": "DYN_SERVE_INDEXER", "value": "true" },
    { "name": "DYN_ROUTER_MODE", "value": "kv" },
    { "name": "DYN_DEBUG_PERF", "value": "true" },
    { "name": "DYN_ROUTER_REPLICA_SYNC", "value": "true" },
    { "name": "DYN_ROUTER_QUEUE_POLICY", "value": "wspt" },
    { "name": "DYN_ROUTER_PREFILL_LOAD_MODEL", "value": "aic" },
    { "name": "DYN_USE_REMOTE_INDEXER", "value": "true" }
  ],
  "args_array": [
    "--serve-indexer",
    "--router-mode",
    "kv",
    "--dyn-debug-perf",
    "--router-replica-sync",
    "--router-queue-policy",
    "wspt",
    "--router-prefill-load-model",
    "aic",
    "--use-remote-indexer"
  ]
}
```

Use the parser from Python:

```python
from frontend_config import build_env_and_args, load_config

config = load_config("./frontend_args.json")
env_array, args_array = build_env_and_args(config)
```

Temporary CLI overrides use `container.bucket.setting=value`, for example:

```bash
--set kv_router.ga.router_queue_policy=wspt
--set aic.exp.aic_backend=vllm
```

Validation rules:

- `value` settings accept string, number, or `null`.
- `choice` settings must use one of the listed `choices`, or `null`.
- `boolean_flag` settings must use `true`, `false`, or `null`.
- `flag` settings must use `true`, `false`, or `null`.
- The parser emits nothing for `null`.
- The parser emits nothing for settings with `emit: false`.
- The parser emits no env entry when `env_var` is `""`.
- Env values are stringified because Kubernetes env values are strings.

## Dynamo manifest transform

`transform_dynamo_manifest.py` converts a raw `DynamoGraphDeployment` into a
deployment-ready manifest. The raw manifest supplies workload facts; the named
policy is loaded from SQLite and supplies deployment choices.

```bash
python3 transform_dynamo_manifest.py init-db \
  --db ./manifest_policies.sqlite \
  --seed-default-policy

python3 transform_dynamo_manifest.py transform \
  --db ./manifest_policies.sqlite \
  --policy aws-p5-efa-vllm-disagg \
  --input ./raw.yaml \
  --output ./k8s_deploy.yaml
```

Useful policy commands:

```bash
python3 transform_dynamo_manifest.py list-policies --db ./manifest_policies.sqlite
python3 transform_dynamo_manifest.py show-policy --db ./manifest_policies.sqlite --policy aws-p5-efa-vllm-disagg
python3 transform_dynamo_manifest.py upsert-policy --db ./manifest_policies.sqlite --policy my-policy --file ./policy.yaml
```
