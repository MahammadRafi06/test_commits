# test_commits

## Dynamo Runtime Config Capture

`config_fetch/` generates versioned JSON catalogs for NVIDIA Dynamo runtime
images. The catalogs are exhaustive inventories of runtime args/envs; profile UI
filtering, AIC ownership, Helm/operator defaults, and K8s patches are applied
later without deleting catalog records.

Read the full guide:

```text
config_fetch/ENGINE_CONFIG_GUIDE.md
```

Pull the Dynamo runtime images and generate all supported catalogs locally:

```bash
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1
```

Generate one component:

```bash
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm_omni
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine sglang
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine tensorrt_llm
config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine frontend
```

Use `--skip-pull` when the images are already local. The lower-level
`config_fetch/generate_dynamo_runtime_configs.sh` script is still available for
running the container capture step directly.

The runner captures:

- Native engine CLI args and env vars.
- Dynamo backend wrapper args and their env-backed forms.
- Dynamo `DYN_*` / `DYNAMO_*` env-only source reads.
- Dynamo frontend args/envs from `python -m dynamo.frontend --help`.
- vLLM-Omni multimodal overlay settings.

Generated files are written under:

```text
config_fetch/dynamo_runtime/<component>/
```

Backend components also get prefill/decode UI bucket views, for example:

```text
config_fetch/dynamo_runtime/vllm/vllm_prefill_args_<version>.json
config_fetch/dynamo_runtime/vllm/vllm_decode_args_<version>.json
```

Those role files keep every captured record and only reorganize the `ui` bucket.

Every args/envs JSON uses the same flat structure:

- Top-level metadata: `engine`, `version`, `date`, `source`.
- One top-level record per arg or env var.
- `value` is always `null` in generated catalogs.
- `ui` and `aic` are metadata for profile rendering; they do not control
  capture.
- Legacy frontend-only fields such as `emit`, `managed_by`, and `reason` are not
  part of the generated structure.

Reapply bucket views without recapturing images:

```bash
python config_fetch/rebucket_runtime_configs.py --root config_fetch/dynamo_runtime
```

## Dynamo Manifest Transform

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
