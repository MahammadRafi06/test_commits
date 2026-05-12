#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Generate config JSONs from NVIDIA Dynamo runtime images.

Run this on a host with Docker and NVIDIA Container Toolkit available.

Usage:
  config_fetch/generate_dynamo_runtime_configs.sh [--tag 1.1.1] [--engine vllm] [--pull]

Options:
  --tag TAG         Dynamo runtime image tag to use. Default: 1.1.1
  --engine ENGINE  Generate one component. Repeatable. One of:
                   vllm, vllm_omni, sglang, tensorrt_llm, frontend
                   Default: generate all backend engines plus frontend.
  --out-dir DIR    Output root. Default: config_fetch/dynamo_runtime
  --pull           docker pull each selected image before running it.
  -h, --help       Show this help.

Environment:
  DYNAMO_DOCKER_FLAGS  Docker flags before the image name.
                       Default: "--gpus all --network host --rm"
  DYNAMO_FRONTEND_IMAGE Image to use for frontend config capture.
                       Default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<--tag>
  DYNAMO_VLLM_OMNI_IMAGE Image to use for vllm_omni config capture.
                       Default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<--tag>

Examples:
  config_fetch/generate_dynamo_runtime_configs.sh --tag 1.1.1
  config_fetch/generate_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm
  config_fetch/generate_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm_omni
  config_fetch/generate_dynamo_runtime_configs.sh --tag 1.1.1 --engine frontend
  DYNAMO_DOCKER_FLAGS="--gpus all --rm" config_fetch/generate_dynamo_runtime_configs.sh
EOF
}

tag="${DYNAMO_IMAGE_TAG:-1.1.1}"
pull=0
engines=()
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="${2:?missing value for --tag}"
      shift 2
      ;;
    --engine)
      engines+=("${2:?missing value for --engine}")
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --pull)
      pull=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
out_dir="${out_dir:-${script_dir}/dynamo_runtime}"

if [[ ${#engines[@]} -eq 0 ]]; then
  engines=(vllm vllm_omni sglang tensorrt_llm frontend)
fi

docker_flags_string="${DYNAMO_DOCKER_FLAGS:---gpus all --network host --rm}"
# shellcheck disable=SC2206
docker_flags=(${docker_flags_string})

tmp_dir="$(mktemp -d "${script_dir}/.generated_dynamo_runtime.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

image_for_engine() {
  case "$1" in
    vllm) echo "nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}" ;;
    vllm_omni) echo "${DYNAMO_VLLM_OMNI_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}}" ;;
    sglang) echo "nvcr.io/nvidia/ai-dynamo/sglang-runtime:${tag}" ;;
    tensorrt_llm|trtllm) echo "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:${tag}" ;;
    frontend) echo "${DYNAMO_FRONTEND_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}}" ;;
    *)
      echo "Unknown engine: $1" >&2
      exit 2
      ;;
  esac
}

canonical_engine() {
  case "$1" in
    vllm) echo "vllm" ;;
    vllm_omni|vllm-omni) echo "vllm_omni" ;;
    sglang) echo "sglang" ;;
    tensorrt_llm|trtllm) echo "tensorrt_llm" ;;
    frontend) echo "frontend" ;;
    *)
      echo "Unknown engine: $1" >&2
      exit 2
      ;;
  esac
}

container_cmd_for_engine() {
  case "$1" in
    vllm)
      cat <<'EOF'
python -m dynamo.vllm --help > /out/vllm_dynamo_help.txt
python vllm/gen_vllm_args.py --out /out/vllm_args.json
python vllm/gen_vllm_envs.py --out /out/vllm_envs.json
python dynamo/gen_dynamo_wrapper_args.py --engine vllm --out /out/vllm_dynamo_wrapper.json
python dynamo/gen_dynamo_envs.py --engine vllm --out /out/vllm_dynamo_envs.json
python dynamo/merge_dynamo_wrapper.py \
  --wrapper-json /out/vllm_dynamo_wrapper.json \
  --dynamo-envs-json /out/vllm_dynamo_envs.json \
  --args-json /out/vllm_args.json \
  --envs-json /out/vllm_envs.json
EOF
      ;;
    vllm_omni)
      cat <<'EOF'
python -m dynamo.vllm --help > /out/vllm_omni_dynamo_help.txt
python vllm/gen_vllm_args.py --out /out/vllm_omni_args.json
python vllm/gen_vllm_envs.py --out /out/vllm_omni_envs.json
python dynamo/gen_dynamo_wrapper_args.py --engine vllm --out /out/vllm_omni_dynamo_wrapper.json
python dynamo/gen_dynamo_envs.py --engine vllm --out /out/vllm_omni_dynamo_envs.json
python dynamo/merge_dynamo_wrapper.py \
  --wrapper-json /out/vllm_omni_dynamo_wrapper.json \
  --dynamo-envs-json /out/vllm_omni_dynamo_envs.json \
  --args-json /out/vllm_omni_args.json \
  --envs-json /out/vllm_omni_envs.json
python vllm_omni/apply_vllm_omni_overlay.py \
  --args-json /out/vllm_omni_args.json \
  --envs-json /out/vllm_omni_envs.json
EOF
      ;;
    sglang)
      cat <<'EOF'
python -m dynamo.sglang --help > /out/sglang_dynamo_help.txt
python sglang/gen_sglang_args.py --out /out/sglang_args.json
python sglang/gen_sglang_envs.py --out /out/sglang_envs.json
python dynamo/gen_dynamo_wrapper_args.py --engine sglang --out /out/sglang_dynamo_wrapper.json
python dynamo/gen_dynamo_envs.py --engine sglang --out /out/sglang_dynamo_envs.json
python dynamo/merge_dynamo_wrapper.py \
  --wrapper-json /out/sglang_dynamo_wrapper.json \
  --dynamo-envs-json /out/sglang_dynamo_envs.json \
  --args-json /out/sglang_args.json \
  --envs-json /out/sglang_envs.json
EOF
      ;;
    tensorrt_llm)
      cat <<'EOF'
python -m dynamo.trtllm --help > /out/trtllm_dynamo_help.txt
python tensorrt_llm/gen_trtllm_args.py --out /out/trtllm_args.json
python tensorrt_llm/gen_trtllm_envs.py --out /out/trtllm_envs.json
python dynamo/gen_dynamo_wrapper_args.py --engine tensorrt_llm --out /out/trtllm_dynamo_wrapper.json
python dynamo/gen_dynamo_envs.py --engine tensorrt_llm --out /out/trtllm_dynamo_envs.json
python dynamo/merge_dynamo_wrapper.py \
  --wrapper-json /out/trtllm_dynamo_wrapper.json \
  --dynamo-envs-json /out/trtllm_dynamo_envs.json \
  --args-json /out/trtllm_args.json \
  --envs-json /out/trtllm_envs.json
EOF
      ;;
    frontend)
      cat <<'EOF'
python frontend/gen_frontend_args.py \
  --out /out/frontend_args.json \
  --env-out /out/frontend_envs.json \
  --help-out /out/frontend_help.txt \
  --meta-out /out/frontend_meta.json
EOF
      ;;
  esac
}

echo "Output root: ${out_dir}"
echo "Temporary output: ${tmp_dir}"

selected=()
for requested_engine in "${engines[@]}"; do
  engine="$(canonical_engine "$requested_engine")"
  image="$(image_for_engine "$engine")"
  selected+=("$engine")

  echo
  echo "==> ${engine}: ${image}"
  if [[ "$pull" -eq 1 ]]; then
    docker pull "$image"
  fi

  container_cmd="$(container_cmd_for_engine "$engine")"
  docker run "${docker_flags[@]}" \
    -v "${repo_root}:/work" \
    -v "${tmp_dir}:/out" \
    -w /work/config_fetch \
    --entrypoint bash \
    "$image" \
    -lc "set -Eeuo pipefail; ${container_cmd}"
done

python - "$tmp_dir" "$out_dir" "$tag" "${selected[@]}" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
tag = sys.argv[3]
engines = sys.argv[4:]

meta_keys = {"engine", "version", "date", "source"}
spec = {
    "vllm": ("vllm", "vllm"),
    "vllm_omni": ("vllm_omni", "vllm_omni"),
    "sglang": ("sglang", "sglang"),
    "tensorrt_llm": ("tensorrt_llm", "trtllm"),
}


def safe_version(version: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", version)


def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def validate_args(path: Path, data: dict) -> None:
    required = {
        "type",
        "flag",
        "default_value",
        "value",
        "description",
        "module",
        "config_class",
        "ui",
        "aic",
    }
    bad = []
    for meta_key in meta_keys:
        if meta_key not in data:
            bad.append(f"<metadata>: missing {meta_key}")
    for key, rec in data.items():
        if key in meta_keys:
            continue
        if "emit" in rec:
            bad.append(f"{key}: contains unsupported emit field")
            continue
        missing = required - set(rec)
        if missing:
            bad.append(f"{key}: missing {sorted(missing)}")
            continue
        if not isinstance(rec["flag"], bool):
            bad.append(f"{key}: flag must be boolean")
            continue
        if rec["flag"]:
            if not (rec.get("true_arg") or rec.get("false_arg") or rec.get("choices")):
                bad.append(f"{key}: flag records need true_arg, false_arg, or choices")
        elif not rec.get("arg"):
            bad.append(f"{key}: value records need arg")
    if bad:
        raise SystemExit(f"{path} has malformed arg records: {bad[:10]}")


def validate_envs(path: Path, data: dict) -> None:
    required = {
        "category",
        "type",
        "default_value",
        "value",
        "description",
        "ui",
        "aic",
    }
    bad = []
    for meta_key in meta_keys:
        if meta_key not in data:
            bad.append(f"<metadata>: missing {meta_key}")
    for key, rec in data.items():
        if key in meta_keys:
            continue
        if "emit" in rec:
            bad.append(f"{key}: contains unsupported emit field")
            continue
        missing = required - set(rec)
        if missing:
            bad.append(f"{key}: missing {sorted(missing)}")
    if bad:
        raise SystemExit(f"{path} has malformed env records: {bad[:10]}")


for engine in engines:
    if engine == "frontend":
        args_src = tmp_dir / "frontend_args.json"
        envs_src = tmp_dir / "frontend_envs.json"
        help_src = tmp_dir / "frontend_help.txt"
        meta_src = tmp_dir / "frontend_meta.json"

        args_data = load(args_src)
        envs_data = load(envs_src)
        meta_data = load(meta_src)
        validate_args(args_src, args_data)
        validate_envs(envs_src, envs_data)

        raw_version = str(meta_data.get("version") or "")
        version = safe_version(raw_version if raw_version != "unknown" else f"dynamo-{tag}")
        target_dir = out_dir / "frontend"
        target_dir.mkdir(parents=True, exist_ok=True)

        args_dst = target_dir / f"frontend_args_{version}.json"
        envs_dst = target_dir / f"frontend_envs_{version}.json"
        help_dst = target_dir / f"frontend_help_{version}.txt"
        meta_dst = target_dir / f"frontend_meta_{version}.json"
        shutil.copyfile(args_src, args_dst)
        shutil.copyfile(envs_src, envs_dst)
        shutil.copyfile(help_src, help_dst)
        shutil.copyfile(meta_src, meta_dst)

        args_count = len([k for k in args_data if k not in meta_keys])
        envs_count = len([k for k in envs_data if k not in meta_keys])
        print(f"frontend: {meta_data.get('version')} -> {args_dst} ({args_count} args)")
        print(f"frontend: {meta_data.get('version')} -> {envs_dst} ({envs_count} envs)")
        print(f"frontend: raw help -> {help_dst}")
        continue

    engine_dir, prefix = spec[engine]
    args_src = tmp_dir / f"{prefix}_args.json"
    envs_src = tmp_dir / f"{prefix}_envs.json"

    args_data = load(args_src)
    envs_data = load(envs_src)
    validate_args(args_src, args_data)
    validate_envs(envs_src, envs_data)

    version = safe_version(str(args_data["version"]))
    target_dir = out_dir / engine_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    args_dst = target_dir / f"{prefix}_args_{version}.json"
    envs_dst = target_dir / f"{prefix}_envs_{version}.json"
    help_src = tmp_dir / f"{prefix}_dynamo_help.txt"
    help_dst = target_dir / f"{prefix}_dynamo_help_{version}.txt"
    shutil.copyfile(args_src, args_dst)
    shutil.copyfile(envs_src, envs_dst)
    if help_src.exists():
        shutil.copyfile(help_src, help_dst)

    args_count = len([k for k in args_data if k not in meta_keys])
    envs_count = len([k for k in envs_data if k not in meta_keys])
    print(f"{engine}: {args_data['version']} -> {args_dst} ({args_count} args)")
    print(f"{engine}: {envs_data['version']} -> {envs_dst} ({envs_count} envs)")
    if help_src.exists():
        print(f"{engine}: raw Dynamo backend help -> {help_dst}")
PY

rebucket_args=(--root "$out_dir")
for engine in "${selected[@]}"; do
  rebucket_args+=(--engine "$engine")
done
python "${script_dir}/rebucket_runtime_configs.py" "${rebucket_args[@]}"
