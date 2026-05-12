#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Pull NVIDIA Dynamo runtime images and capture config catalogs locally.

This is the end-to-end automation entrypoint. It pulls the selected images,
runs the config_fetch generators inside the matching runtime containers, copies
the generated files into a local output folder, and creates role-specific
prefill/decode catalog views.

Usage:
  config_fetch/capture_dynamo_runtime_configs.sh [--tag 1.1.1] [--engine vllm]

Options:
  --tag TAG         Dynamo runtime image tag to use. Default: 1.1.1
  --engine ENGINE  Capture one component. Repeatable. One of:
                   vllm, vllm_omni, sglang, tensorrt_llm, frontend, all
                   Default: all supported components.
  --out-dir DIR    Local output root. Default: config_fetch/dynamo_runtime
  --jobs N         Parallel docker pulls. Default: DYNAMO_PULL_JOBS or 3.
  --skip-pull      Do not pull images before running containers.
  --dry-run        Print the images and generator command without running them.
  -h, --help       Show this help.

Environment:
  DYNAMO_DOCKER_FLAGS   Docker flags passed to docker run by the generator.
                        Default: "--gpus all --network host --rm"
  DYNAMO_FRONTEND_IMAGE Image to use for frontend config capture.
                        Default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<--tag>
  DYNAMO_VLLM_OMNI_IMAGE Image to use for vllm_omni config capture.
                        Default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:<--tag>
  DYNAMO_PULL_JOBS      Default parallel pull count.

Examples:
  config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1
  config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --engine vllm --engine sglang
  config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --out-dir /tmp/dynamo-configs
  config_fetch/capture_dynamo_runtime_configs.sh --tag 1.1.1 --skip-pull
EOF
}

tag="${DYNAMO_IMAGE_TAG:-1.1.1}"
out_dir=""
pull=1
dry_run=0
pull_jobs="${DYNAMO_PULL_JOBS:-3}"
requested_engines=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="${2:?missing value for --tag}"
      shift 2
      ;;
    --engine)
      requested_engines+=("${2:?missing value for --engine}")
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --jobs)
      pull_jobs="${2:?missing value for --jobs}"
      shift 2
      ;;
    --skip-pull|--no-pull)
      pull=0
      shift
      ;;
    --dry-run)
      dry_run=1
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
out_dir="${out_dir:-${script_dir}/dynamo_runtime}"
generator="${script_dir}/generate_dynamo_runtime_configs.sh"

canonical_engine() {
  case "$1" in
    all) echo "all" ;;
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

image_for_engine() {
  case "$1" in
    vllm) echo "nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}" ;;
    vllm_omni) echo "${DYNAMO_VLLM_OMNI_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}}" ;;
    sglang) echo "nvcr.io/nvidia/ai-dynamo/sglang-runtime:${tag}" ;;
    tensorrt_llm) echo "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:${tag}" ;;
    frontend) echo "${DYNAMO_FRONTEND_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:${tag}}" ;;
    *)
      echo "Unknown engine: $1" >&2
      exit 2
      ;;
  esac
}

validate_jobs() {
  if ! [[ "$pull_jobs" =~ ^[0-9]+$ ]] || [[ "$pull_jobs" -lt 1 ]]; then
    echo "--jobs must be a positive integer, got: ${pull_jobs}" >&2
    exit 2
  fi
}

pull_images() {
  local -a images=("$@")
  local running=0
  local status=0
  local image

  if [[ ${#images[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Pulling ${#images[@]} unique image(s) with ${pull_jobs} parallel job(s)..."
  for image in "${images[@]}"; do
    (
      echo "==> docker pull ${image}"
      docker pull "$image"
    ) &
    running=$((running + 1))

    if [[ "$running" -ge "$pull_jobs" ]]; then
      if ! wait -n; then
        status=1
      fi
      running=$((running - 1))
    fi
  done

  while [[ "$running" -gt 0 ]]; do
    if ! wait -n; then
      status=1
    fi
    running=$((running - 1))
  done

  return "$status"
}

print_summary() {
  local output_root="$1"
  echo
  echo "Generated local files under: ${output_root}"
  if [[ -d "$output_root" ]]; then
    find "$output_root" -maxdepth 2 -type f \
      \( -name '*.json' -o -name '*_help_*.txt' \) \
      | sort \
      | sed 's/^/  /'
  fi
}

validate_jobs

engines=()
if [[ ${#requested_engines[@]} -eq 0 ]]; then
  engines=(vllm vllm_omni sglang tensorrt_llm frontend)
else
  declare -A seen_engines=()
  for requested_engine in "${requested_engines[@]}"; do
    engine="$(canonical_engine "$requested_engine")"
    if [[ "$engine" == "all" ]]; then
      engines=(vllm vllm_omni sglang tensorrt_llm frontend)
      break
    fi
    if [[ -z "${seen_engines[$engine]:-}" ]]; then
      seen_engines["$engine"]=1
      engines+=("$engine")
    fi
  done
fi

images=()
declare -A seen_images=()
for engine in "${engines[@]}"; do
  image="$(image_for_engine "$engine")"
  if [[ -z "${seen_images[$image]:-}" ]]; then
    seen_images["$image"]=1
    images+=("$image")
  fi
done

generator_cmd=("$generator" --tag "$tag" --out-dir "$out_dir")
for engine in "${engines[@]}"; do
  generator_cmd+=(--engine "$engine")
done

echo "Dynamo tag: ${tag}"
echo "Engines: ${engines[*]}"
echo "Output root: ${out_dir}"
echo "Docker run flags: ${DYNAMO_DOCKER_FLAGS:---gpus all --network host --rm}"
echo
echo "Images:"
printf '  %s\n' "${images[@]}"

if [[ "$dry_run" -eq 1 ]]; then
  echo
  echo "Generator command:"
  printf '  %q' "${generator_cmd[@]}"
  echo
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH" >&2
  exit 127
fi

if [[ "$pull" -eq 1 ]]; then
  pull_images "${images[@]}"
else
  echo "Skipping docker pull; using local images."
fi

echo
echo "Running config capture containers..."
"${generator_cmd[@]}"

print_summary "$out_dir"
