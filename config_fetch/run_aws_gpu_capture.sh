#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Copy this repo to an already-running GPU EC2 instance, run Dynamo config
capture there, copy generated files back locally, and stop the instance after
successful completion.

This script does not run Terraform and will not create, start, or replace
instances. Run config_fetch/provision_aws_gpu_capture.sh first when the
instance does not exist or is stopped.

Usage:
  config_fetch/run_aws_gpu_capture.sh [--tag 1.1.1] [--engine vllm]

Options:
  --tag TAG              Dynamo runtime image tag. Default: 1.1.1
  --engine ENGINE        Component to capture. Repeatable. Defaults to all.
  --region REGION        AWS region. Default: us-west-2
  --instance-id ID       Existing GPU instance ID. Defaults to latest tagged
                         Dynamo capture instance.
  --ssh-user USER        SSH user for the instance. Default: ubuntu
  --out-dir DIR          Local output root. Default: config_fetch/dynamo_runtime
  --pull-jobs N          Parallel docker pulls on the EC2 host. Default: 3
  --skip-pull            Use images already cached on the EC2 host.
  --no-stop              Leave the instance running after successful capture.
  -h, --help             Show this help.

Environment:
  DYNAMO_DOCKER_FLAGS    Docker flags for remote docker run.
                         Default: "--gpus all --network host --rm"
  DYNAMO_FRONTEND_IMAGE  Optional frontend runtime image override.
  DYNAMO_VLLM_OMNI_IMAGE Optional vllm_omni runtime image override.
EOF
}

tag="${DYNAMO_IMAGE_TAG:-1.1.1}"
region="us-west-2"
instance_id=""
ssh_user="ubuntu"
out_dir=""
pull_jobs="${DYNAMO_PULL_JOBS:-3}"
skip_pull=0
stop_after=1
engines=()

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
    --region)
      region="${2:?missing value for --region}"
      shift 2
      ;;
    --instance-id)
      instance_id="${2:?missing value for --instance-id}"
      shift 2
      ;;
    --ssh-user)
      ssh_user="${2:?missing value for --ssh-user}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --pull-jobs|--jobs)
      pull_jobs="${2:?missing value for --pull-jobs}"
      shift 2
      ;;
    --skip-pull|--no-pull)
      skip_pull=1
      shift
      ;;
    --no-stop)
      stop_after=0
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
aws_dir="${script_dir}/aws_gpu_capture"
generated_dir="${aws_dir}/.generated"
key_path="${generated_dir}/dynamo-config-capture-ed25519"
out_dir="${out_dir:-${script_dir}/dynamo_runtime}"

log() {
  printf '\n==> %s\n' "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "${cmd} is required but was not found in PATH" >&2
    exit 127
  fi
}

shell_join() {
  printf '%q ' "$@"
}

stop_instance() {
  if [[ "$stop_after" -ne 1 || -z "$instance_id" ]]; then
    return
  fi

  log "Stopping EC2 instance ${instance_id}; Docker image cache will remain on its EBS volume"
  aws ec2 stop-instances --region "$region" --instance-ids "$instance_id" >/dev/null
  aws ec2 wait instance-stopped --region "$region" --instance-ids "$instance_id"
  log "Instance ${instance_id} is stopped"
}

require_cmd aws
require_cmd ssh
require_cmd rsync

if ! [[ "$pull_jobs" =~ ^[0-9]+$ ]] || [[ "$pull_jobs" -lt 1 ]]; then
  echo "--pull-jobs must be a positive integer" >&2
  exit 2
fi

if [[ ! -f "$key_path" ]]; then
  echo "Missing SSH key ${key_path}; run config_fetch/provision_aws_gpu_capture.sh first" >&2
  exit 2
fi

if [[ -z "$instance_id" ]]; then
  instance_id="$(aws ec2 describe-instances \
    --region "$region" \
    --filters \
      'Name=tag:Component,Values=dynamo-config-capture' \
      'Name=instance-state-name,Values=pending,running,stopping,stopped' \
    --query 'sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId' \
    --output text)"
  if [[ "$instance_id" == "None" ]]; then
    instance_id=""
  fi
fi

if [[ -z "$instance_id" ]]; then
  echo "No existing Dynamo capture instance found. Run config_fetch/provision_aws_gpu_capture.sh first." >&2
  exit 2
fi

state="$(aws ec2 describe-instances \
  --region "$region" \
  --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"

if [[ "$state" == "stopped" ]]; then
  echo "Instance ${instance_id} is stopped. Run config_fetch/provision_aws_gpu_capture.sh first." >&2
  exit 2
fi
if [[ "$state" == "terminated" || "$state" == "shutting-down" ]]; then
  echo "Instance ${instance_id} is ${state}. Run config_fetch/provision_aws_gpu_capture.sh first." >&2
  exit 2
fi

log "Waiting for existing EC2 instance ${instance_id} to be ready"
aws ec2 wait instance-running --region "$region" --instance-ids "$instance_id"
aws ec2 wait instance-status-ok --region "$region" --instance-ids "$instance_id"

public_ip="$(aws ec2 describe-instances \
  --region "$region" \
  --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

if [[ -z "$public_ip" || "$public_ip" == "None" ]]; then
  echo "Instance ${instance_id} has no public IP; run provision/start first." >&2
  exit 2
fi

ssh_opts=(
  -i "$key_path"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="${generated_dir}/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
remote="${ssh_user}@${public_ip}"
remote_dir="/home/${ssh_user}/test_commits"

log "Waiting for SSH on ${remote}"
for attempt in $(seq 1 60); do
  if ssh "${ssh_opts[@]}" -o ConnectTimeout=10 "$remote" "echo ssh-ready" >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 60 ]]; then
    echo "SSH did not become ready on ${remote}" >&2
    exit 1
  fi
  sleep 10
done

log "Checking Docker and GPU runtime on ${remote}"
ssh "${ssh_opts[@]}" "$remote" '
  set -Eeuo pipefail
  sudo systemctl enable --now docker
  nvidia-smi
  sudo docker version
  df -h / /var/lib/docker || true
'

log "Syncing repo to ${remote}:${remote_dir}"
ssh "${ssh_opts[@]}" "$remote" "rm -rf $(printf '%q' "$remote_dir") && mkdir -p $(printf '%q' "$remote_dir")"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.terraform/' \
  --exclude '*.tfstate' \
  --exclude '*.tfstate.*' \
  --exclude 'config_fetch/aws_gpu_capture/.generated/' \
  -e "$(shell_join ssh "${ssh_opts[@]}")" \
  "${repo_root}/" \
  "${remote}:${remote_dir}/"

remote_capture_args=(config_fetch/capture_dynamo_runtime_configs.sh --tag "$tag" --out-dir config_fetch/dynamo_runtime --jobs "$pull_jobs")
for engine in "${engines[@]}"; do
  remote_capture_args+=(--engine "$engine")
done
if [[ "$skip_pull" -eq 1 ]]; then
  remote_capture_args+=(--skip-pull)
fi

remote_env=(
  "DYNAMO_DOCKER_FLAGS=${DYNAMO_DOCKER_FLAGS:---gpus all --network host --rm}"
)
if [[ -n "${DYNAMO_FRONTEND_IMAGE:-}" ]]; then
  remote_env+=("DYNAMO_FRONTEND_IMAGE=${DYNAMO_FRONTEND_IMAGE}")
fi
if [[ -n "${DYNAMO_VLLM_OMNI_IMAGE:-}" ]]; then
  remote_env+=("DYNAMO_VLLM_OMNI_IMAGE=${DYNAMO_VLLM_OMNI_IMAGE}")
fi

remote_env_cmd="$(shell_join "${remote_env[@]}")"
remote_capture_cmd="$(shell_join "${remote_capture_args[@]}")"

log "Running Dynamo config capture on ${remote}"
ssh "${ssh_opts[@]}" "$remote" "
  set -Eeuo pipefail
  cd $(printf '%q' "$remote_dir")
  sudo -E env ${remote_env_cmd} bash -lc $(printf '%q' "$remote_capture_cmd")
  sudo chown -R ${ssh_user}:${ssh_user} config_fetch/dynamo_runtime
"

log "Copying generated config files back to ${out_dir}"
mkdir -p "$out_dir"
rsync -az --delete \
  -e "$(shell_join ssh "${ssh_opts[@]}")" \
  "${remote}:${remote_dir}/config_fetch/dynamo_runtime/" \
  "${out_dir}/"

log "Capture completed; local files are in ${out_dir}"
stop_instance
