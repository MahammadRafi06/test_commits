#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Create or start the Terraform-managed GPU EC2 instance for Dynamo config capture.

This script only handles infrastructure readiness. It does not copy repo files
or run config capture.

Usage:
  config_fetch/provision_aws_gpu_capture.sh

Options:
  --region REGION        AWS region. Default: us-west-2
  --instance-type TYPE   GPU instance type. Default: g4dn.xlarge
  --volume-size-gb GB    Root EBS size. Default: 300
  -h, --help             Show this help.
EOF
}

region="us-west-2"
instance_type="g4dn.xlarge"
volume_size_gb="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      region="${2:?missing value for --region}"
      shift 2
      ;;
    --instance-type)
      instance_type="${2:?missing value for --instance-type}"
      shift 2
      ;;
    --volume-size-gb)
      volume_size_gb="${2:?missing value for --volume-size-gb}"
      shift 2
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
tf_dir="${script_dir}/aws_gpu_capture"
generated_dir="${tf_dir}/.generated"
key_path="${generated_dir}/dynamo-config-capture-ed25519"
ssh_user="ubuntu"

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

require_cmd aws
require_cmd terraform
require_cmd ssh
require_cmd curl
require_cmd ssh-keygen

if ! [[ "$volume_size_gb" =~ ^[0-9]+$ ]] || [[ "$volume_size_gb" -lt 200 ]]; then
  echo "--volume-size-gb must be an integer >= 200; Dynamo images are large" >&2
  exit 2
fi

mkdir -p "$generated_dir"
if [[ ! -f "$key_path" ]]; then
  log "Generating local SSH key for Terraform-managed EC2 access"
  ssh-keygen -q -t ed25519 -N "" -f "$key_path"
fi
chmod 600 "$key_path"

log "Detecting local public IP for SSH security group"
local_ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
ssh_cidr="${local_ip}/32"

log "Applying Terraform in ${tf_dir}"
terraform -chdir="$tf_dir" init -input=false
terraform -chdir="$tf_dir" apply -auto-approve -input=false \
  -var "region=${region}" \
  -var "instance_type=${instance_type}" \
  -var "root_volume_size_gb=${volume_size_gb}" \
  -var "ssh_cidr=${ssh_cidr}" \
  -var "public_key_path=${key_path}.pub"

instance_id="$(terraform -chdir="$tf_dir" output -raw instance_id)"
ssh_user="$(terraform -chdir="$tf_dir" output -raw ssh_user)"

state="$(aws ec2 describe-instances \
  --region "$region" \
  --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"

if [[ "$state" == "stopped" ]]; then
  log "Starting EC2 instance ${instance_id}"
  aws ec2 start-instances --region "$region" --instance-ids "$instance_id" >/dev/null
fi

log "Waiting for EC2 instance ${instance_id} to be ready"
aws ec2 wait instance-running --region "$region" --instance-ids "$instance_id"
aws ec2 wait instance-status-ok --region "$region" --instance-ids "$instance_id"

public_ip="$(aws ec2 describe-instances \
  --region "$region" \
  --instance-ids "$instance_id" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

ssh_opts=(
  -i "$key_path"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="${generated_dir}/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
remote="${ssh_user}@${public_ip}"

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

log "Checking Docker, GPU, and disk on ${remote}"
ssh "${ssh_opts[@]}" "$remote" '
  set -Eeuo pipefail
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y docker.io
  fi
  sudo systemctl enable --now docker
  nvidia-smi
  sudo docker version
  df -h / /var/lib/docker || true
'

cat <<EOF

Ready:
  instance_id: ${instance_id}
  public_ip:   ${public_ip}
  ssh:         ssh -i ${key_path} ${remote}

Run capture:
  config_fetch/run_aws_gpu_capture.sh --skip-pull
EOF
