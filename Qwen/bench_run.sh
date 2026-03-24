#!/bin/bash
set -euo pipefail

concurrency_array=(1 2 8 16 32 64 128 1459 1536 1612)

ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-/tmp/bench_artifacts}"

for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"
  aiperf profile \
    --artifact-dir "${ARTIFACT_DIR}/concurrency_${concurrency}" \
    -m Qwen/Qwen3-32B-FP8 \
    --endpoint-type chat \
    -u http://0.0.0.0:8000 \
    --tokenizer Qwen/Qwen3-32B-FP8 \
    --isl 4000 --isl-stddev 0 \
    --osl 1000 --osl-stddev 0 \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${concurrency} \
    --num-requests $(($concurrency*50)) \
    --warmup-request-count $(($concurrency*2)) \
    --random-seed 100 \
    --ui simple \
    --streaming
done