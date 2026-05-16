#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
MEMORY_THRESHOLD_MIB="${MEMORY_THRESHOLD_MIB:-5000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-39527}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"

free_poll() {
  local used
  used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
  GPU_MEMORY_USED="$used" python - "$MEMORY_THRESHOLD_MIB" "${GPU_ARRAY[@]}" <<'PY'
import os
import sys

threshold = int(sys.argv[1])
targets = {int(x) for x in sys.argv[2:]}
seen = {}
for line in os.environ["GPU_MEMORY_USED"].splitlines():
    if not line.strip():
        continue
    idx_s, mem_s = [part.strip() for part in line.split(",", 1)]
    seen[int(idx_s)] = int(mem_s)

missing = sorted(targets - seen.keys())
busy = {idx: seen[idx] for idx in sorted(targets & seen.keys()) if seen[idx] >= threshold}
if missing or busy:
    if missing:
        print(f"missing GPUs: {missing}", file=sys.stderr)
    if busy:
        print("busy GPUs: " + ", ".join(f"{idx}={mem}MiB" for idx, mem in busy.items()), file=sys.stderr)
    raise SystemExit(1)
print("free GPUs: " + ", ".join(f"{idx}={seen[idx]}MiB" for idx in sorted(targets)), file=sys.stderr)
PY
}

stable_count=0
while (( stable_count < STABLE_POLLS )); do
  if free_poll; then
    stable_count=$((stable_count + 1))
    echo "[$(date)] GPU set $GPU_IDS free poll $stable_count/$STABLE_POLLS"
  else
    stable_count=0
    echo "[$(date)] GPU set $GPU_IDS not free; sleeping ${POLL_SECONDS}s"
  fi
  if (( stable_count < STABLE_POLLS )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date)] launching continuation training on GPUs $GPU_IDS"
CUDA_VISIBLE_DEVICES="$GPU_IDS" MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
  bash "$SCRIPT_DIR/train_reconstruction_rgb_stage1_lr5e5_continue_1k.sh"
