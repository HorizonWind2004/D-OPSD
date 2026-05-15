#!/usr/bin/env bash
set -euo pipefail

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export no_proxy="${no_proxy:-.byted.org,code.byted.org}"
export PYTHONUNBUFFERED=1

EXP_ROOT="${EXP_ROOT:-/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/rgb_stage1_zimage_reca_style_lora_lr5e5_1k}"
OUT_ROOT="${OUT_ROOT:-/opt/tiger/why-reca/outputs/geneval_zimage_recon_lr5e5_1k_25step_512}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_adapter_dir() {
  local root="$1"
  if [[ -f "$root/adapter_config.json" ]]; then
    printf "%s\n" "$root"
    return 0
  fi
  if [[ -f "$root/recon/adapter_config.json" ]]; then
    printf "%s\n" "$root/recon"
    return 0
  fi
  return 1
}

wait_for_adapter() {
  local label="$1"
  local primary="$2"
  local fallback="${3:-}"

  echo "[$(date)] waiting for $label" >&2
  echo "primary=$primary" >&2
  if [[ -n "$fallback" ]]; then
    echo "fallback=$fallback" >&2
  fi

  while true; do
    if adapter_dir="$(find_adapter_dir "$primary")"; then
      printf "%s\n" "$adapter_dir"
      return 0
    fi
    if [[ -n "$fallback" ]] && adapter_dir="$(find_adapter_dir "$fallback")"; then
      printf "%s\n" "$adapter_dir"
      return 0
    fi
    echo "[$(date)] $label not ready; sleeping ${POLL_SECONDS}s" >&2
    sleep "$POLL_SECONDS"
  done
}

run_geneval_for_adapter() {
  local label="$1"
  local adapter="$2"
  echo "[$(date)] running Geneval for $label: $adapter"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    LORA="$adapter" \
    PHASE="$label" \
    OUT_ROOT="$OUT_ROOT" \
    bash "$SCRIPT_DIR/run_geneval_zimage_lora_512_25.sh"
}

# Wait for 1k first so Geneval does not compete with the active training job.
step1000_adapter="$(wait_for_adapter \
  "step1000/final adapter" \
  "$EXP_ROOT/checkpoints/step_001000/recon" \
  "$EXP_ROOT/checkpoints/final/recon")"

step500_adapter="$(wait_for_adapter \
  "step500 adapter" \
  "$EXP_ROOT/checkpoints/step_000500/recon")"

run_geneval_for_adapter "finetuned_lr5e5_step500" "$step500_adapter"
run_geneval_for_adapter "finetuned_lr5e5_step1000" "$step1000_adapter"

echo "[$(date)] completed Geneval for step500 and step1000"
