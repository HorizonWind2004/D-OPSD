#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export no_proxy="${no_proxy:-.byted.org,code.byted.org}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCELERATE_BIN="${ACCELERATE_BIN:-/opt/tiger/RGB/.venv/bin/accelerate}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "${REPO_DIR}"

"${ACCELERATE_BIN}" launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT:-39541}" \
  train_reconstruction.py \
  --pretrained-model "/mnt/hdfs/jixie/checkpoints/Z-Image/" \
  --image-root "/mnt/hdfs/jixie/RGB_stage1/images/" \
  --output-dir "/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/" \
  --exp-name "rgb_stage1_zimage_reca_style_lora_lr5e5_2k_res1024" \
  --train-mode lora \
  --resolution 1024 \
  --vl-resolution 224 \
  --image-count 100000 \
  --image-name-template "{index:08d}.png" \
  --max-image-load-retries 1000 \
  --reca-prompts-path "/opt/tiger/why-reca/train/src/qflux/data/reca_prompts.py" \
  --max-reca-prompts 360 \
  --max-train-samples 100000 \
  --max-train-steps 2000 \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-5 \
  --adam-weight-decay 0.0 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --num-workers 2 \
  --mixed-precision bf16 \
  --vae-dtype bf16 \
  --enable-gc \
  --log-steps 1 \
  --sample-steps 100 \
  --checkpoint-steps 100 \
  --sample-num-images 2 \
  --sample-inference-steps 28 \
  --sample-guidance-scale 4.0
