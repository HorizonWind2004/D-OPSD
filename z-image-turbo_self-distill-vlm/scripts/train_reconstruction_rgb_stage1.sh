#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/tiger/RGB/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/opt/tiger/RGB/.venv/bin/accelerate}"

cd "${REPO_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 "${ACCELERATE_BIN}" launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT:-39501}" \
  train_reconstruction.py \
  --pretrained-model "/mnt/hdfs/jixie/checkpoints/Z-Image/" \
  --image-root "/mnt/hdfs/jixie/RGB_stage1/images/" \
  --output-dir "/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/" \
  --exp-name "rgb_stage1_zimage_reca_style_lora" \
  --resolution 256 \
  --vl-resolution 224 \
  --image-count 100000 \
  --image-name-template "{index:08d}.png" \
  --max-image-load-retries 1000 \
  --reca-prompts-path "/opt/tiger/why-reca/train/src/qflux/data/reca_prompts.py" \
  --max-reca-prompts 360 \
  --max-train-samples 100000 \
  --max-train-steps 2000 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --mixed-precision bf16 \
  --vae-dtype bf16 \
  --sample-steps 100 \
  --checkpoint-steps 500 \
  --log-steps 1 \
  --sample-num-images 4 \
  --sample-inference-steps 28 \
  --sample-guidance-scale 4.0 \
  --num-workers 4 \
  --seed 30
