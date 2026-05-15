#!/usr/bin/env bash
set -euo pipefail

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export no_proxy="${no_proxy:-.byted.org,code.byted.org}"
export HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/opt/tiger/why-reca/.venv/bin/python}"
MODEL="${MODEL:-/mnt/hdfs/jixie/checkpoints/Z-Image/}"
METADATA="${METADATA:-/opt/tiger/why-reca/geneval/prompts/evaluation_metadata.jsonl}"
GENEVAL_DIR="${GENEVAL_DIR:-/opt/tiger/why-reca/geneval}"
MODEL_PATH="${MODEL_PATH:-/opt/tiger/why-reca/third_party/reconstruction-alignment/Benchmark/geneval/model}"
if [[ ! -d "$MODEL_PATH" ]]; then
  MODEL_PATH="/opt/tiger/why-reca/geneval/<OBJECT_DETECTOR_FOLDER>"
fi

LORA="${LORA:?Set LORA to a Z-Image LoRA directory}"
PHASE="${PHASE:-finetuned}"
OUT_ROOT="${OUT_ROOT:-/opt/tiger/why-reca/outputs/geneval_zimage_lora_25step_512}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STEPS="${STEPS:-25}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
NUM_IMAGES="${NUM_IMAGES:-4}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
LIMIT_ARGS=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"
OUT_DIR="$OUT_ROOT/$PHASE"
mkdir -p "$OUT_DIR" "$OUT_ROOT/eval_logs" "$OUT_ROOT/eval_results"

echo "[$(date)] generating $PHASE"
echo "model=$MODEL"
echo "lora=$LORA"
echo "out=$OUT_DIR"
echo "gpus=$GPUS shards=$NUM_GPUS steps=$STEPS cfg=$GUIDANCE_SCALE size=${HEIGHT}x${WIDTH}"

pids=()
for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  (
    cd "$REPO_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" "$PYTHON" \
      "$SCRIPT_DIR/generate_geneval_zimage.py" \
      --model "$MODEL" \
      --metadata "$METADATA" \
      --out-dir "$OUT_DIR" \
      --device cuda:0 \
      --steps "$STEPS" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-images-per-prompt "$NUM_IMAGES" \
      --num-shards "$NUM_GPUS" \
      --shard-index "$idx" \
      --lora "$LORA" \
      "${LIMIT_ARGS[@]}"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[$(date)] generation finished for $PHASE"

echo "[$(date)] scoring $PHASE"
source "/home/tiger/miniconda3/etc/profile.d/conda.sh"
conda activate geneval
cd "$GENEVAL_DIR"

outfile="$OUT_ROOT/eval_results/${PHASE}_results.jsonl"
summary="$OUT_ROOT/eval_results/${PHASE}_summary.txt"
score_log="$OUT_ROOT/eval_logs/${PHASE}_eval.log"

for port in 39941 39942 39943 39944 39945 39946; do
  echo "[$(date)] trying score port $port"
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$port" \
    evaluation/evaluate_images_mp.py "$OUT_DIR" \
    --outfile "$outfile" \
    --model-path "$MODEL_PATH" \
    |& tee "$score_log"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -eq 0 ]]; then
    python evaluation/summary_scores.py "$outfile" |& tee "$summary"
    echo "[$(date)] scoring finished for $PHASE"
    exit 0
  fi
  if ! rg -q "EADDRINUSE|Address already in use" "$score_log"; then
    echo "[$(date)] scoring failed with non-port error, status=$status"
    exit "$status"
  fi
done

echo "[$(date)] failed: all scoring ports were occupied" >&2
exit 1
