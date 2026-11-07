#!/usr/bin/env bash
# Trains, merges and converts the LocalForge browser model on this PC (WSL2 + RTX 3070). Close LocalForge first so
# the GPU's 8 GB are free.   wsl bash finetune/local/run_wsl.sh [test|train|export|all]
#   test    ~15 min: 6 training steps, prints GPU memory and speed, then stops (nothing is kept)
#   train   the full training run (resumes by itself if it was stopped)
#   export  merge + GGUF (Q4_K_M, Q5_K_M, mmproj) + copy into LocalForge's models folder
set -euo pipefail
cd "$(dirname "$0")/../.."
. /opt/lf-ft/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE="${LF_BASE:-huihui-ai/Huihui-Qwen3.5-9B-abliterated}"
NAME="${LF_NAME:-Qwen3.5-9B-abliterated-LocalForge-Browser}"
RUN="${LF_RUN:-$HOME/lf-runs/lf-browser}"
MODELS="${LF_MODELS:-/mnt/c/Users/Big_P/AppData/Local/LocalForge/models}"
MAXLEN="${LF_MAXLEN:-12288}"
STEP="${1:-all}"
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit !($1 > 1500)}'; then
  echo "The GPU already has $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use. Close LocalForge first."; exit 1
fi
if [ "$STEP" = test ]; then
  python finetune/train.py --base "$BASE" --data finetune/data/train.jsonl.gz --out "$RUN-test" --qlora --low-vram \
    --max-len "$MAXLEN" --max-steps 6 --accum 1 --log-every 1 --save-every 0 --no-merge
  rm -rf "$RUN-test"; exit 0
fi
if [ "$STEP" = train ] || [ "$STEP" = all ]; then
  RESUME=""; [ -f "$RUN/adapter/progress.json" ] && RESUME="--resume"
  python finetune/train.py --base "$BASE" --data finetune/data/train.jsonl.gz --out "$RUN" --qlora --low-vram \
    --max-len "$MAXLEN" --epochs 2 --accum 8 --log-every 2 --save-every 5 --no-merge $RESUME 2>&1 | tee -a "$RUN.log"
fi
if [ "$STEP" = export ] || [ "$STEP" = all ]; then
  python finetune/train.py --base "$BASE" --out "$RUN" --merge-only
  mkdir -p "$RUN/gguf"
  python /opt/llama.cpp/convert_hf_to_gguf.py "$RUN/merged" --outtype bf16 --outfile "$RUN/gguf/$NAME-BF16.gguf"
  python /opt/llama.cpp/convert_hf_to_gguf.py "$RUN/merged" --mmproj --outtype f16 --outfile "$RUN/gguf/mmproj-$NAME-f16.gguf" || true
  for q in Q4_K_M Q5_K_M; do /opt/llama.cpp/build/bin/llama-quantize "$RUN/gguf/$NAME-BF16.gguf" "$RUN/gguf/$NAME-$q.gguf" "$q"; done
  rm -f "$RUN/gguf/$NAME-BF16.gguf"
  rm -rf "$RUN/merged"          # 18 GB; the adapter is kept, so it can be merged again any time
  cp "$RUN/gguf/"*.gguf "$MODELS/"
  ls -la "$MODELS" | grep -i "$NAME" || true
  echo "Done: the model is in LocalForge's models folder."
fi
