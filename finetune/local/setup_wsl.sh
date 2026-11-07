#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu on the PC: Python tools for training on the RTX 3070, and llama.cpp's converter
# + quantizer for making the GGUF. Everything stays on this PC. Run as root:  wsl -u root bash finetune/local/setup_wsl.sh
set -euo pipefail
LLAMA_TAG="${LLAMA_TAG:-b10549}"      # same llama.cpp build as LocalForge's engines
ENV=/opt/lf-ft
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev build-essential cmake git curl >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -m venv "$ENV"
. "$ENV/bin/activate"
pip install -q --upgrade pip wheel packaging ninja
pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers==5.20.0 peft==0.21.1 accelerate==1.14.0 bitsandbytes==0.50.1 \
    "huggingface_hub[hf_transfer]" safetensors sentencepiece protobuf numpy
# Fast kernels for Qwen3.5's linear-attention layers (much less memory and time than the fallback).
pip install -q flash-linear-attention==0.5.1
MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d==1.6.1 || echo "causal-conv1d didn't build: training still works, a bit slower"
python - <<'PY'
import torch; print("torch", torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
# llama.cpp at LocalForge's build: convert_hf_to_gguf.py + llama-quantize (CPU build is enough for quantizing).
if [ ! -d /opt/llama.cpp ]; then git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp /opt/llama.cpp; fi
pip install -q /opt/llama.cpp/gguf-py
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build /opt/llama.cpp/build --target llama-quantize -j 4 >/dev/null
ls -la /opt/llama.cpp/build/bin/llama-quantize
echo "Setup done."
