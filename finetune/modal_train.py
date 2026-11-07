"""Fine-tunes the LocalForge browser model on a Modal GPU, turns it into a GGUF for the PC, and checks it.

    modal run finetune/modal_train.py                      # train + export + check (about 1.5-2.5 h on an L40S)
    modal run finetune/modal_train.py --step export        # only redo the GGUF export
    modal run finetune/modal_train.py --step check         # only re-run the checks
    modal volume get localforge-finetune lf-browser/gguf/ .   # download the result to the PC

Everything stays in your own Modal account (Volume "localforge-finetune"). Nothing goes to an AI company's API.
Billing: only while a container runs. An L40S is about $1.95/h, so a full run is roughly $3-5 of the free credit.

Settings (environment variables):
    LF_BASE    the uncensored model to fine-tune (default huihui-ai/Huihui-Qwen3.5-9B-abliterated)
    LF_NAME    output name (must keep "abliterated" or "uncensored" so LocalForge lists it)
    LF_GPU     L40S (default), A100-80GB, H100
    LF_EPOCHS  1 (default)
    LF_DATA    finetune/data/train.jsonl.gz
"""
import os
import subprocess
from pathlib import Path

import modal

BASE = os.environ.get("LF_BASE", "huihui-ai/Huihui-Qwen3.5-9B-abliterated")
NAME = os.environ.get("LF_NAME", "Qwen3.5-9B-abliterated-LocalForge-Browser")
GPU = os.environ.get("LF_GPU", "L40S")
EPOCHS = os.environ.get("LF_EPOCHS", "1")
DATA = os.environ.get("LF_DATA", str(Path(__file__).parent / "data" / "train.jsonl.gz"))
QUANTS = ["Q4_K_M", "Q5_K_M"]          # Q4_K_M (~5.6 GB) fits an 8 GB card with room for context; Q5_K_M is ~6.5 GB
VOL = "/vol"
RUN = f"{VOL}/lf-browser"

if not any(w in (BASE + NAME).lower() for w in ("abliterat", "uncensored")):
    raise SystemExit("LocalForge only runs uncensored / abliterated models: LF_BASE and LF_NAME must say so.")

app = modal.App("localforge-finetune")
volume = modal.Volume.from_name("localforge-finetune", create_if_missing=True)
HERE = Path(__file__).parent

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.20.0", "peft==0.21.1", "accelerate==1.14.0", "bitsandbytes==0.50.1",
                 "huggingface_hub[hf_transfer]", "safetensors", "packaging", "wheel")
    # Fast kernels for Qwen3.5's linear-attention layers. Without them training is several times slower and uses far more memory.
    .pip_install("flash-linear-attention==0.5.1")
    .run_commands("pip install --no-build-isolation causal-conv1d==1.6.1 || echo 'causal-conv1d build failed: slower fallback'",
                  gpu="L4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{VOL}/hf"})
    .add_local_file(HERE / "train.py", "/root/train.py")
)

# llama.cpp's official image with its converter and quantizer.
gguf_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:full-cuda", add_python="3.12")
    .entrypoint([])
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.20.0", "numpy", "sentencepiece", "protobuf", "safetensors", "huggingface_hub")
    .run_commands("pip install /app/gguf-py")
    .env({"HF_HOME": f"{VOL}/hf"})
)

check_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
    .add_local_file(HERE / "check_model.py", "/root/check_model.py")
)


@app.function(image=train_image, gpu=GPU, volumes={VOL: volume}, timeout=6 * 3600)
def train(epochs: str = EPOCHS):
    import gzip
    import shutil
    import sys
    data = Path(f"{RUN}/train.jsonl")
    with gzip.open(f"{RUN}/train.jsonl.gz", "rb") as src, open(data, "wb") as dst:
        shutil.copyfileobj(src, dst)
    sys.argv = ["train.py", "--base", BASE, "--data", str(data), "--out", RUN, "--epochs", epochs, "--qlora",
                "--max-len", "12288", "--log-every", "10"]
    sys.path.insert(0, "/root")
    import train as t
    t.main()
    volume.commit()


@app.function(image=gguf_image, cpu=8, memory=65536, volumes={VOL: volume}, timeout=3 * 3600)
def export():
    out = Path(f"{RUN}/gguf")
    out.mkdir(parents=True, exist_ok=True)
    conv = "/app/convert_hf_to_gguf.py"
    quant = next(p for p in ("/app/llama-quantize", "/app/build/bin/llama-quantize") if Path(p).exists())
    bf16 = out / f"{NAME}-BF16.gguf"
    subprocess.run(["python", conv, f"{RUN}/merged", "--outtype", "bf16", "--outfile", str(bf16)], check=True)
    # The vision file, so LocalForge can still look at pictures with the fine-tuned model.
    subprocess.run(["python", conv, f"{RUN}/merged", "--mmproj", "--outtype", "f16",
                    "--outfile", str(out / f"mmproj-{NAME}-f16.gguf")], check=False)
    for q in QUANTS:
        subprocess.run([quant, str(bf16), str(out / f"{NAME}-{q}.gguf"), q], check=True)
    bf16.unlink()
    volume.commit()
    for f in sorted(out.iterdir()):
        print(f"{f.name}  {f.stat().st_size / 1e9:.2f} GB")


@app.function(image=check_image, gpu="L4", volumes={VOL: volume}, timeout=3600)
def check():
    import sys
    sys.path.insert(0, "/root")
    import check_model
    return check_model.run(f"{RUN}/gguf/{NAME}-Q4_K_M.gguf", f"{RUN}/train.jsonl", base_url_port=8080)


@app.local_entrypoint()
def main(step: str = "all"):
    if step in ("all", "train"):
        with volume.batch_upload(force=True) as b:
            b.put_file(DATA, "lf-browser/train.jsonl.gz")
        print(f"Training {BASE} on {GPU} …")
        train.remote()
    if step in ("all", "train", "export"):
        print("Making the GGUF files …")
        export.remote()
    if step in ("all", "train", "export", "check"):
        print("Checking the model (tags, scripts, refusals) …")
        print(check.remote())
    print("\nDownload to the PC:  modal volume get localforge-finetune lf-browser/gguf/ .\n"
          "Then put the .gguf (and the mmproj file) in %LOCALAPPDATA%\\LocalForge\\models and load it in LocalForge.")
