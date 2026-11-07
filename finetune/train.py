"""LoRA / QLoRA fine-tune of an uncensored Qwen3.5 model on the recorded LocalForge browser conversations.

    python finetune/train.py --base huihui-ai/Huihui-Qwen3.5-9B-abliterated --data finetune/data/train.jsonl --out runs/lf-browser

Runs anywhere with a CUDA GPU (Modal: see modal_train.py). Writes:
    OUT/adapter/   the LoRA adapter
    OUT/merged/    the full model with the adapter merged in (bf16), ready for llama.cpp's convert_hf_to_gguf.py

Each training example is one model call from record.py: the prompt is rendered with the model's own chat template,
exactly as llama.cpp renders it for LocalForge (enable_thinking on or off, like the app asked), and the loss is
only on the reply (plus the short thinking when the app asked for thinking).
"""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

TARGETS = r".*(language_model|model\.layers).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"


def load_examples(path, tok, max_len, log):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out, skipped = [], 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=r["think"])
        reply = ((r["thought"].strip() + "\n</think>\n\n") if r["think"] else "") + r["reply"].strip() + "<|im_end|>\n"
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(reply, add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len:
            skipped += 1
            continue
        out.append((p, c))
    toks = sum(len(p) + len(c) for p, c in out)
    log(f"{len(out)} examples, {toks:,} tokens ({toks // max(1, len(out)):,} per example); {skipped} skipped as longer than {max_len}")
    return out


def load_model(base, qlora, dtype):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
    kw = {"dtype": dtype, "low_cpu_mem_usage": True}
    if qlora:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
        kw["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    try:
        # Keep the full model (with its vision part) so the merged model converts to GGUF + mmproj like the original.
        return AutoModelForImageTextToText.from_pretrained(base, **kw)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(base, **kw)


def train(a, log=print):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    data = load_examples(a.data, tok, a.max_len, log)
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    model = load_model(a.base, a.qlora and cuda, dtype)
    model.config.use_cache = False
    if a.qlora and cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(data) / a.accum)
    total = a.max_steps or steps_per_epoch * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    if a.qlora and cuda:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    dev = next(model.parameters()).device
    model.train()
    step, t0, seen_tok, run_loss = 0, time.time(), 0, 0.0
    order = []
    while step < total:
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            if not order:
                break
            p, c = data[order.pop()]
            ids = torch.tensor([p + c], device=dev)
            labels = torch.tensor([[-100] * len(p) + c], device=dev)
            loss = model(input_ids=ids, labels=labels).loss / a.accum
            loss.backward()
            run_loss += loss.item()
            seen_tok += len(p) + len(c)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % a.log_every == 0 or step == total:
            el = time.time() - t0
            log(f"step {step}/{total} loss {run_loss / a.log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{seen_tok / el:,.0f} tok/s, {el / 60:.1f} min, ~{(total - step) * el / step / 60:.0f} min left")
            run_loss = 0.0
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    log(f"adapter saved to {out / 'adapter'}")
    return out / "adapter"


def merge(base, adapter, dest, log=print):
    """Full-precision base + adapter -> merged model folder (with the base's tokenizer, template and processor files)."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    model = load_model(base, False, torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    dest = Path(dest)
    model.save_pretrained(dest, safe_serialization=True, max_shard_size="4GB")
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"]))
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json" and not (dest / f.name).exists():
            shutil.copy(f, dest / f.name)
    for f in ("config.json",):     # keep the base's exact config (architectures, vision settings) for the GGUF converter
        if (src / f).exists():
            shutil.copy(src / f, dest / f)
    log(f"merged model saved to {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="huihui-ai/Huihui-Qwen3.5-9B-abliterated")
    ap.add_argument("--data", default="finetune/data/train.jsonl")
    ap.add_argument("--out", default="runs/lf-browser")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qlora", action="store_true", help="load the base in 4-bit (less GPU memory)")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    adapter = train(a)
    if not a.no_merge:
        merge(a.base, adapter, Path(a.out) / "merged")


if __name__ == "__main__":
    main()
