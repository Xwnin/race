"""LoRA / QLoRA fine-tune of an uncensored Qwen3.5 model on the recorded LocalForge browser conversations.

On Harsh's PC (RTX 3070 8 GB, in WSL2, with LocalForge closed so the GPU is free):
    python finetune/train.py --qlora --low-vram --data finetune/data/train.jsonl.gz --out runs/lf-browser

Writes:
    OUT/adapter/   the LoRA adapter (saved every --save-every steps too, so a stopped run can be resumed with --resume)
    OUT/merged/    the base model with the adapter merged in (bf16), ready for llama.cpp's convert_hf_to_gguf.py

Data modes:
    --mode episodes (default)  one sequence per website task: the app's last call of the episode, exactly as the app
                               sent it, with the loss on EVERY reply in it. Plus each step where the app asked the
                               model to think (the planning step), so the short planning thought is learned too.
                               About 5x less compute than "steps", which is what makes an 8 GB card practical.
    --mode steps               one sequence per model call (exact context for every reply; for big GPUs).

The prompt is rendered with the model's own chat template, as llama.cpp renders it for LocalForge, and logits are
only computed where there's a loss (the vocabulary is 248k, so full logits for 10k tokens would need ~10 GB).
"""
import argparse
import gzip
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import torch

TARGETS = r".*(language_model|model\.layers).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"
ASSISTANT = re.compile(r"<\|im_start\|>assistant\n([\s\S]*?<\|im_end\|>)")


def read_rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def episodes(rows):
    eps, cur = [], []
    for r in rows:
        if r["step"] == 0 and cur:
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    return eps


def encode(tok, row, all_replies):
    """Token ids + which positions to learn: the final reply, and (all_replies) every earlier reply in the history."""
    prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=row["think"])
    reply = ((row["thought"].strip() + "\n</think>\n\n") if row["think"] else "") + row["reply"].strip() + "<|im_end|>\n"
    text = prompt + reply
    spans = [(len(prompt), len(text))]
    if all_replies:
        spans += [m.span(1) for m in ASSISTANT.finditer(prompt)]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    learn = [any(a <= s < b for a, b in spans) for s, _ in enc["offset_mapping"]]
    return ids, learn


def load_examples(path, tok, max_len, mode, log):
    rows = read_rows(path)
    picked = []
    if mode == "steps":
        picked = [(r, False) for r in rows]
    else:
        for ep in episodes(rows):
            picked.append((ep[-1], True))
            picked += [(r, False) for r in ep[:-1] if r["think"]]
    out, skipped = [], 0
    for r, allr in picked:
        ids, learn = encode(tok, r, allr)
        if len(ids) > max_len and allr:
            # Too long: fall back to the episode's longest earlier call that fits.
            ep = next(e for e in episodes(rows) if e[-1] is r)
            for cand in reversed(ep[:-1]):
                ids, learn = encode(tok, cand, True)
                if len(ids) <= max_len:
                    break
        if len(ids) > max_len:
            skipped += 1
            continue
        out.append((ids, learn))
    toks = sum(len(i) for i, _ in out)
    lt = sum(sum(l) for _, l in out)
    log(f"{len(out)} sequences ({mode}), {toks:,} tokens, {lt:,} of them learned; {skipped} skipped as longer than {max_len}")
    return out


def load_model(base, qlora, dtype, low_vram=False):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
    kw = {"dtype": dtype, "low_cpu_mem_usage": True}
    if qlora:
        skip = ["visual", "model.visual"] + ([] if low_vram else ["lm_head"])
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=skip,
                                                       llm_int8_enable_fp32_cpu_offload=low_vram)
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    if low_vram:
        # 8 GB cards: the 2 GB embedding table and the unused vision tower stay in system RAM; the lm_head is 4-bit.
        kw["device_map"] = {"model.language_model.embed_tokens": "cpu", "model.visual": "cpu", "model.language_model": 0,
                            "lm_head": 0}
    try:
        # Keep the full model (with its vision part) so the merged model converts to GGUF + mmproj like the original.
        return AutoModelForImageTextToText.from_pretrained(base, **kw)
    except (ValueError, KeyError):
        if low_vram:
            kw["device_map"] = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "lm_head": 0}
        return AutoModelForCausalLM.from_pretrained(base, **kw)


def step_loss(model, ids, learn, dev):
    """Cross-entropy on the learned positions only, computing logits just for those positions."""
    pos = [i for i in range(len(ids) - 1) if learn[i + 1]]
    x = torch.tensor([ids], device=dev)
    out = model(input_ids=x, logits_to_keep=torch.tensor(pos, device=dev), use_cache=False)
    target = torch.tensor([ids[i + 1] for i in pos], device=out.logits.device)
    return torch.nn.functional.cross_entropy(out.logits[0].float(), target), len(pos)


def train(a, log=print):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    data = load_examples(a.data, tok, a.max_len, a.mode, log)
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    model = load_model(a.base, a.qlora and cuda, dtype, a.low_vram and cuda)
    model.config.use_cache = False
    if a.qlora and cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    start = 0
    if a.resume and (out / "adapter").exists():
        model = PeftModel.from_pretrained(model, out / "adapter", is_trainable=True)
        start = json.loads((out / "adapter" / "progress.json").read_text()).get("step", 0) if (out / "adapter" / "progress.json").exists() else 0
        log(f"resuming from step {start}")
    else:
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    total = a.max_steps or math.ceil(len(data) / a.accum) * a.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    if a.qlora and cuda:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    for _ in range(start):
        sched.step()
    dev = torch.device("cuda:0") if cuda else torch.device("cpu")
    model.train()

    def save(step):
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        (out / "adapter" / "progress.json").write_text(json.dumps({"step": step, "total": total}))

    rng = random.Random(a.seed)
    order = []
    for _ in range(start * a.accum):         # same data order as an uninterrupted run
        if not order:
            order = list(range(len(data)))
            rng.shuffle(order)
        order.pop()
    step, t0, seen_tok, run_loss, n_loss = start, time.time(), 0, 0.0, 0
    while step < total:
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            if not order:
                order = list(range(len(data)))
                rng.shuffle(order)
            ids, learn = data[order.pop()]
            loss, _ = step_loss(model, ids, learn, dev)
            (loss / a.accum).backward()
            run_loss += loss.item()
            n_loss += 1
            seen_tok += len(ids)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % a.log_every == 0 or step == total:
            el = time.time() - t0
            mem = f", GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if cuda else ""
            log(f"step {step}/{total} loss {run_loss / max(1, n_loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{seen_tok / el:,.0f} tok/s{mem}, {el / 60:.1f} min, ~{(total - step) * el / (step - start) / 60:.0f} min left")
            run_loss, n_loss = 0.0, 0
        if a.save_every and step % a.save_every == 0:
            save(step)
    save(step)
    log(f"adapter saved to {out / 'adapter'}")
    return out / "adapter"


def merge(base, adapter, dest, log=print):
    """Base + adapter -> merged model, one safetensors shard at a time (needs only a few GB of RAM).

    W' = W + (B @ A) * alpha / r for every adapted Linear. Other tensors (vision tower, mtp, norms) are copied as is,
    and the base's config, tokenizer, chat template and processor files are kept, so llama.cpp converts it exactly
    like the original model (GGUF + mmproj)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base))
    ad = Path(adapter)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(ad / "adapter_model.safetensors"))
    pairs = {}
    for k, v in lora.items():
        m = re.match(r"(?:base_model\.model\.)?(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
        if m:
            pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = v
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    done = 0
    for shard in sorted(src.glob("*.safetensors")):
        t = load_file(str(shard))
        for name in list(t):
            p = pairs.get(name)
            if p and "A" in p and "B" in p:
                w = t[name]
                t[name] = (w.float() + (p["B"].float() @ p["A"].float()) * scale).to(w.dtype)
                done += 1
        save_file(t, str(dest / shard.name), metadata={"format": "pt"})
        del t
    if done != len(pairs):
        log(f"WARNING: merged {done} of {len(pairs)} adapted weights; names may not match the base checkpoint")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, dest / f.name)
    log(f"merged {done} weights; model saved to {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="huihui-ai/Huihui-Qwen3.5-9B-abliterated")
    ap.add_argument("--data", default="finetune/data/train.jsonl.gz")
    ap.add_argument("--out", default="runs/lf-browser")
    ap.add_argument("--mode", choices=["episodes", "steps"], default="episodes")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qlora", action="store_true", help="load the base in 4-bit (less GPU memory)")
    ap.add_argument("--low-vram", action="store_true", help="8 GB cards: embeddings + vision in RAM, 4-bit lm_head")
    ap.add_argument("--resume", action="store_true", help="carry on from OUT/adapter")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    adapter = Path(a.out) / "adapter" if a.merge_only else train(a)
    if not a.no_merge:
        merge(a.base, adapter, Path(a.out) / "merged")


if __name__ == "__main__":
    main()
