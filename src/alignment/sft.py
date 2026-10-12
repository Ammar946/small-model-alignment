"""From-scratch SFT training loop (PyTorch + Transformers, no Trainer).

fp32 master weights + fp16 autocast + GradScaler, gradient accumulation,
cosine LR with warmup, gradient clipping, best/final checkpoints, and an
exact-resume state file so a Kaggle disconnect costs at most `resume_every`
optimizer steps.
"""
from __future__ import annotations

import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import build_sft_batch, sft_loss_from_batch, batch_response_logps
from .data import format_prompt, load_sft_data
from .utils import (append_jsonl, build_order, cosine_with_warmup, load_resume_state,
                    param_groups, save_hf_checkpoint, save_resume_state, set_seed,
                    write_json_atomic)


@torch.no_grad()
def evaluate_sft(model, tok, val, args, device) -> float:
    """Token-mean validation NLL over all response tokens."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(val), args.eval_batch_size):
        chunk = val[i:i + args.eval_batch_size]
        batch = build_sft_batch(tok, [format_prompt(r["prompt"]) for r in chunk],
                                [r["response"] for r in chunk], args.max_length)
        with torch.autocast("cuda", dtype=torch.float16):
            logps, counts = batch_response_logps(model, batch)
        total_nll += -logps.sum().item()
        total_tok += counts.sum().item()
    model.train()
    return total_nll / max(1, total_tok)


def run_sft(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    resume_path = os.path.join(args.output_dir, "resume.pt")
    write_json_atomic(os.path.join(args.output_dir, "config.json"), vars(args))

    tok = AutoTokenizer.from_pretrained(args.model)
    train, val = load_sft_data(args.train_examples, args.val_examples, args.seed)
    print(f"SFT data: {len(train)} train / {len(val)} val")

    # Trainable weights stay fp32 (required for GradScaler); compute is fp16 autocast.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95), fused=True)
    scaler = torch.cuda.amp.GradScaler()

    order = build_order(len(train), args.epochs, args.seed)
    micro_bs, accum = args.batch_size, args.grad_accum
    total_steps = math.ceil(len(order) / (micro_bs * accum))

    step, best_val = 0, float("inf")
    state = None if args.no_resume else load_resume_state(resume_path)
    if state is not None:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step, best_val = state["step"], state["best_val"]
        print(f"Resumed from step {step}")
        del state

    t0, run_loss, run_n = time.time(), 0.0, 0
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = args.lr * cosine_with_warmup(step, total_steps, args.warmup_frac)

        for m in range(accum):
            lo = (step * accum + m) * micro_bs
            idx = order[lo:lo + micro_bs]
            if not idx:
                break
            rows = [train[i] for i in idx]
            batch = build_sft_batch(tok, [format_prompt(r["prompt"]) for r in rows],
                                    [r["response"] for r in rows], args.max_length)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = sft_loss_from_batch(model, batch)
            scaler.scale(loss / accum).backward()  # average over accumulation window
            run_loss += loss.item(); run_n += 1

        scaler.unscale_(opt)  # clip true (unscaled) gradients
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            rec = {"type": "train", "step": step, "loss": run_loss / max(1, run_n),
                   "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
                   "scale": scaler.get_scale(), "elapsed_s": time.time() - t0}
            append_jsonl(metrics_path, rec); print(rec)
            run_loss, run_n = 0.0, 0

        if step % args.eval_every == 0 or step == total_steps:
            vl = evaluate_sft(model, tok, val, args, device)
            append_jsonl(metrics_path, {"type": "eval", "step": step, "val_loss": vl})
            print(f"[eval] step {step} val_loss {vl:.4f}")
            if vl < best_val:
                best_val = vl
                save_hf_checkpoint(model, tok, os.path.join(args.output_dir, "best"))

        if args.resume_every and step % args.resume_every == 0 and step < total_steps:
            save_resume_state(resume_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                            "scaler": scaler.state_dict(), "step": step,
                                            "best_val": best_val})

    save_hf_checkpoint(model, tok, os.path.join(args.output_dir, "final"))
    if os.path.exists(resume_path):
        os.remove(resume_path)  # free ~6 GB of Kaggle disk
    print(f"Done. best val_loss={best_val:.4f}")
