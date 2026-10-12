"""From-scratch DPO training loop.

Policy: fp32 trainable on cuda:0 (fp16 autocast + GradScaler).
Reference: frozen fp16 copy on cuda:1 (no gradients, ever).
Each micro-step is one preference pair; gradients accumulate to the effective batch.
"""
from __future__ import annotations

import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import (_pad_id, build_dpo_batch, compute_dpo_loss_from_logps, dpo_reward_stats,
                   paired_response_logps)
from .data import format_prompt, load_hh_rlhf, load_ultrafeedback
from .utils import (append_jsonl, build_order, load_resume_state, param_groups,
                    save_hf_checkpoint, save_resume_state, set_seed, write_json_atomic)

LOADERS = {"hh": load_hh_rlhf, "ultrafeedback": load_ultrafeedback}


def _batch_for(tok, rows, args):
    return build_dpo_batch(tok, [format_prompt(r["prompt"]) for r in rows],
                           [r["chosen"] for r in rows], [r["rejected"] for r in rows],
                           args.max_length, args.max_prompt_length)


@torch.no_grad()
def reference_logps(ref, tok, rows, args):
    """Reference log-probs for rows; frozen model => safe to cache for validation."""
    c, r = _batch_for(tok, rows, args)
    ref_c, ref_r = paired_response_logps(ref, c, r, _pad_id(tok))  # fp16 forward on cuda:1
    return ref_c.float().cpu(), ref_r.float().cpu()


@torch.no_grad()
def evaluate_dpo(policy, ref, tok, val, args, cache: dict):
    """Validation DPO loss, reward accuracy (strict r_w > r_l), and mean margin."""
    policy.eval()
    losses, accs, margins = [], [], []
    for i in range(0, len(val), args.eval_batch_size):
        rows = val[i:i + args.eval_batch_size]
        if i not in cache:
            cache[i] = reference_logps(ref, tok, rows, args)
        ref_c, ref_r = (t.to("cuda:0") for t in cache[i])
        c, r = _batch_for(tok, rows, args)
        with torch.autocast("cuda", dtype=torch.float16):
            pol_c, pol_r = paired_response_logps(policy, c, r, _pad_id(tok))
        losses.append(-torch.nn.functional.logsigmoid(args.beta * ((pol_c - pol_r) - (ref_c - ref_r))))
        s = dpo_reward_stats(pol_c, pol_r, ref_c, ref_r, args.beta)
        accs.append(s["accuracy"]); margins.append(s["margin"])
    policy.train()
    cat = lambda xs: torch.cat(xs).mean().item()
    return {"val_loss": cat(losses), "reward_accuracy": cat(accs), "reward_margin": cat(margins)}


def run_dpo(args) -> None:
    assert torch.cuda.device_count() >= 2, "Need 2 GPUs (policy cuda:0, reference cuda:1)"
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    resume_path = os.path.join(args.output_dir, "resume.pt")
    write_json_atomic(os.path.join(args.output_dir, "config.json"), vars(args))

    tok = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    train, val = LOADERS[args.dataset](args.train_pairs, args.val_pairs, args.seed)
    print(f"DPO[{args.dataset}] data: {len(train)} train / {len(val)} val")

    policy = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float32).to("cuda:0")
    ref = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.float16).to("cuda:1")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable(); policy.config.use_cache = False
    policy.train()

    if args.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(param_groups(policy, args.weight_decay), lr=args.lr, fused=True)
    scaler = torch.cuda.amp.GradScaler()

    order = build_order(len(train), args.epochs, args.seed)
    micro_bs, accum = args.batch_size, args.grad_accum
    total_steps = math.ceil(len(order) / (micro_bs * accum))
    pad_id = _pad_id(tok)

    step, best = 0, {"reward_accuracy": -1.0, "reward_margin": -1e9}
    state = None if args.no_resume else load_resume_state(resume_path)
    if state is not None:
        policy.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"]); step, best = state["step"], state["best"]
        print(f"Resumed from step {step}"); del state

    cache: dict = {}
    def validate(tag_step: int):
        nonlocal best
        m = evaluate_dpo(policy, ref, tok, val, args, cache)
        append_jsonl(metrics_path, {"type": "eval", "step": tag_step, **m}); print("[eval]", tag_step, m)
        better = (m["reward_accuracy"], m["reward_margin"]) > (best["reward_accuracy"], best["reward_margin"])
        if tag_step > 0 and better:  # step 0 is the tied (acc=0) baseline; never "best"
            best = {"reward_accuracy": m["reward_accuracy"], "reward_margin": m["reward_margin"]}
            save_hf_checkpoint(policy, tok, os.path.join(args.output_dir, "best"))

    if step == 0:
        validate(0)  # sanity: policy == reference => loss=log2, acc=0, margin=0

    t0, acc_buf = time.time(), {"loss": [], "acc": [], "margin": []}
    while step < total_steps:
        for m_i in range(accum):
            lo = (step * accum + m_i) * micro_bs
            idx = order[lo:lo + micro_bs]
            if not idx:
                break
            rows = [train[i] for i in idx]
            c, r = _batch_for(tok, rows, args)
            ref_c, ref_r = reference_logps(ref, tok, rows, args)          # no_grad, cuda:1
            ref_c, ref_r = ref_c.to("cuda:0"), ref_r.to("cuda:0")
            with torch.autocast("cuda", dtype=torch.float16):
                pol_c, pol_r = paired_response_logps(policy, c, r, pad_id)  # chosen+rejected, 1 pass
            loss = compute_dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, args.beta)
            scaler.scale(loss / accum).backward()
            s = dpo_reward_stats(pol_c, pol_r, ref_c, ref_r, args.beta)
            acc_buf["loss"].append(loss.item()); acc_buf["acc"].append(s["accuracy"].mean().item())
            acc_buf["margin"].append(s["margin"].mean().item())

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            mean = lambda k: sum(acc_buf[k]) / max(1, len(acc_buf[k]))
            rec = {"type": "train", "step": step, "loss": mean("loss"), "reward_accuracy": mean("acc"),
                   "reward_margin": mean("margin"), "grad_norm": float(gnorm),
                   "elapsed_s": time.time() - t0}
            append_jsonl(metrics_path, rec); print(rec)
            acc_buf = {"loss": [], "acc": [], "margin": []}

        if step % args.eval_every == 0 or step == total_steps:
            validate(step)
        if args.resume_every and step % args.resume_every == 0 and step < total_steps:
            save_resume_state(resume_path, {"model": policy.state_dict(), "opt": opt.state_dict(),
                                            "scaler": scaler.state_dict(), "step": step, "best": best})

    save_hf_checkpoint(policy, tok, os.path.join(args.output_dir, "final"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("Done. best:", best)
