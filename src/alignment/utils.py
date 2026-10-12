"""Shared helpers: seeding, JSONL logging, LR schedule, durable checkpointing."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------- JSON / JSONL ---------------------------------- #
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append + flush + fsync so metrics survive a notebook disconnect."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ------------------------------ LR schedule --------------------------------- #
def cosine_with_warmup(step: int, total_steps: int, warmup_frac: float = 0.03) -> float:
    """LR multiplier: linear warmup then cosine decay to 0."""
    warmup = max(1, int(round(total_steps * warmup_frac)))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ------------------------------ checkpointing ------------------------------- #
def save_hf_checkpoint(model, tokenizer, out_dir: str) -> None:
    """Write model+tokenizer to a temp dir then swap in, so a crash mid-save
    never leaves a half-written 'best' checkpoint."""
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def save_resume_state(path: str, state: Dict[str, Any]) -> None:
    """Single overwritten resume file (model+optimizer+scaler+counters)."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(path: str):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def build_order(n: int, epochs: int, seed: int) -> List[int]:
    """Deterministic example order across epochs (enables exact resume)."""
    order: List[int] = []
    for ep in range(epochs):
        perm = list(range(n))
        random.Random(seed + ep).shuffle(perm)
        order.extend(perm)
    return order


def param_groups(model, weight_decay: float):
    """No weight decay on biases / norm weights (1-D tensors)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
