"""Dataset loaders and the (single) prompt template used everywhere.

The same ``format_prompt`` is used for SFT, DPO, and generation so train/eval
formats never drift.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
STOP_MARKERS = ("\n### Instruction", "\n### Response")


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


# ------------------------------- SFT data ------------------------------------ #
def _first_turn(messages) -> Tuple[str, str] | None:
    """Single-turn (user, assistant) from a chat message list, else None."""
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    u, a = messages[0]["content"].strip(), messages[1]["content"].strip()
    return (u, a) if u and a else None


def load_sft_data(n_train: int = 30000, n_val: int = 1000, seed: int = 42):
    """UltraChat-200k -> single-turn (instruction, response) lists."""
    from datasets import load_dataset

    def take(split: str, n: int):
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=seed)
        out: List[Dict[str, str]] = []
        for row in ds:
            pair = _first_turn(row["messages"])
            if pair:
                out.append({"prompt": pair[0], "response": pair[1]})
            if len(out) >= n:
                break
        return out

    return take("train_sft", n_train), take("test_sft", n_val)


# ------------------------------- DPO data ------------------------------------ #
def _split_hh(text: str) -> Tuple[str, str] | None:
    """Parse a single-turn HH-RLHF transcript into (prompt, response)."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    head, resp = text.split("\n\nAssistant:")
    prompt = head.split("\n\nHuman:")[1].strip()
    resp = resp.strip()
    return (prompt, resp) if prompt and resp else None


def load_hh_rlhf(n_train: int = 15000, n_val: int = 1000, seed: int = 42):
    from datasets import load_dataset

    def take(split: str, n: int):
        ds = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
        out = []
        for row in ds:
            c, r = _split_hh(row["chosen"]), _split_hh(row["rejected"])
            if c and r and c[0] == r[0] and c[1] != r[1]:
                out.append({"prompt": c[0], "chosen": c[1], "rejected": r[1]})
            if len(out) >= n:
                break
        return out

    train, val = take("train", n_train), take("test", n_val)
    if len(train) < n_train:
        print(f"[warn] only {len(train)} single-turn HH train pairs found")
    return train, val


def load_ultrafeedback(n_train: int = 15000, n_val: int = 1000, seed: int = 42):
    from datasets import load_dataset

    def take(split: str, n: int):
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=seed)
        out = []
        for row in ds:
            ch, rj = row["chosen"], row["rejected"]
            if len(ch) == 2 and len(rj) == 2:
                c, r = ch[-1]["content"].strip(), rj[-1]["content"].strip()
                if c and r and c != r:
                    out.append({"prompt": row["prompt"].strip(), "chosen": c, "rejected": r})
            if len(out) >= n:
                break
        return out

    return take("train_prefs", n_train), take("test_prefs", n_val)


# ------------------------------- eval sets ----------------------------------- #
def load_alpaca_subset(n: int = 300, seed: int = 42) -> List[Dict[str, str]]:
    """Fixed AlpacaEval subset: rng(seed).sample over the original ordering."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path) as f:
        rows = json.load(f)
    idx = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [{"id": f"alpaca_{i}", "instruction": rows[i]["instruction"]} for i in idx]


def load_prompt_file(path: str) -> List[Dict[str, str]]:
    """Load an evaluation prompt JSON (list, or dict wrapping a list)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    out = []
    for i, row in enumerate(data):
        if isinstance(row, str):
            row = {"instruction": row}
        text = row.get("instruction") or row.get("prompt") or row.get("question")
        out.append({"id": str(row.get("id", i)), "instruction": text})
    return out


def load_persona_pairs(path: str) -> List[Dict]:
    """Persona rows -> TRL conversational format (prompt/chosen/rejected lists)."""
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))

    def as_msgs(x, role):
        return x if isinstance(x, list) else [{"role": role, "content": x}]

    return [{"prompt": as_msgs(r["prompt"], "user"),
             "chosen": as_msgs(r["chosen"], "assistant"),
             "rejected": as_msgs(r["rejected"], "assistant")} for r in rows]
