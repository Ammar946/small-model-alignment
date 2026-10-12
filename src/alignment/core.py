"""Core SFT / DPO math: batching, response-only log-probs, and losses.

Everything here is framework-light (plain PyTorch) and independent of any
training loop, so it can be unit-tested with toy tokenizers/models and reused
by the SFT loop, the DPO loop, and the private submission adapter.

Conventions
-----------
* ``prompts`` are *already formatted* strings (see ``alignment.data``). The
  response is tokenized separately and appended, so the prompt/response
  boundary is exact and the prompt mask is unambiguous.
* Labels use ``-100`` for prompt tokens and padding tokens. Only response
  tokens (plus the appended EOS, if it survives truncation) are ever scored.
* Sequences are right-padded to the longest sequence in the batch.
* Row order of the inputs is preserved in every returned tensor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
Batch = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------- #
# Tokenization / batching
# --------------------------------------------------------------------------- #
def _encode(tokenizer, text: str) -> List[int]:
    """Encode text without special tokens (robust to minimal toy tokenizers)."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(i) for i in ids]


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return int(pad) if pad is not None else 0


def _build_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Return (input_ids, labels) for one example, truncated to max_length.

    EOS is appended to the response *before* truncation, so a response cut at
    ``max_length`` loses its EOS (as required by the assignment contract).
    """
    p_ids = _encode(tokenizer, prompt)
    if max_prompt_length is not None and len(p_ids) > max_prompt_length:
        p_ids = p_ids[-max_prompt_length:]  # keep the end of the prompt
    r_ids = _encode(tokenizer, response)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        r_ids = r_ids + [int(eos)]

    input_ids = (p_ids + r_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:max_length]
    return input_ids, labels


def _collate(examples: Sequence[Tuple[List[int], List[int]]], pad_id: int) -> Batch:
    """Right-pad (ids, labels) pairs into LongTensors with an attention mask."""
    width = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, (ids, lab) in enumerate(examples):
        n = len(ids)
        input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(lab, dtype=torch.long)
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    max_prompt_length: Optional[int] = None,
) -> Batch:
    """Tokenize prompt+response pairs into an SFT batch (response-only labels)."""
    assert len(prompts) == len(responses), "prompts/responses length mismatch"
    pad_id = _pad_id(tokenizer)
    examples = [
        _build_example(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, responses)
    ]
    return _collate(examples, pad_id)


def build_dpo_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    max_prompt_length: Optional[int] = None,
) -> Tuple[Batch, Batch]:
    """Return (chosen_batch, rejected_batch); rows are aligned with ``prompts``."""
    chosen = build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length)
    rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length)
    return chosen, rejected


def concat_batches(a: Batch, b: Batch, pad_id: int) -> Batch:
    """Stack two batches along dim 0, right-padding to a common width."""
    width = max(a["input_ids"].shape[1], b["input_ids"].shape[1])

    def pad(t: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(t, (0, width - t.shape[1]), value=value)

    return {
        "input_ids": torch.cat([pad(a["input_ids"], pad_id), pad(b["input_ids"], pad_id)]),
        "attention_mask": torch.cat([pad(a["attention_mask"], 0), pad(b["attention_mask"], 0)]),
        "labels": torch.cat([pad(a["labels"], IGNORE_INDEX), pad(b["labels"], IGNORE_INDEX)]),
    }


# --------------------------------------------------------------------------- #
# Log-probabilities
# --------------------------------------------------------------------------- #
def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _logits(model, batch: Batch) -> torch.Tensor:
    try:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    except TypeError:  # minimal toy models may only accept input_ids
        out = model(batch["input_ids"])
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def batch_response_logps(model, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-example (sum of response token log-probs, number of response tokens).

    Next-token alignment: the logit at position t predicts the label at t+1,
    so logits are shifted left and labels shifted right. Positions whose label
    is ``-100`` (prompt / padding) contribute exactly zero. Gradients flow
    through the logits unless the caller wraps this in ``torch.no_grad()``.
    """
    batch = _to_device(batch, _model_device(model))
    logits = _logits(model, batch)[:, :-1, :].float()  # float32 for a stable log-softmax
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)  # any valid index; masked out below
    vocab = logits.shape[-1]
    token_logps = -F.cross_entropy(
        logits.reshape(-1, vocab), safe_labels.reshape(-1), reduction="none"
    ).view_as(labels)
    token_logps = token_logps * mask  # zero prompt/pad positions
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def paired_response_logps(
    model, chosen: Batch, rejected: Batch, pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score chosen and rejected in ONE forward pass (concat along batch)."""
    n = chosen["input_ids"].shape[0]
    logps, _ = batch_response_logps(model, concat_batches(chosen, rejected, pad_id))
    return logps[:n], logps[n:]


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    """Sum of response-token log-probs per example, shape ``[B]`` (differentiable)."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, _ = batch_response_logps(model, batch)
    return logps


# --------------------------------------------------------------------------- #
# SFT loss
# --------------------------------------------------------------------------- #
def sft_loss_from_batch(model, batch: Batch) -> torch.Tensor:
    """Mean NLL over all unmasked response tokens in the batch (token-mean)."""
    logps, counts = batch_response_logps(model, batch)
    return -logps.sum() / counts.sum().clamp(min=1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length) -> torch.Tensor:
    """Scalar response-only SFT loss. Reduction: mean over response tokens."""
    return sft_loss_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


# --------------------------------------------------------------------------- #
# DPO loss
# --------------------------------------------------------------------------- #
def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
) -> torch.Tensor:
    """Batch-mean DPO loss: -log sigmoid(beta * (delta_theta - delta_ref))."""
    delta_theta = policy_chosen_logps - policy_rejected_logps
    delta_ref = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (delta_theta - delta_ref)).mean()


def dpo_reward_stats(
    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
) -> Dict[str, torch.Tensor]:
    """Implicit rewards r = beta * (log pi_theta - log pi_ref); detached diagnostics."""
    r_w = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    r_l = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return {"reward_chosen": r_w, "reward_rejected": r_l,
            "accuracy": (r_w > r_l).float(), "margin": r_w - r_l}


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
) -> torch.Tensor:
    """End-to-end DPO loss. Reference forward is under ``no_grad``; the
    reference may live on a different device than the policy."""
    chosen, rejected = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )
    pad_id = _pad_id(tokenizer)
    pol_c, pol_r = paired_response_logps(policy_model, chosen, rejected, pad_id)
    with torch.no_grad():
        ref_c, ref_r = paired_response_logps(reference_model, chosen, rejected, pad_id)
    dev = pol_c.device
    return compute_dpo_loss_from_logps(pol_c, pol_r, ref_c.to(dev), ref_r.to(dev), beta)
