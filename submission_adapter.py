"""Root-level submission adapter: thin wrappers over ``src/alignment/core.py``.

Signatures match the private-evaluation contract exactly.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:  # works even without `pip install -e .`
    sys.path.insert(0, _SRC)

from alignment import core  # noqa: E402


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return core.build_sft_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return core.compute_response_logprobs(model, tokenizer, prompts, responses, max_length)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return core.compute_sft_loss(model, tokenizer, prompts, responses, max_length)


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
):
    chosen, rejected = core.build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length
    )
    return {"chosen": chosen, "rejected": rejected}


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    return core.compute_dpo_loss_from_logps(
        policy_chosen_logps, policy_rejected_logps,
        reference_chosen_logps, reference_rejected_logps, beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    return core.compute_dpo_loss(
        policy_model, reference_model, tokenizer, prompts,
        chosen_responses, rejected_responses, beta, max_length,
    )
