"""Extra correctness tests for core SFT/DPO math (offline, CPU, toy tokenizer + LM).

Run: PYTHONPATH=. python -m pytest -q tests/test_core_extra.py
"""
import math

import torch
import torch.nn as nn

import submission_adapter as sa
from alignment.core import IGNORE_INDEX


class CharTok:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 20) for c in text]


class ToyLM(nn.Module):
    def __init__(self, v=24, d=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.rnn, self.out = nn.Embedding(v, d), nn.GRU(d, d, batch_first=True), nn.Linear(d, v)

    def forward(self, input_ids, attention_mask=None):
        h, _ = self.rnn(self.emb(input_ids))
        return self.out(h)


def test_masking_eos_and_truncation():
    b = sa.build_sft_batch(CharTok(), ["abc", "a"], ["xy", "pqrst"], max_length=6)
    ids, lab = b["input_ids"], b["labels"]
    assert (lab[0, :3] == IGNORE_INDEX).all() and lab[0, 3] == ids[0, 3]
    assert ids[0, 5] == 1 and lab[0, 5] == 1               # EOS appended, scored
    assert ids.shape[1] == 6 and b["attention_mask"][0].sum() == 6
    assert (lab[0][b["attention_mask"][0] == 0] == IGNORE_INDEX).all()
    t = sa.build_sft_batch(CharTok(), ["abc"], ["wxyz"], max_length=5)  # EOS truncated away
    assert 1 not in t["input_ids"][0].tolist()


def test_logps_ignore_prompt_and_order():
    m, tok = ToyLM(), CharTok()
    lp = sa.compute_response_logprobs(m, tok, ["abc", "zzzz"], ["xy", "q"], 16)
    lp_swap = sa.compute_response_logprobs(m, tok, ["zzzz", "abc"], ["q", "xy"], 16)
    assert torch.allclose(lp, lp_swap.flip(0), atol=1e-5)
    # Changing a pad-only difference (batching with a longer neighbour) must not change the score.
    solo = sa.compute_response_logprobs(m, tok, ["abc"], ["xy"], 16)
    assert torch.allclose(solo[0], lp[0], atol=1e-5)


def test_dpo_loss_formula_and_identity():
    pc, pr, rc, rr = (torch.tensor([v]) for v in (-1.0, -3.0, -2.0, -2.5))
    beta = 0.1
    expect = -math.log(1 / (1 + math.exp(-beta * ((pc - pr) - (rc - rr)).item())))
    assert abs(sa.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta).item() - expect) < 1e-6
    m, tok = ToyLM(), CharTok()
    loss = sa.compute_dpo_loss(m, m, tok, ["ab", "cd"], ["xx", "yy"], ["zz", "w"], 0.1, 16)
    assert abs(loss.item() - math.log(2)) < 1e-5            # policy == reference


def test_grads_only_policy():
    pol, ref = ToyLM(seed=0), ToyLM(seed=1)
    loss = sa.compute_dpo_loss(pol, ref, CharTok(), ["ab"], ["xx"], ["zz"], 0.1, 16)
    loss.backward()
    assert all(p.grad is not None for p in pol.parameters())
    assert all(p.grad is None for p in ref.parameters())
    sft = sa.compute_sft_loss(pol, CharTok(), ["ab"], ["xx"], 16)
    assert sft.ndim == 0 and sft.item() > 0
