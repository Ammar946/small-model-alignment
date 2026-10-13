"""Tasks 2 & 6: deterministic generation on a fixed prompt set.

Baseline (Task 2):
  python scripts/generate.py --model Qwen/Qwen2.5-0.5B --name qwen2.5-0.5b-base \
      --prompts alpaca --out outputs/gen/base_alpaca.json
SFT / DPO (Task 6): same flags with --model outputs/.../best and the same decoding settings.
Persona LoRA: --format chat --adapter outputs/lora_dpo/adapter --prompts eval_sets/persona_style_prompts.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import STOP_MARKERS, format_prompt, load_alpaca_subset, load_prompt_file
from alignment.utils import write_json_atomic

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
p.add_argument("--name", required=True, help="generator label stored in output")
p.add_argument("--prompts", required=True, help="'alpaca' or a prompt JSON path")
p.add_argument("--dataset_name", default=None)
p.add_argument("--out", required=True)
p.add_argument("--format", choices=["plain", "chat"], default="plain")
p.add_argument("--n", type=int, default=300)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--limit", type=int, default=None, help="sanity check on first K prompts")
p.add_argument("--max_new_tokens", type=int, default=256)
p.add_argument("--repetition_penalty", type=float, default=1.1)
p.add_argument("--no_repeat_ngram_size", type=int, default=0)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--force", action="store_true")
a = p.parse_args()

if a.limit:
    a.out = a.out + ".sanity.json"  # never collide with the real output file
elif os.path.exists(a.out) and not a.force:
    print("exists, skipping:", a.out); sys.exit(0)  # restartable notebooks

rows = load_alpaca_subset(a.n, a.seed) if a.prompts == "alpaca" else load_prompt_file(a.prompts)
if a.limit:
    rows = rows[:a.limit]
dataset = a.dataset_name or ("alpaca_eval" if a.prompts == "alpaca" else os.path.basename(a.prompts))

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"  # required for batched decoder-only generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda:0")
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()


def render(instr: str) -> str:
    if a.format == "chat":
        return tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True)
    return format_prompt(instr)


outputs = []
for i in range(0, len(rows), a.batch_size):
    chunk = rows[i:i + a.batch_size]
    enc = tok([render(r["instruction"]) for r in chunk], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda:0")  # chat template already adds specials
    with torch.no_grad():
        gen = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens,
                             repetition_penalty=a.repetition_penalty,
                             no_repeat_ngram_size=a.no_repeat_ngram_size,
                             pad_token_id=tok.pad_token_id)
    for r, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
        text = tok.decode(g, skip_special_tokens=True)
        for m in STOP_MARKERS:  # base model may run on into a fake next turn
            text = text.split(m)[0]
        outputs.append({"id": r["id"], "instruction": r["instruction"], "output": text.strip(),
                        "generator": a.name, "dataset": dataset})
    print(f"{len(outputs)}/{len(rows)}", flush=True)

write_json_atomic(a.out, outputs)  # written immediately after the model finishes
write_json_atomic(a.out + ".settings.json", {**vars(a), "n_generated": len(outputs)})
if a.limit:
    for o in outputs:
        print("-" * 60, "\n", o["instruction"], "\n>>", o["output"][:500])
