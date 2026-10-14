"""Task 5: LoRA persona DPO with TRL DPOTrainer + PEFT (the one permitted TRL use).

python scripts/train_lora_dpo.py --data data/persona_lora/ramsay_style_preferences.json
Extra deps: pip install -q trl peft
"""
import argparse, dataclasses, inspect, os, sys
# DPOTrainer does not allow DataParallel when multiple GPUs are visible in a single process.
# SmolLM2-360M is lightweight and easily fits on a single GPU (cuda:0).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.data import load_persona_pairs

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
p.add_argument("--output_dir", default="outputs/lora_dpo")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--lr", type=float, default=5e-5)   # LoRA tolerates ~50x the full-FT DPO LR
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)

ds = Dataset.from_list(load_persona_pairs(a.data))
lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])

# TRL's config surface shifts between releases; keep only fields this version supports.
wanted = dict(output_dir=a.output_dir, beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
              per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
              max_length=a.max_length, max_prompt_length=a.max_length // 2, fp16=True,
              lr_scheduler_type="cosine", warmup_ratio=0.1, logging_steps=1, save_strategy="no",
              seed=a.seed, report_to="none", remove_unused_columns=False)
fields = {f.name for f in dataclasses.fields(DPOConfig)}
cfg = DPOConfig(**{k: v for k, v in wanted.items() if k in fields})

kw = dict(model=model, args=cfg, train_dataset=ds, peft_config=lora)  # ref = adapter-disabled base
key = "processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"
kw[key] = tok
trainer = DPOTrainer(**kw)
trainer.train()
trainer.model.save_pretrained(os.path.join(a.output_dir, "adapter"))
tok.save_pretrained(os.path.join(a.output_dir, "adapter"))
print("saved adapter to", os.path.join(a.output_dir, "adapter"))
