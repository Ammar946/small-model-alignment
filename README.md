# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B (Kaggle T4 x2)
**Author:** Muhammad Ammar | **Roll Number:** 25280019  

From-scratch SFT and DPO (PyTorch + Transformers only), a TRL/LoRA persona run, and a resumable LLM-judge evaluation.

## Layout
| Path | Purpose |
|---|---|
| `submission_adapter.py` | Private-grader contract; thin wrappers over `src/alignment/core.py` |
| `src/alignment/core.py` | Batching, response-only log-probs, SFT/DPO losses |
| `src/alignment/{sft,dpo}.py` | Training loops (AMP fp16, accumulation, clipping, resume) |
| `src/alignment/data.py` | Prompt template + dataset loaders |
| `scripts/` | `train_sft`, `train_dpo`, `train_lora_dpo`, `generate`, `build_pairs`, `judge_pairwise_kaggle`, `plot_metrics` |
| `tests/test_core_extra.py` | Offline tests (masking, EOS/truncation, ordering, DPO math, grads) |

Prompt template (all stages): `### Instruction:\n{instruction}\n\n### Response:\n`. SFT loss = mean NLL over response tokens (token-mean).

## Kaggle runbook (cells, in order)
```bash
# Task 1 – setup (do NOT reinstall torch)
git clone https://github.com/Ammar946/small-model-alignment && cd small-model-alignment
bash scripts/kaggle_setup.sh   # installs deps, prints versions, checks 2 GPUs, runs tests

# Task 2 – baseline (300 prompts, seed 42, greedy)
python scripts/generate.py --model Qwen/Qwen2.5-0.5B --name qwen2.5-0.5b-base --prompts alpaca --out outputs/gen/base_alpaca.json --limit 3   # sanity
python scripts/generate.py --model Qwen/Qwen2.5-0.5B --name qwen2.5-0.5b-base --prompts alpaca --out outputs/gen/base_alpaca.json

# Task 3 – SFT (notebook 1; Save Version afterwards). Add --grad_ckpt if OOM.
python scripts/train_sft.py --output_dir /kaggle/working/outputs/sft

# Task 4 – DPO (notebook 2 and 3, one run each; attach SFT outputs as input dataset)
python scripts/train_dpo.py --dataset hh            --sft_checkpoint <SFT>/best --output_dir /kaggle/working/outputs/dpo_hh
python scripts/train_dpo.py --dataset ultrafeedback --sft_checkpoint <SFT>/best --output_dir /kaggle/working/outputs/dpo_uf

# Task 5 – LoRA persona (pip install -q trl peft first)
python scripts/train_lora_dpo.py

# Task 6 – generation (identical decoding for SFT and DPO)
python scripts/generate.py --model <SFT>/best --name sft --prompts alpaca --out outputs/gen/sft_alpaca.json
python scripts/generate.py --model <DPO_UF>/best --name dpo-uf --prompts alpaca --out outputs/gen/dpo_uf_alpaca.json
python scripts/generate.py --model <SFT>/best --name sft --prompts eval_sets/hh_rlhf_eval_300_seed42.json --dataset_name hh_eval --out outputs/gen/sft_hh.json
python scripts/generate.py --model <DPO_HH>/best --name dpo-hh --prompts eval_sets/hh_rlhf_eval_300_seed42.json --dataset_name hh_eval --out outputs/gen/dpo_hh_hh.json
python scripts/build_pairs.py --dpo outputs/gen/dpo_uf_alpaca.json --sft outputs/gen/sft_alpaca.json --out outputs/eval/uf_pairs.jsonl
python scripts/build_pairs.py --dpo outputs/gen/dpo_hh_hh.json   --sft outputs/gen/sft_hh.json     --out outputs/eval/hh_pairs.jsonl

# Task 7 – judge (utility-benchmark notebook; rerun freely, it resumes)
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/uf_pairs.jsonl --output_path outputs/eval/uf_judged.jsonl \
  --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
python scripts/judge_pairwise_kaggle.py --pairs_path outputs/eval/hh_pairs.jsonl --output_path outputs/eval/hh_judged.jsonl \
  --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments --safety
```

## Durability
- `metrics.jsonl` is fsync'd on every record; `best/` and `final/` are written via temp-dir swap.
- `resume.pt` (model+optimizer+scaler+step) is overwritten every `--resume_every` steps; rerunning the same command resumes exactly (deterministic data order). It is deleted on completion to save disk.
- `generate.py` skips finished outputs; the judge appends per batch and skips judged ids.

## Notes / assumptions to verify on Kaggle
- `kaggle_benchmarks` call style (`kbench.llms[name].prompt(text)`) is based on common usage; adjust `make_caller` if your environment differs. `--dry_run` exercises the full judge plumbing offline.
- HH-RLHF eval JSON field names are auto-detected (`instruction`/`prompt`/`question`).
- DPO validation reference log-probs are cached once (reference is frozen).
- DPO step 0 logs loss≈ln2, accuracy=0, margin=0 as a sanity check.
