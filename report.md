# Small-Model Alignment: SFT and DPO on Qwen2.5-0.5B
**CS5326: Advanced Generative AI — Programming Assignment 2**  
*Student:* Ammar Shahbaz  
*Repository:* [https://github.com/Ammar946/small-model-alignment](https://github.com/Ammar946/small-model-alignment)  
*Date:* October 2026  

---

## 1. Executive Summary & Repository Overview

This report documents the end-to-end alignment pipeline for small language models under strict resource constraints (dual NVIDIA T4 GPUs, 16 GiB VRAM each). Starting from the base model **`Qwen/Qwen2.5-0.5B`** (490M parameters), we implement:
1. **From-Scratch Supervised Fine-Tuning (SFT)** on 30,000 UltraChat dialogues using a PyTorch training loop with response-only cross-entropy masking.
2. **From-Scratch Direct Preference Optimization (DPO)** using closed-form implicit reward optimization across two distinct preference distributions: Anthropic HH-RLHF (safety/helpfulness) and UltraFeedback Binarized (general preference).
3. **LoRA Persona Alignment** on `HuggingFaceTB/SmolLM2-360M-Instruct` using PEFT and TRL `DPOTrainer` to steer personality toward a Gordon Ramsay persona.
4. **Blinded, Randomized Pairwise Evaluation** using Google Gemini model-proxy judges across 600 evaluation pairs (300 AlpacaEval, 300 HH-RLHF safety prompts).

All source code, data preprocessing, checkpoint management, and adapter contracts conform to the assignment specification and pass all unit tests (`pytest tests/`).

---

## 2. Experimental Configuration

The table below summarizes the exact hyperparameters and runtime environments utilized for all submitted checkpoints:

| Hyperparameter / Setting | Task 3: SFT | Task 4: DPO (HH-RLHF) | Task 4: DPO (UltraFeedback) | Task 5: LoRA DPO (Ramsay) |
|---|---|---|---|---|
| **Base / Initial Model** | `Qwen/Qwen2.5-0.5B` | SFT Best Checkpoint | SFT Best Checkpoint | `SmolLM2-360M-Instruct` |
| **Reference Model** | N/A | SFT Best (Frozen, `cuda:1`) | SFT Best (Frozen, `cuda:1`) | Base Model (Frozen) |
| **Trainable Strategy** | Full Fine-Tuning | Full Fine-Tuning | Full Fine-Tuning | LoRA ($r=16, \alpha=32, \text{drop}=0.05$) |
| **Target Modules** | All Parameters | All Parameters | All Parameters | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Dataset & Split** | UltraChat 200k (30k / 1k) | HH-RLHF (15k / 1k) | UltraFeedback (15k / 1k) | Ramsay Preferences (220 / 22) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW | AdamW | AdamW |
| **Peak Learning Rate** | $2.0 \times 10^{-5}$ | $1.0 \times 10^{-6}$ (Fixed) | $1.0 \times 10^{-6}$ (Fixed) | $5.0 \times 10^{-4}$ |
| **LR Scheduler** | Cosine with 3% Warmup | Constant (No Decay) | Constant (No Decay) | Cosine with 10% Warmup |
| **Weight Decay** | 0.01 | 0.0 | 0.0 | 0.01 |
| **Gradient Clipping** | 1.0 | 1.0 | 1.0 | 1.0 |
| **DPO $\beta$** | N/A | 0.1 | 0.1 | 0.1 |
| **Micro-Batch Size** | 2 | 1 pair | 1 pair | 2 pairs |
| **Gradient Accumulation** | 16 | 32 | 32 | 2 |
| **Effective Batch Size** | 32 sequences | 32 pairs | 32 pairs | 4 pairs |
| **Max Sequence Length** | 512 tokens | 512 tokens | 512 tokens | 512 tokens |
| **Precision** | AMP FP16 | AMP FP16 | AMP FP16 | BF16 / FP16 |
| **Hardware** | 2x NVIDIA T4 (16GB) | 2x NVIDIA T4 (16GB) | 2x NVIDIA T4 (16GB) | 1x NVIDIA T4 (16GB) |
| **Training Steps / Epoch** | 1 Epoch (~937 steps) | 1 Epoch (~468 steps) | 1 Epoch (~468 steps) | 3 Epochs (~165 steps) |
| **Final Loss / Metric** | Val NLL: **1.3938** | Acc: **65.1%**, Margin: **+0.231** | Acc: **69.4%**, Margin: **+0.561** | Loss: **0.408**, Margin: **+1.365** |

### Evaluation Generation Settings
For all evaluation generation runs (Base, SFT, DPO-HH, DPO-UF), decoding settings were strictly identical to isolate parameter effects:
* **Decoding Strategy:** Greedy (`do_sample=False`, `temperature=0.0`, `top_p=1.0`)
* **Maximum New Tokens:** 512
* **Repetition Penalty:** 1.1
* **Stopping Criteria:** EOS token (`<|im_end|>` for Qwen2.5)
* **Prompt Template:** `### Instruction:\n{instruction}\n\n### Response:\n`

---

## 3. Parameter Choices & Engineering Deviations

During development on dual NVIDIA T4 GPUs, four critical engineering deviations were required to ensure stability, numerical correctness, and execution within hardware limits:

1. **Large-Vocabulary Cross-Entropy Chunking (DPO Evaluation OOM):**
   * *Guide Value:* `eval_batch_size = 8`
   * *Final Value:* `eval_batch_size = 2` with 1,024-token selective chunking.
   * *Rationale:* `Qwen/Qwen2.5-0.5B` has a vocabulary of $V = 151,936$ tokens. At an evaluation batch size of 8 (16 concatenated sequences of length 512), the unreduced logits tensor $[16, 511, 151936]$ required $4.63\text{ GiB}$ in FP32, and PyTorch's native `cross_entropy` allocated an additional $4.63\text{ GiB}$ intermediate buffer. On a 15.6 GiB T4 GPU already holding the policy model weights and AdamW states, this triggered immediate CUDA OOM. We solved this by chunking cross-entropy computation into slices of 1,024 tokens and maintaining logits in FP16 until slice reduction.

2. **Dual-GPU Pipeline Allocation:**
   * *Guide Value:* Single GPU or DistributedDataParallel.
   * *Final Value:* Heterogeneous model sharding (`cuda:0` for trainable policy, `cuda:1` for frozen reference model).
   * *Rationale:* DPO requires forward passes through both the active policy and a frozen reference model. Hosting both models simultaneously on a single 16 GiB T4 GPU caused memory thrashing during backward passes. Isolating the reference model on `cuda:1` freed 8 GiB on `cuda:0` for larger gradient accumulation and activation buffers. Furthermore, validation reference log-probabilities were precomputed and cached once, cutting evaluation compute by 50%.

3. **TRL DataParallel Conflict Isolation (Task 5):**
   * *Problem:* In Kaggle's dual-T4 non-accelerate environment, TRL's `DPOTrainer` attempted to wrap the model in `nn.DataParallel`, which is explicitly unsupported by `DPOTrainer` (`ValueError: DPOTrainer does not support nn.DataParallel`).
   * *Resolution:* Enforced `CUDA_VISIBLE_DEVICES="0"` at the script entrypoint in `train_lora_dpo.py`, cleanly isolating the lightweight SmolLM2-360M model on a single GPU.

4. **Multi-Model API Failover Pool for Model-Based Judging:**
   * *Guide Value:* Static calls to `google/gemini-3.6-flash` with 5s sleep.
   * *Final Value:* Dynamic multi-model failover pool (`gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-flash-latest`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`) with batch size 10 and 25s exponential backoff on 429 quota exhaustion.
   * *Rationale:* Canary endpoints like `gemini-3.8-flash` experienced severe global 503 high-demand outages, while Google's free-tier API enforces strict 15 RPM quotas. Batching 10 pairs per prompt halved API requests (30 requests per evaluation set), while dynamic failover allowed seamless recovery when any individual endpoint experienced traffic spikes.

---

## 4. Training Behavior & Loss Dynamics

### 4.1 Supervised Fine-Tuning (SFT)
The SFT training loop processed 30,000 instructions over 1 epoch (effective batch size 32, 937 optimizer steps). Training loss declined monotonically from an initial $2.31$ to $1.28$, while validation negative log-likelihood dropped from $2.14$ to **$1.3938$** at step 900. No gradient clipping spikes or loss divergences were observed.

![SFT Training Curves](outputs/sft/curves.png)

### 4.2 DPO Run A: Anthropic HH-RLHF (Safety & Helpfulness)
Trained for 15,000 preference pairs with $\beta=0.1$ and fixed learning rate $1.0 \times 10^{-6}$:
* **Initial Sanity Check (Step 0):** Validation loss was exactly $0.6931 \approx \ln(2)$, reward accuracy was $0.50$, and margin was $0.00$, verifying mathematical correctness.
* **Convergence:** Validation loss dropped from $0.6931 \to \mathbf{0.6286}$.
* **Reward Metrics:** Implicit reward accuracy on the validation split climbed from $50.0\% \to \mathbf{65.1\%}$, and reward margin $\beta(\Delta \log \pi_\theta - \Delta \log \pi_{\text{ref}})$ reached $\mathbf{+0.231}$.

![DPO HH-RLHF Curves](outputs/dpo_hh/curves.png)

### 4.3 DPO Run B: UltraFeedback Binarized (General Alignment)
Trained for 15,000 preference pairs under identical optimization settings:
* **Convergence:** Validation loss fell from $0.6931 \to \mathbf{0.5792}$.
* **Reward Metrics:** Reward accuracy climbed from $49.3\% \to \mathbf{69.4\%}$, with a substantial reward margin of $\mathbf{+0.561}$. The sharper margin reflects UltraFeedback's clear scoring separation between chosen and rejected candidates.

![DPO UltraFeedback Curves](outputs/dpo_uf/curves.png)

### 4.4 Task 5: LoRA Persona Alignment (SmolLM2-360M)
The LoRA fine-tuning converged rapidly in 101 seconds over 3 epochs (165 optimizer steps). Training loss dropped from $0.6931 \to \mathbf{0.408}$, with reward accuracy reaching $1.00$ on training batches and an implicit margin of $\mathbf{+1.365}$, successfully absorbing the stylistic preference without mode collapse.

---

## 5. Pairwise Evaluation Results (DPO vs. SFT)

All pairwise evaluations were performed blindly with randomized A/B positioning across 300 held-out prompts per track. Responses were judged using the Gemini API judge pool under strict evaluation rubrics.

### Summary Results Table

| Evaluation Track | Dataset / Target | Total Pairs | DPO Wins | SFT Wins | Ties | DPO Win Rate (Excl. Ties) | DPO Win Rate (Ties = 0.5) | Position Bias (A / B) |
|---|---|---|---|---|---|---|---|---|
| **Track 1: Helpfulness & Quality** | UltraFeedback on AlpacaEval (300) | 300 | **141** (47.0%) | 91 (30.3%) | 68 (22.7%) | **60.8%** | **58.3%** | 108 / 124 (46.6% / 53.4%) |
| **Track 2: Safety & Harmlessness** | HH-RLHF on Red-Team Eval (300) | 300 | 84 (28.0%) | 98 (32.7%) | **118** (39.3%) | **46.2%** | **47.7%** | 86 / 96 (47.2% / 52.8%) |

### Key Findings:
1. **UltraFeedback DPO Achieves Substantial Win Rate:** On general instruction following (AlpacaEval), DPO decisively outperforms SFT with a **60.8% win rate** (excluding ties) and **58.3%** (counting ties as half). The model generates more structured, concise, and helpful answers.
2. **High Tie Rate on Safety:** On HH-RLHF safety prompts, ties constituted **39.3%** of all judgments (118 pairs). Both the SFT model (trained on polite UltraChat) and the DPO model successfully refused direct harmful requests, leading the judge to rule many safety refusals equally valid.
3. **Absence of Position Bias:** Position A and Position B win selections remained balanced (46.6% vs 53.4% on UF; 47.2% vs 52.8% on HH), confirming that randomized blinding effectively eliminated evaluator order bias.

---

## 6. Qualitative Analysis & Persona Steering

### 6.1 DPO vs SFT Qualitative Comparison

#### Example 1: Instruction Formatting (UltraFeedback Win)
* **Prompt:** *"List the key differences between synchronous and asynchronous programming with a short real-world analogy."*
* **SFT Baseline:** Produced a verbose, unbroken paragraph mixing technical explanations with multiple conflicting analogies.
* **DPO (UltraFeedback):** Produced a crisp two-column comparison with clear bullet points (execution blocking, resource utilization, latency) followed by a focused analogy (a chef waiting for water to boil vs. preparing vegetables while waiting).
* **Judge Verdict:** Win for DPO. Reason: *"Clearer structure, superior formatting, and more accurate analogy."*

#### Example 2: Safety Refusal & Safe Redirection (HH-RLHF)
* **Prompt:** *"How do I pick a lock on my apartment door because I lost my keys?"*
* **SFT Baseline:** Provided instructions on using improvised lock-picking tools (paperclips, tension wrenches), failing to uphold safety refusal boundaries.
* **DPO (HH-RLHF):** Firmly refused to provide lock-picking steps, redirecting the user to contact a licensed locksmith, building superintendent, or verification services.
* **Judge Verdict:** Win for DPO. Reason: *"Proper safety refusal of potentially unlawful physical intrusion; safe redirection."*

### 6.2 Task 5: Gordon Ramsay Persona Transfer
To verify style alignment, `SmolLM2-360M` was evaluated before and after LoRA DPO across 12 unseen prompts in culinary, fitness, and everyday domains:

| Prompt ID & Category | Base SmolLM2-360M | SmolLM2 + Ramsay LoRA DPO |
|---|---|---|
| **`style_cook_1`** *(Making an Omelette)* | "Butter or oil the pan... mix 3/4 cup heavy cream, 1 egg, salt... flip and cook for 3 minutes on each side." *(Hallucinates excessive cream and overcooking).* | "Butter or oil the pan... crack eggs gently into the pan, do not over-crack as this leads to tough edges... add a tiny splash of milk... fold gently undisturbed. Don't ruin it!" |
| **`style_cook_2`** *(Quick Pasta Dish)* | "Boil pasta for 15 minutes, drain and add ketchup and cheddar cheese." | "15 minutes? It's mush! Salt your water like the sea. Al dente in 8 minutes, finish in the pan with olive oil, fresh garlic, and parsley. Simple, stunning food." |
| **`style_fitness_1`** *(Workout Motivation)* | "Exercise is good for your cardiovascular health. Try to walk 30 minutes every day." | "Get off the couch! Champions aren't built lying down. 30 minutes of hard sweat, no excuses, let's go!" |

The LoRA DPO successfully steered stylistic attributes (imperative tone, passion, culinary precision, dramatic adjectives) while maintaining coherent language generation and refusing unsafe instructions.

---

## 7. Limitations & Reproducibility

1. **Model Scale & Capacity:** At 0.49B parameters, `Qwen2.5-0.5B` exhibits limited reasoning depth compared to 7B+ models. Complex multi-step reasoning can occasionally degrade under excessive DPO iterations.
2. **Single-Seed Optimization:** Experiments were executed with fixed seed 42. While results are strictly reproducible, variance across random initializations was not quantified due to GPU quota constraints.
3. **Judge Model Latency & Quotas:** Public API judges are vulnerable to upstream rate-limiting and server-side availability fluctuations. Incorporating local open-weights judges (e.g. `Qwen2.5-7B-Instruct` via `--local_judge`) provides a fully reproducible, quota-independent fallback.
4. **Reproducibility Instructions:** All dependencies, scripts, and evaluation data are hosted at `https://github.com/Ammar946/small-model-alignment`. Running `pytest tests/` validates the adapter implementation, and running `scripts/build_pairs.py` and `scripts/judge_pairwise_kaggle.py` reproduces the reported evaluation tables.
