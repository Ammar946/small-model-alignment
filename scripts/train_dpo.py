"""Task 4: DPO.  python scripts/train_dpo.py --dataset hh --sft_checkpoint outputs/sft/best --output_dir outputs/dpo_hh"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alignment.dpo import run_dpo  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
p.add_argument("--sft_checkpoint", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--train_pairs", type=int, default=15000)
p.add_argument("--val_pairs", type=int, default=1000)
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--max_prompt_length", type=int, default=256)
p.add_argument("--batch_size", type=int, default=1)         # pairs per micro-step
p.add_argument("--grad_accum", type=int, default=32)        # effective batch = 32
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lr", type=float, default=1e-6)            # fixed, no decay
p.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
p.add_argument("--weight_decay", type=float, default=0.0)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--eval_batch_size", type=int, default=2)
p.add_argument("--eval_every", type=int, default=50)        # optimizer steps
p.add_argument("--log_every", type=int, default=5)
p.add_argument("--resume_every", type=int, default=100, help="0 disables resume state")
p.add_argument("--no_resume", action="store_true")
p.add_argument("--grad_ckpt", action="store_true")
p.add_argument("--seed", type=int, default=42)
run_dpo(p.parse_args())
