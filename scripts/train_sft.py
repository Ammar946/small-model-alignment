"""Task 3: SFT.  python scripts/train_sft.py --output_dir outputs/sft"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alignment.sft import run_sft  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--output_dir", default="outputs/sft")
p.add_argument("--train_examples", type=int, default=30000)
p.add_argument("--val_examples", type=int, default=1000)
p.add_argument("--max_length", type=int, default=512)
p.add_argument("--batch_size", type=int, default=2)
p.add_argument("--grad_accum", type=int, default=16)       # effective batch = 32
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup_frac", type=float, default=0.03)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--eval_batch_size", type=int, default=8)
p.add_argument("--eval_every", type=int, default=100)      # optimizer steps
p.add_argument("--log_every", type=int, default=10)
p.add_argument("--resume_every", type=int, default=200, help="0 disables resume state")
p.add_argument("--no_resume", action="store_true")
p.add_argument("--grad_ckpt", action="store_true", help="enable if OOM on T4")
p.add_argument("--seed", type=int, default=42)
run_sft(p.parse_args())
