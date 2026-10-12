"""Plot SFT/DPO training curves from metrics.jsonl.  python scripts/plot_metrics.py outputs/sft outputs/dpo_hh ..."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alignment.utils import read_jsonl  # noqa: E402

for run in sys.argv[1:]:
    recs = read_jsonl(os.path.join(run, "metrics.jsonl"))
    tr = [r for r in recs if r["type"] == "train"]; ev = [r for r in recs if r["type"] == "eval"]
    keys = ["val_loss"] + (["reward_accuracy", "reward_margin"] if "reward_accuracy" in (ev[0] if ev else {}) else [])
    fig, axs = plt.subplots(1, 1 + len(keys), figsize=(5 * (1 + len(keys)), 3.6))
    axs[0].plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train loss (per log window)")
    axs[0].set_xlabel("optimizer step"); axs[0].set_ylabel("loss"); axs[0].legend()
    for ax, k in zip(axs[1:], keys):
        ax.plot([r["step"] for r in ev], [r[k] for r in ev], "o-", label=f"validation {k}")
        ax.set_xlabel("optimizer step"); ax.set_ylabel(k); ax.legend()
    fig.suptitle(os.path.basename(run.rstrip("/"))); fig.tight_layout()
    fig.savefig(os.path.join(run, "curves.png"), dpi=150); print("saved", run)
