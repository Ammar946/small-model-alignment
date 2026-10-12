"""Build blinded pairwise JSONL (DPO vs SFT) with randomized A/B order.

python scripts/build_pairs.py --dpo outputs/gen/dpo_alpaca.json --sft outputs/gen/sft_alpaca.json \
    --out outputs/eval/dpo_vs_sft_pairs.jsonl
The identity mapping lives only under the "meta" key, which the judge never sees.
"""
import argparse, json, os, random

p = argparse.ArgumentParser()
p.add_argument("--dpo", required=True)
p.add_argument("--sft", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

dpo = {r["id"]: r for r in json.load(open(a.dpo))}
sft = {r["id"]: r for r in json.load(open(a.sft))}
ids = [i for i in dpo if i in sft]  # keep DPO file order; identical prompt sets expected
rng = random.Random(a.seed)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    for i in ids:
        flip = rng.random() < 0.5
        first, second = (sft[i], dpo[i]) if flip else (dpo[i], sft[i])
        f.write(json.dumps({
            "id": i, "instruction": dpo[i]["instruction"],
            "response_a": first["output"], "response_b": second["output"],
            "meta": {"a_model": first["generator"], "b_model": second["generator"],
                     "dpo_model": dpo[i]["generator"], "sft_model": sft[i]["generator"]},
        }, ensure_ascii=False) + "\n")
print(f"wrote {len(ids)} pairs -> {a.out} (dropped {len(dpo) - len(ids)} unmatched)")
