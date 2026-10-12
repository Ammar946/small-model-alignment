#!/usr/bin/env bash
# Task 1: Kaggle setup. Run in the first cell of every notebook:  !bash scripts/kaggle_setup.sh
# (clone first:  !git clone https://github.com/Ammar946/small-model-alignment && cd small-model-alignment)
set -e
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors "pytest>=8"
python -m pip install -q -e . --no-deps
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'EOF'
import torch, transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__)
n = torch.cuda.device_count()
print("GPUs visible:", n)
assert n >= 2, "Select the 'GPU T4 x2' accelerator (needed for DPO: policy cuda:0, reference cuda:1)"
EOF
PYTHONPATH=. python -m pytest -q tests
