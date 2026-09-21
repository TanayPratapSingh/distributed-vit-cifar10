#!/usr/bin/env bash
# Paste into a Kaggle notebook cell with accelerator set to "GPU T4 x2".
# Kaggle images already ship torch and torchvision built against CUDA, so
# installing this package must NOT pull a different torch in on top of them.
set -euo pipefail

nvidia-smi --query-gpu=index,name,memory.total --format=csv

git clone https://github.com/USER/distributed-vit-cifar10.git || true
cd distributed-vit-cifar10

pip install -q --no-deps -e .          # --no-deps: keep Kaggle CUDA torch
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count(), 'bf16', torch.cuda.is_bf16_supported())"

python -m dvit.sweep kaggle --continue-on-error
python -m dvit.report
