#!/bin/bash
set -euo pipefail

echo "===== Starting Fine-Tune Experiments ====="


python sam2/training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_DAVIS_finetune.yaml \
    --use-cluster 0 \
    --num-gpus $(nvidia-smi -L | wc -l)