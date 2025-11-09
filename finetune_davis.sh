#!/bin/bash
set -euo pipefail

echo "===== Starting Fine-Tune Experiments ====="

# Loop over each memstride

for memstride in {1..10}; do
    echo "===== Running memstride${memstride} experiment ====="

    # Build the config filename dynamically
    config_file="configs/sam2.1_training/sam2.1_hiera_b+_DAVIS_finetune_memstride${memstride}.yaml"

    # Run the training script
    python sam2/training/train.py \
        -c "$config_file" \
        --use-cluster 0 \
        --num-gpus $(nvidia-smi -L | wc -l)
done

echo "===== All experiments finished ====="