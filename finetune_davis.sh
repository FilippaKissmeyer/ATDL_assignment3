#!/bin/bash
set -euo pipefail

echo "===== Starting Fine-Tune Experiments ====="


# Array of memory strides
memstrides=(2 6)

# Loop over each memstride
for stride in "${memstrides[@]}"; do
    echo "===== Running memstride${stride} experiment ====="

    # Build the config filename dynamically
    config_file="configs/sam2.1_training/sam2.1_hiera_b+_DAVIS_finetune_memstride${stride}.yaml"

    # Run the training script
    python sam2/training/train.py \
        -c "$config_file" \
        --use-cluster 0 \
        --num-gpus $(nvidia-smi -L | wc -l)
done

echo "===== All experiments finished ====="