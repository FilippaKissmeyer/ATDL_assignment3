#!/bin/bash
set -euo pipefail

echo "===== Starting Inference experiments ====="

# Function to get GPU info
get_gpu_info() {
  # Number of GPUs
  num_gpus=$(nvidia-smi -L | wc -l)
  # GPU names (comma-separated)
  gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
  echo "$num_gpus,$gpu_names"
}

# Function to run and time inference
run_and_time() {
  local dataset=$1
  local model=$2
  local memstride=$3

  echo ">>> Running $model model on $dataset with memstride=${memstride}"

  # Record start time
  start=$(date +%s.%N)

  python run_model_script.py --dataset "$dataset" --sam2_model "$model" --sam2_memstride "$memstride"

  # Record end time
  end=$(date +%s.%N)
  runtime=$(echo "$end - $start" | bc)

  # Get GPU info
  IFS=',' read -r num_gpus gpu_names <<< "$(get_gpu_info)"

  echo ">>> Inference completed in ${runtime}s on ${num_gpus} GPU(s): ${gpu_names}"

  # Append timing + GPU info to CSV
  echo "${dataset},${model},${memstride},${num_gpus},\"${gpu_names}\",${runtime}" >> "$OUTFILE"
}

# --- Run for base_plus and large model ---
# for dataset in MOSE DAVIS; do
for dataset in SeCVOS; do

  # Create per-dataset CSV
  OUTFILE="inference_times_${dataset}.csv"
  echo "Dataset,Model,MemStride,NumGPUs,GPU_Names,InferenceTime_s" > "$OUTFILE"
  
  for memstride in {1..10}; do
    model="finetuned${model_num}"
    run_and_time "$dataset" "$model" "$memstride"
  done

done


echo "===== All experiments complete! ====="