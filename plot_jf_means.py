import os
import sys
import re
import csv
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster environments
import matplotlib.pyplot as plt

# --- Usage check ---
if len(sys.argv) < 2:
    print("Usage: python plot_jfmeans.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]  # e.g., DAVIS, MOSE, SeCVOS
print(f"Generating J&F plot for {dataset}")

# --- Configuration ---
base_path = f"outputs/{dataset}_pred_pngs"
results = {}

# --- Detect available folders dynamically ---
pattern = re.compile(rf"{dataset}_sam2\.1_hiera_(.+)_memstride(\d+)")
subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Extract model/memstride pairs
available = []
for d in subdirs:
    match = pattern.match(d)
    if match:
        model, mem = match.groups()
        available.append((model, int(mem)))

if not available:
    print("No matching result folders found. Check your output directory name pattern.")
    sys.exit(1)

# Group by model
models = sorted(set(model for model, _ in available))
print(f"Detected models: {models}")

# --- Load results ---
for model in models:
    jf_means = []
    mems_for_model = sorted(mem for m, mem in available if m == model)

    for mem in mems_for_model:
        result_dir = os.path.join(base_path, f"{dataset}_sam2.1_hiera_{model}_memstride{mem}")
        csv_path = os.path.join(result_dir, "results_overall.csv")

        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            jf_means.append(None)
            continue

        jf_mean = None
        with open(csv_path, "r", newline="") as f:
            header_line = f.readline().strip()
            headers = [h.strip() for h in header_line.split(",")]
            reader = csv.DictReader(f, fieldnames=headers, skipinitialspace=True)

            for row in reader:
                row = {k.strip(): (v.strip() if v else v) for k, v in row.items()}
                if row.get("sequence") == "Global score":
                    jf_str = row.get("J&F")
                    if jf_str:
                        jf_mean = float(jf_str)
                    break

        if jf_mean is None:
            print(f"No 'Global score' found in {csv_path}")
        jf_means.append(jf_mean)

    results[model] = (mems_for_model, jf_means)

# --- Plot ---
plt.figure(figsize=(8, 5))
for model, (mems, jf_means) in results.items():
    valid_x = [m for m, v in zip(mems, jf_means) if v is not None]
    valid_y = [v for v in jf_means if v is not None]
    plt.plot(valid_x, valid_y, marker='o', label=model)

plt.xlabel("Memory Stride")
plt.ylabel("J&F Mean")
plt.title(f"{dataset} Performance vs Memory Stride")
plt.legend(title="Model")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# --- Save the plot ---
output_path = f"jfmean_vs_memstride_{dataset}.png"
plt.savefig(output_path, dpi=200)
print(f"Plot saved to {output_path}")
