import os
import subprocess
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["MOSEv2", "SeCVOS"], required=True,
                        help="Dataset to run inference on (MOSEv2 or SeCVOS)")
    
    parser.add_argument(
        '--sam2_model',
        choices=['base_plus', 'large'] + [f'finetuned{i}' for i in range(1, 11)],
        required=True,
    )
    
    parser.add_argument("--extra_flags", nargs="*", default=[],
                        help="Additional flags to pass to vos_inference.py, e.g. --use_all_masks")
    
    # New argument to control memory stride
    parser.add_argument('--sam2_memstride',type=int, default=1,
                        help='Memory temporal stride for evaluation')

    return parser.parse_args()

def get_dataset_paths(dataset):
    if dataset == "MOSEv2":
        base_video_dir = "././MOSEv2/valid/JPEGImages"
        input_mask_dir = "././MOSEv2/valid/Annotations"
        video_list_file = "././MOSEv2/ImageSets/val.txt"

    elif dataset == "SeCVOS":
        base_video_dir = "././SeCVOS/JPEGImages/"
        input_mask_dir = "././SeCVOS/Annotations/"
        video_list_file = "././SeCVOS/ImageSets/val.txt"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return base_video_dir, input_mask_dir, video_list_file

def get_sam_config_and_checkpoint(sam2_model):
    if sam2_model == "base_plus":
        sam2_config = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
        sam2_checkpoint = 'sam2/checkpoints/sam2.1_hiera_base_plus.pt'

    elif sam2_model == "large":
        sam2_config = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        sam2_checkpoint = 'sam2/checkpoints/sam2.1_hiera_large.pt'

    elif sam2_model.startswith("finetuned"):
        model_num = int(sam2_model.replace("finetuned", ""))
        sam2_config = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
        sam2_checkpoint = f'sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_DAVIS_finetune_memstride{model_num}/checkpoints/checkpoint.pt'


    else:
        raise ValueError(f"Unknown SAM2 model: {sam2_model}")
    return sam2_config, sam2_checkpoint


#Helper functions for splitting videos evenly between GPUs based on video size (Generated with chatgpt)
def count_frames(video_dir):
    """Count number of .jpg frames in a given video folder."""
    return len([f for f in os.listdir(video_dir) if f.lower().endswith(".jpg")])

def distribute_videos_evenly(video_names, base_video_dir, num_gpus):
    """Distribute videos across GPUs so each gets a similar total frame count."""
    # Count frames for each video
    video_frame_counts = []
    for v in video_names:
        v_dir = os.path.join(base_video_dir, v)
        if not os.path.isdir(v_dir):
            print(f"[WARNING] Video directory not found: {v_dir}")
            continue
        frame_count = count_frames(v_dir)
        video_frame_counts.append((v, frame_count))

    # Sort videos by frame count (largest first) for greedy balancing
    video_frame_counts.sort(key=lambda x: x[1], reverse=True)

    # Initialize GPU buckets
    gpu_splits = [[] for _ in range(num_gpus)]
    gpu_frame_sums = [0] * num_gpus

    # Greedy assignment: always give next largest video to GPU with least total frames
    for v, fcount in video_frame_counts:
        min_gpu = gpu_frame_sums.index(min(gpu_frame_sums))
        gpu_splits[min_gpu].append(v)
        gpu_frame_sums[min_gpu] += fcount

    print("[INFO] Frame distribution per GPU:")
    for i, total in enumerate(gpu_frame_sums):
        print(f"  GPU {i}: {total} frames, {len(gpu_splits[i])} videos")

    return gpu_splits


def main():
    args = parse_args()
    base_video_dir, input_mask_dir, video_list_file = get_dataset_paths(args.dataset)

    sam2_config, sam2_checkpoint = get_sam_config_and_checkpoint(args.sam2_model)

    # Generate output folder based on dataset and checkpoint filename and memory size.
    checkpoint_name = os.path.splitext(os.path.basename(sam2_checkpoint))[0]
    output_mask_dir = os.path.join("./outputs", f"{args.dataset}_pred_pngs", f"{args.dataset}_{checkpoint_name}_memstride{args.sam2_memstride}")
    os.makedirs(output_mask_dir, exist_ok=True)

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    # Read all videos
    with open(video_list_file, "r") as f:
        video_names = [v.strip() for v in f.readlines()]

    # Inserted by William
    # Sample x% of videos
    if args.dataset == "MOSEv2":
        seed = random.Random(42)
        video_names = seed.sample(video_names, k=len(video_names)//10)

    # Split video names across GPUs based on total frame count
    split_video_lists = distribute_videos_evenly(video_names, base_video_dir, num_gpus)


    # Save temporary split TXT files
    split_txt_files = []
    for gpu_idx, vids in enumerate(split_video_lists):
        split_txt = f"./temp_val_part_gpu{gpu_idx}.txt"
        with open(split_txt, "w") as f:
            for v in vids:
                f.write(v + "\n")
        split_txt_files.append(split_txt)

    # Launch inference processes
    vos_script = "sam2/tools/vos_inference.py"
    processes = []
    for gpu_idx, split_txt in enumerate(split_txt_files):
        cmd = [
            "python", vos_script,
            "--sam2_cfg", sam2_config,
            "--sam2_checkpoint", sam2_checkpoint,
            "--base_video_dir", base_video_dir,
            "--input_mask_dir", input_mask_dir,
            "--video_list_file", split_txt,
            "--output_mask_dir", output_mask_dir,
            "--track_object_appearing_later_in_video", #Needed for SeCVOS (Maybe needs to be removed for MOSE)
            "--sam2_memstride", str(args.sam2_memstride)
        ] + args.extra_flags

        # Only add --per_obj_png_file if dataset is not MOSEv2
        if args.dataset == "SeCVOS":
            cmd.insert(-2, "--per_obj_png_file")  # Insert before the last two flags
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print(f"[INFO] Launching inference on GPU {gpu_idx} with {len(split_video_lists[gpu_idx])} videos...")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.wait()

    print(f"[INFO] All inference completed. Output saved to {output_mask_dir}")

    # --- Cleanup temporary files ---
    for temp_file in split_txt_files:
        try:
            os.remove(temp_file)
            print(f"[INFO] Removed temporary file: {temp_file}")
        except OSError as e:
            print(f"[WARNING] Could not remove temp file {temp_file}: {e}")


if __name__ == "__main__":
    main()
