# Running code on UCloud cluster

Open a job with a terminal.

---

## Installing Conda

Follow the guide: [https://docs.cloud.sdu.dk/hands-on/conda-setup.html](https://docs.cloud.sdu.dk/hands-on/conda-setup.html)  

Navigate to `/work` folder and run:

```bash
curl -s -L -o /tmp/miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash /tmp/miniconda_installer.sh -b -f -p /work/miniconda3

sudo ln -s /work/miniconda3/bin/conda /usr/bin/conda

conda init

conda update conda

conda create --name ass3 python=3.10

conda activate ass3

conda install -c conda-forge bc

```
---

## Clone git
Clone our code from github:
```bash
git clone https://github.com/FilippaKissmeyer/ATDL_assignment3.git
```
---

## SAM2 setup
We use code from Sam2: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
with slight modifications.
We install required packages:

```bash
pip3 install torch torchvision

cd ATDL_assignment3

cd sam2

pip install -e .
```

Now we download the pretrained model checkpoints:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

---

## SeCVOS dataset setup

We now download and unzip the SeCVOS Semi-supervised dataset

```bash
cd ..

git clone https://huggingface.co/datasets/OpenIXCLab/SeCVOS

cd SeCVOS

sudo apt install git-lfs
git lfs install
git lfs pull

unzip Annotations.zip
rm Annotations.zip

unzip JPEGImages.zip
rm JPEGImages.zip

cd ..
```

---

## Running Inference

We are now ready to perform Semi-supervised VOS inference.  
The script allows switching between DATASETS (DAVIS, MOSE or SeCVOS), SAM2 models (base_plus vs large)

This script also dynamically runs on multiple GPUs by checking available GPUs and splitting videos evenly between them (for larger datasets, splitting by size may be optimal).

To run inference on the DAVIS dataset using the `base_plus` model:

```bash
python run_model_script.py --dataset SeCVOS --sam2_model base_plus --sam2_memstride 2
```

---

## Evaluation

Evaluation follows: [https://github.com/OpenIXCLab/SeC/blob/main/vos_evaluation/EVALUATE.md](https://github.com/OpenIXCLab/SeC/blob/main/vos_evaluation/EVALUATE.md)  

Download and install their code:

```bash
pip install opencv-python-headless
conda install scikit-image
```

Run evaluation on generated masks:
```bash
python SeCVOS_eval/sav_evaluator.py \
    --gt_root SeCVOS/Annotations \
    --pred_root outputs/SeCVOS_pred_pngs/SeCVOS_sam2.1_hiera_base_plus \
    --strict 
```

Global score: J&F: 57.2 J: 57.0 F: 57.4


---
To plot inference times and J&F-means for DAVIS:

```bash
python plot_jf_means.py DAVIS

python plot_inference_times.py inference_times_DAVIS.csv
```

---

## MOSE

For the MOSE dataset:

Download and unzip `valid.tar.gz` and metadata:

```bash
mkdir MOSE

cd MOSE

pip install gdown

gdown 'https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR'

tar -xvzf valid.tar.gz

rm valid.tar.gz

gdown 'https://drive.google.com/uc?id=1MmhRXKXFnEwzaoeE0b_e_RQjvVLHOc7I'

cd ..
```

Convert MOSE metadata to DAVIS-like structure:

```bash
python convert_MOSE_json_to_txt.py
```

Run inference on MOSE using `base_plus` model with memory size 7:

```bash
python run_model_script.py --dataset MOSE --sam2_model base_plus --sam2_memsize 7
```

For MOSE evaluation, use their evaluation server: [https://codalab.lisn.upsaclay.fr/competitions/10703](https://codalab.lisn.upsaclay.fr/competitions/10703)
