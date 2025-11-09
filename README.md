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

conda install pandas

conda install matplotlib

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

To restructure the dataset to fit with requirements from SAM2:

```bash
convert_SecVOS_json_to_text.py     
```
---
---

## MOSEv2 dataset setup

We now download and unzip the MOSEv2 Semi-supervised dataset

```bash

mkdir MOSEv2

cd MOSEv2

wget https://huggingface.co/datasets/FudanCVL/MOSEv2/resolve/main/valid.tar.gz

wget https://huggingface.co/datasets/FudanCVL/MOSEv2/resolve/main/meta_valid.json

tar -xvzf valid.tar.gz

cd ..
```

To restructure the dataset to fit with requirements from SAM2:

```bash
convert_MOSE_json_to_text.py     
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

## SeCVOS Evaluation

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
## MOSEv2 Evaluation

```bash
cd outputs/MOSEv2_pred_pngs
bash zip_to_MOSEv2_submission.sh
```

The file paths within zip_to_MOSEv2_submission.sh should be either
```bash
cd "MOSEv2_sam2.1_hiera_base_plus_memstride${memstride}"
zip -r "MOSEv2_sam2.1_hiera_base_plus_memstride${memstride}.zip" */
cd ..
```
or
```bash
cd "MOSEv2_finetuned${memstride}_memstride${memstride}"
zip -r "finetuned_sam2.1_memstride${memstride}.zip" */
cd ..
```
Each zip file should be uploaded to [https://www.codabench.org/competitions/10062/#/participate-tab](https://www.codabench.org/competitions/10062/#/participate-tab).
The scores can be downloaded by clicking on a submission then Output from scoring step. This downloads a zip file containing a metrics.json file where the average should be taken on those entries where J&F are non-zero.

---

## Finetune on DAVIS dataset:

We use guide from here:
https://github.com/facebookresearch/sam2/blob/main/training/README.md

install required packages:
```bash
cd sam2
pip install -e ".[dev]"
cd ..
```

Downloading DAVIS dataset:
```bash
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip

unzip DAVIS-2017-trainval-Full-Resolution.zip

rm DAVIS-2017-trainval-Full-Resolution.zip
```

We edit the paths for the DAVIS dataset,
and edit num_frames from 8 to 6 (runs out of GPU memory if not changed) in:
configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml

We can then run our finetuning script with memstride 1,2,6:
```bash
bash finetune_davis.sh
```
