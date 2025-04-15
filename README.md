

# CountingDINO 🧮🦕
## **A Training-free Pipeline for Exemplar-based Class-Agnostic Counting**

This is the official implementation of the paper:  
**“CountingDINO: A Training-free Pipeline for Exemplar-based Class-Agnostic Counting”**

---

## Installation

```bash
conda create --name countingdino python=3.9 -y
conda activate countingdino

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

---

## Dataset Preparation

CountingDINO is evaluated on the [FSC-147 dataset](https://github.com/cvlab-stonybrook/LearningToCountEverything).  
Please follow the instructions in the [official FSC-147 repository](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master) to download and prepare the dataset.

---

##  Evaluating CountingDINO

To evaluate CountingDINO on the FSC-147 with DINOv2 ViT-L/14 (with registers), run:

```bash
python convolutional_counting.py \
  --model_name dinov2_vitl14_reg \
  --divide_et_impera True \
  --divide_et_impera_twice True \ # maximum resolution
  --filter_background True \
  --ellipse_normalization True \
  --ellipse_kernel_cleaning True \
  --split test # either 'test' or 'val'
```

---

## Reproducing CutLER Baseline

The CutLER baseline uses [CutLER](https://github.com/facebookresearch/CutLER) and [Detectron2](https://github.com/facebookresearch/detectron2).  
Install both inside the `third_party/` directory by following [CutLER’s installation guide](https://github.com/facebookresearch/CutLER/blob/main/INSTALL.md).  
We use the model checkpoint `cutler_cascade_final.pth` with the configuration file `cascade_mask_rcnn_R_50_FPN_demo.yaml`.

### Step-by-step

1. **Generate predictions using CutLER**:
```bash
python cutler_dataset_inference.py \
  --output_path path/to/out/preds \
  --model_weights path/to/model/weights \
  --config_file path/to/config \
  --img_dir path/to/FSC147_dataset_images \
  --split_file path/to/split_file \
  --split test  # either 'test' or 'val'
```

2. **Compute DINO-based similarity scores for detected objects**:
```bash
python cutler_exemplar_based_counting.py \
  --annotation_path path/to/FSC147_annotations.json \
  --pred_path path/to/preds \
  --img_dir path/to/FSC147_dataset_images
```

3. **Evaluate predictions against ground-truth density maps**:
```bash
python cutler_evaluation_script.py \
  --pred_path path/to/out/preds \
  --density_map_dir path/to/FSC147_density_maps
```
---