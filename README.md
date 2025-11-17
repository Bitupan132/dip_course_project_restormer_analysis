# Quantifying the Impact of Transformer-based Image Restoration On Downstream Vision Tasks

**Author:** Bitupan Arandhara

## 1. Project Overview

This project quantitatively analyzes the impact of image restoration on the performance of object detection models.

State-of-the-art object detectors like YOLOv8 perform exceptionally well on clean, high-quality benchmark datasets. However, real-world images are often degraded by sensor noise, motion blur, and other artifacts, which significantly reduces detection accuracy.

This project investigates:

- How much performance (mAP) is lost when a YOLOv8s model is run on images with synthetic Gaussian noise and motion blur.
- How much of that lost performance can be recovered by first "cleaning" the images with the Restormer (a Transformer-based restoration model).

## 2. Core Technologies

- **Restoration Model:** [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://github.com/swz30/Restormer)
- **Detection Model:** [YOLOv8s (from Ultralytics)](https://github.com/ultralytics/ultralytics)
- **Dataset:** MS-COCO 2017 Validation Set (5,000 images)

## 3. Directory Structure

This project assumes the following folder structure:

```
PROJECT_ROOT/
├── data/
│   ├── val2017/                # (Input) Original 5,000 COCO JPG images
│   ├── annotations/            # (Input) instances_val2017.json
│   │
│   ├── original_images/        # (Step 2) Original images converted to PNG
│   │
│   ├── degraded_images/        # Contains degraded dataset for different type and levels of degradation
│   │   ├── gaussian_noise_high/
│   │   └── motion_blur_high_hor/
│   │
│   ├── restored_images/        # Contains degraded dataset for different type and levels of degradation
│   │   ├── gaussian_noise_high/
│   │   └── motion_blur_high_hor/
│   │
│   ├──  detection_output/      # (Output) mAP results and prediction images
|   |
│   └── dataset_yaml            # Contains yaml files for each dataset
│
└── src/
|   ├── Denoising/               # (Input) Restormer .pth model weights for Denoising
│       ├── pretrained_models/
|   ├── Motion_Deblurring/       # (Input) Restormer .pth model weights for Motion Deblurring
│       ├── pretrained_models/
|   ├── restormer/              # Directory containing the original code of the restormer model
    ├── configs.py              # Main config file
    ├── convert_to_png.py       # (Step 2) Converts original JPGs to PNGs
    ├── convert_annotations_to_txt.py  # (Step 3) Converts COCO JSON to YOLO .txt
    ├── create_degraded_dataset.py      # (Step 4) Generates all 50,000 degraded images
    ├── restore.py              # (Step 5) Generates all 50,000 restored images
    ├── run_evaluation.py       # (Step 6) Runs mAP evaluation on a dataset
    ├── run_predict.py          # (Step 7) Runs prediction on a single image
|
└── requirements.txt            # Python dependencies
```

## 4. Setup Instructions

### Clone the Repository (or setup your project):

```bash
git clone https://github.com/Bitupan132/dip_course_project_restormer_analysis.git
cd dip_course_project_restormer_analysis
```

### Create a Virtual Environment:

```bash
python3 -m venv dip_project_env
source dip_project_env/bin/activate
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If you are on a CUDA-enabled machine, install PyTorch with CUDA support first by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

### Download Data:

- Download the COCO 2017 Validation images (`val2017.zip`) and extract them to `data/val2017/`.
- Download the COCO 2017 Annotations (`annotations_trainval2017.zip`), extract them, and place `instances_val2017.json` in `data/annotations/`.

### Download Pre-trained Restormer Models:

Download the pre-trained models (e.g., `gaussian_color_denoising_sigma25.pth`, `motion_deblurring.pth`, etc.) from the [Restormer repository](https://github.com/swz30/Restormer).

Place all `.pth` files in the `src/Denoising/pretrained_models/` and `src/Motion_Deblurring/pretrained_models/` directory.

## 5. Execution Workflow

All scripts are run from the `src/` directory.

```bash
cd src
```

### Step 1: (One Time) Convert Original Images to PNG

This script nconverts the 5,000 origial `.jpg` files from COCO into a lossless `.png` format.

```bash
python convert_to_png.py
```

### Step 2: (One Time for each Dataset) Convert Annotations to YOLO .txt

This script reads the single `instances_val2017.json` file and creates 5,000 corresponding `.txt` label files for each dataset directory.

```bash
python convert_annotations_to_txt.py --images_dir ../data/degraded_images/gaussian_noise_med/
```

### Step 3: Create Degraded Datasets

This script generates a full 5,000-image dataset for a specific degradation type. You must run this for each of the 10 degradation levels.

```bash
# Example for medium Gaussian noise
python create_degraded_dataset.py --type noise_med

# Example for horizontal motion blur (kernel 17)
python create_degraded_dataset.py --type blur_high_hor
```

### Step 4: Create Restored Datasets

This script reads a degraded dataset and runs the appropriate Restormer model to restore it. You must run this for each of the 10 degraded datasets.

```bash
# Example for restoring medium Gaussian noise
python restore.py --task Gaussian_Color_Denoising --input_dir '../data/degraded_images/gaussian_noise_high/' --result_dir '../data/restored_images/gaussian_noise_high/' --sigma 50

# Example for restoring motion blur
python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur_med_hor/' --result_dir '../data/restored_images/motion_blur_med_hor'
```

### Step 5: Run mAP Evaluation

This script runs the full YOLOv8 mAP evaluation on a single dataset. Run this for all 21 of the datasets (1 original, 10 degraded, 10 restored) to get the final quantitative results.

```bash
# Example for the original (clean) dataset
python run_evaluation.py --dataset original --yaml ../data/original_val.yaml

# Example for a degraded dataset
python run_evaluations.py --dataset degraded_noise_med --yaml ../data/dataset_yaml/degraded_gaussian_noise_med.yaml

# Example for a restored dataset
python run_evaluations.py --dataset restored_noise_med --yaml ../data/dataset_yaml/restored_gaussian_noise_med.yaml
```

### Step 6: (Optional) Run Qualitative Prediction

This script runs prediction on a single image and saves the annotated output.

```bash
# Example:
python run_predict.py \
--source ../data/original_images/000000000776.jpg \
--name predict_original_teddy
```
