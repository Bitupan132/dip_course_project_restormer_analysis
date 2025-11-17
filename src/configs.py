import os

# All paths are relative to the 'src' directory (where this file lives).

# --- 1. Base Directories ---
DATA_DIR = '../data'
PRETRAINED_MODEL_NOISE_DIR = './Denoising/pretrained_models'
PRETRAINED_MODEL_BLUR_DIR = './Motion_Deblurring/pretrained_models'
OUTPUT_PROJECT_DIR = '../data/detection_output'
COCO_JSON_PATH = '../data/annotations/instances_val2017.json'
OUTPUT_PREDICT_DIR = '../data/prediction_output'

# --- 2. Input Data Paths ---
VAL_2017_DIR = os.path.join(DATA_DIR, 'val2017')
ORIGINAL_IMG_DIR = os.path.join(DATA_DIR, 'original_images')

# --- 3. Degraded Data Paths ---
DEGRADED_IMAGES_DIR = os.path.join(DATA_DIR, 'degraded_images')
DEGRADED_NOISE_LOW_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'gaussian_noise_low')
DEGRADED_NOISE_MED_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'gaussian_noise_med')
DEGRADED_NOISE_HIGH_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'gaussian_noise_high')
DEGRADED_BLUR_LOW_HOR_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_low_hor')
DEGRADED_BLUR_LOW_VER_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_low_ver')
DEGRADED_BLUR_MED_HOR_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_med_hor')
DEGRADED_BLUR_MED_VER_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_med_ver')
DEGRADED_BLUR_HIGH_HOR_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_high_hor')
DEGRADED_BLUR_HIGH_VER_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur_high_ver')

# --- 4. Restored Data Paths ---
RESTORED_IMAGES_DIR = os.path.join(DATA_DIR, 'restored_images')
RESTORED_NOISE_LOW_DIR = os.path.join(RESTORED_IMAGES_DIR, 'gaussian_noise_low')
RESTORED_NOISE_MED_DIR = os.path.join(RESTORED_IMAGES_DIR, 'gaussian_noise_med')
RESTORED_NOISE_HIGH_DIR = os.path.join(RESTORED_IMAGES_DIR, 'gaussian_noise_high')
RESTORED_BLUR_LOW_HOR_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_low_hor')
RESTORED_BLUR_LOW_VER_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_low_ver')
RESTORED_BLUR_MED_HOR_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_med_hor')
RESTORED_BLUR_MED_VER_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_med_ver')
RESTORED_BLUR_HIGH_HOR_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_high_hor')
RESTORED_BLUR_HIGH_VER_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_blur_high_ver')

# --- 5. Project Constants ---
MEAN = 0
SIGMA_LOW = 15
SIGMA_MED = 25
SIGMA_HIGH = 50
BLUR_ANLGE_HORIZONTAL = 0
BLUR_ANGLE_VERTICAL = 90
BLUR_KERNEL_LOW = 5
BLUR_KERNEL_MED = 11
BLUR_KERNEL_HIGH = 17

# --- 6. YAML File Paths for evaluation  ---
# ORIGINAL_YAML = '../data/dataset_yaml/original_val.yaml'

# DEGRADED_NOISE_LOW_YAML = '../data/dataset_yaml/degraded_gaussian_noise_low.yaml'
# DEGRADED_NOISE_MED_YAML = '../data/dataset_yaml/degraded_gaussian_noise_med.yaml'
# DEGRADED_NOISE_HIGH_YAML = '../data/dataset_yaml/degraded_gaussian_noise_high.yaml'
# DEGRADED_BLUR_LOW_HOR_YAML = '../data/dataset_yaml/degraded_motion_blur_low_hor.yaml'
# DEGRADED_BLUR_LOW_VER_YAML = '../data/dataset_yaml/degraded_motion_blur_low_ver.yaml'
# DEGRADED_BLUR_MED_HOR_YAML = '../data/dataset_yaml/degraded_motion_blur_med_hor.yaml'
# DEGRADED_BLUR_MED_VER_YAML = '../data/dataset_yaml/degraded_motion_blur_med_ver.yaml'
# DEGRADED_BLUR_HIGH_HOR_YAML = '../data/dataset_yaml/degraded_motion_blur_high_hor.yaml'
# DEGRADED_BLUR_HIGH_VER_YAML = '../data/dataset_yaml/degraded_motion_blur_high_ver.yaml'

# RESTORED_NOISE_LOW_YAML = '../data/dataset_yaml/restored_gaussian_noise_low.yaml'
# RESTORED_NOISE_MED_YAML = '../data/dataset_yaml/restored_gaussian_noise_med.yaml'
# RESTORED_NOISE_HIGH_YAML = '../data/dataset_yaml/restored_gaussian_noise_high.yaml'
# RESTORED_BLUR_LOW_HOR_YAML = '../data/dataset_yaml/restored_motion_blur_low_hor.yaml'
# RESTORED_BLUR_LOW_VER_YAML = '../data/dataset_yaml/restored_motion_blur_low_ver.yaml'
# RESTORED_BLUR_MED_HOR_YAML = '../data/dataset_yaml/restored_motion_blur_med_hor.yaml'
# RESTORED_BLUR_MED_VER_YAML = '../data/dataset_yaml/restored_motion_blur_med_ver.yaml'
# RESTORED_BLUR_HIGH_HOR_YAML = '../data/dataset_yaml/restored_motion_blur_high_hor.yaml'
# RESTORED_BLUR_HIGH_VER_YAML = '../data/dataset_yaml/restored_motion_blur_high_ver.yaml'

# CREATE DATASET:
# python create_degraded_datasets.py --type 'noise_low'
# python create_degraded_datasets.py --type 'blur_med_hor'
# python create_degraded_datasets.py --type 'blur_med_ver'
# python create_degraded_datasets.py --type 'blur_high_hor'
# python create_degraded_datasets.py --type 'blur_high_ver'

# export CUDA_VISIBLE_DEVICES=1

# MOTION DEBLUR:

# python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur_med_hor/' --result_dir '../data/restored_images/motion_blur_med_hor'
# python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur_med_ver/' --result_dir '../data/restored_images/motion_blur_med_ver'
# python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur_high_hor/' --result_dir '../data/restored_images/motion_blur_high_hor'
# python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur_high_ver/' --result_dir '../data/restored_images/motion_blur_high_ver'


# GAUSSIAN_DENOISE:
# python restore.py --task Gaussian_Color_Denoising --input_dir '../data/degraded_images/gaussian_noise_high/' --result_dir '../data/restored_images/gaussian_noise_high/' --sigma 50


# ANNOTATIONS TO TXT:
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/gaussian_noise_med/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/gaussian_noise_high/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_med_hor/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_med_ver/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_high_hor/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_high_ver/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_low_hor/
# python convert_annotations_to_txt.py --images_dir ../data/degraded_images/motion_blur_low_ver/

# python convert_annotations_to_txt.py --images_dir ../data/restored_images/gaussian_noise_med/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/gaussian_noise_high/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_med_hor/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_med_ver/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_high_hor/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_high_ver/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_low_hor/
# python convert_annotations_to_txt.py --images_dir ../data/restored_images/motion_blur_low_ver/


# EVALUATE
# python run_evaluations.py --dataset degraded_noise_med --yaml ../data/dataset_yaml/degraded_gaussian_noise_med.yaml
# python run_evaluations.py --dataset degraded_noise_high --yaml ../data/dataset_yaml/degraded_gaussian_noise_high.yaml
# python run_evaluations.py --dataset degraded_blur_low_hor --yaml ../data/dataset_yaml/degraded_motion_blur_low_hor.yaml
# python run_evaluations.py --dataset degraded_blur_low_ver --yaml ../data/dataset_yaml/degraded_motion_blur_low_ver.yaml
# python run_evaluations.py --dataset degraded_blur_med_hor --yaml ../data/dataset_yaml/degraded_motion_blur_med_hor.yaml
# python run_evaluations.py --dataset degraded_blur_med_ver --yaml ../data/dataset_yaml/degraded_motion_blur_med_ver.yaml
# python run_evaluations.py --dataset degraded_blur_high_hor --yaml ../data/dataset_yaml/degraded_motion_blur_high_hor.yaml
# python run_evaluations.py --dataset degraded_blur_high_ver --yaml ../data/dataset_yaml/degraded_motion_blur_high_ver.yaml

# python run_evaluations.py --dataset restored_noise_med --yaml ../data/dataset_yaml/restored_gaussian_noise_med.yaml
# python run_evaluations.py --dataset restored_noise_high --yaml ../data/dataset_yaml/restored_gaussian_noise_high.yaml
# python run_evaluations.py --dataset restored_blur_low_hor --yaml ../data/dataset_yaml/restored_motion_blur_low_hor.yaml
# python run_evaluations.py --dataset restored_blur_low_ver --yaml ../data/dataset_yaml/restored_motion_blur_low_ver.yaml
# python run_evaluations.py --dataset restored_blur_med_hor --yaml ../data/dataset_yaml/restored_motion_blur_med_hor.yaml
# python run_evaluations.py --dataset restored_blur_med_ver --yaml ../data/dataset_yaml/restored_motion_blur_med_ver.yaml
# python run_evaluations.py --dataset restored_blur_high_hor --yaml ../data/dataset_yaml/restored_motion_blur_high_hor.yaml
# python run_evaluations.py --dataset restored_blur_high_ver --yaml ../data/dataset_yaml/restored_motion_blur_high_ver.yaml


# PREDICTIONS:
# # 000000298251, 000000460682, 000000485237, 000000329614, 000000242934, 000000050943
# python run_predict.py --source ../data/original_images/000000050943.png --name predict_original
# python run_predict.py --source ../data/degraded_images/gaussian_noise_high/000000050943.png --name predict_degraded_noise
# python run_predict.py --source ../data/restored_images/gaussian_noise_high/000000050943.png --name predict_restored_noise

# # 000000460682, 000000485237, 000000209972, 000000425702, 000000050943
# python run_predict.py --source ../data/original_images/000000050943.png --name predict_original_blur
# python run_predict.py --source ../data/degraded_images/motion_blur_high_hor/000000050943.png --name predict_degraded_blur
# python run_predict.py --source ../data/restored_images/motion_blur_high_hor/000000050943.png --name predict_restored_blur