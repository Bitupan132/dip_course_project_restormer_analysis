import os

# All paths are relative to the 'src' directory (where this file lives).

# --- 1. Base Directories ---
DATA_DIR = '../data'
PRETRAINED_MODEL_NOISE_DIR = './Denoising/pretrained_models'
PRETRAINED_MODEL_BLUR_DIR = './Motion_Deblurring/pretrained_models'

# --- 2. Input Data Paths ---
VAL_2017_DIR = os.path.join(DATA_DIR, 'val2017')
ORIGINAL_IMG_DIR = os.path.join(DATA_DIR, 'original_images')

# --- 3. Degraded Data Paths ---
DEGRADED_IMAGES_DIR = os.path.join(DATA_DIR, 'degraded_images')
DEGRADED_NOISE_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'gaussian_noise')
DEGRADED_BLUR_DIR = os.path.join(DEGRADED_IMAGES_DIR, 'motion_blur')

# --- 4. Restored Data Paths ---
RESTORED_IMAGES_DIR = os.path.join(DATA_DIR, 'restored_images')
RESTORED_NOISE_DIR = os.path.join(RESTORED_IMAGES_DIR, 'gaussian_denoise')
RESTORED_BLUR_DIR = os.path.join(RESTORED_IMAGES_DIR, 'motion_deblur')

# --- 5. Project Constants ---
MEAN = 0
SIGMA_LOW = 15
SIGMA_MED = 25
SIGMA_HIGH = 50
BLUR_ANLGE_HORIZONTAL = 0
BLUR_ANGLE_VERTICAL = 90
BLUR_ANGLE_DIAGONAL = 45
BLUR_KERNEL_SIZE = 13

# Motion Deblur:
# python restore.py --task Motion_Deblurring --input_dir '../data/degraded_images/motion_blur/' --result_dir '../data/restored_images/motion_deblur/'

# Gaussian Denoise:
# python restore.py --task Gaussian_Color_Denoising --input_dir '../data/degraded_images/gaussian_noise/' --result_dir '../data/restored_images/gaussian_denoise/' --sigma 25