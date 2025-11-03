import numpy as np
import cv2
from PIL import Image
import os
import configs
import argparse
import glob
from tqdm import tqdm

def custom_pil_imread(filepath):
    pil_img = Image.open(filepath)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return np.array(pil_img)
    
def add_gaussian_noise(image, mean = configs.MEAN, sigma = configs.SIGMA_MED):
    row, col, ch = image.shape

    gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))

    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def apply_motion_blur(image, kernel_size = configs.BLUR_KERNEL_LOW, angle = configs.BLUR_ANLGE_HORIZONTAL):
    # angle 0: horizontal motion blur
    # angle 90: vertical motion blur

    center = (kernel_size - 1) // 2

    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)

    motion_blur_kernel = np.zeros((kernel_size, kernel_size))   #, dtype=np.float32)
    # Create a line of Ones in the center of the kernel
    motion_blur_kernel[center, :] = np.ones(kernel_size)    
    # Rotate the line to the provided angle
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, rotation_matrix, (kernel_size, kernel_size))
    motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel) # Normalize

    blurred_image = cv2.filter2D(image, -1, motion_blur_kernel)

    return blurred_image

def create_degraded_dataset():
    parser = argparse.ArgumentParser(description="Create degraded dataset.")
    parser.add_argument('--type', required=True, type=str,
        choices=['noise_low', 'noise_med', 'noise_high', 'blur_low_hor', 'blur_low_ver', 'blur_med_hor', 'blur_med_ver', 'blur_high_hor', 'blur_high_ver'],
        help="The type of degradation to apply."
    )
    args = parser.parse_args()

    if args.type == 'noise_low':
        output_dir = configs.DEGRADED_NOISE_LOW_DIR
    elif args.type == 'noise_med':
        output_dir = configs.DEGRADED_NOISE_MED_DIR
    elif args.type == 'noise_high':
        output_dir = configs.DEGRADED_NOISE_HIGH_DIR
    elif args.type == 'blur_low_hor':
        output_dir = configs.DEGRADED_BLUR_LOW_HOR_DIR
    elif args.type == 'blur_low_ver':
        output_dir = configs.DEGRADED_BLUR_LOW_VER_DIR
    elif args.type == 'blur_med_hor':
        output_dir = configs.DEGRADED_BLUR_MED_HOR_DIR
    elif args.type == 'blur_med_ver':
        output_dir = configs.DEGRADED_BLUR_MED_VER_DIR
    elif args.type == 'blur_high_hor':
        output_dir = configs.DEGRADED_BLUR_HIGH_HOR_DIR
    elif args.type == 'blur_high_ver':
        output_dir = configs.DEGRADED_BLUR_HIGH_VER_DIR
    print(f"output dir:{output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    input_dir = configs.ORIGINAL_IMG_DIR
    original_image_paths = glob.glob(os.path.join(input_dir, '*.png'))

    print(f"\nFound {len(original_image_paths)} images to degrade.")

    i = 1

    for image_path in tqdm(original_image_paths, desc=f"Processing {args.type}."):
        img = custom_pil_imread(image_path)

        if args.type == 'noise_low':
            degraded_img = add_gaussian_noise(img, configs.MEAN, configs.SIGMA_LOW)
        elif args.type == 'noise_med':
            degraded_img = add_gaussian_noise(img, configs.MEAN, configs.SIGMA_MED)
        elif args.type == 'noise_high':
            degraded_img = add_gaussian_noise(img, configs.MEAN, configs.SIGMA_HIGH)
        elif args.type == 'blur_low_hor':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_LOW, configs.BLUR_ANLGE_HORIZONTAL)
        elif args.type == 'blur_low_ver':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_LOW, configs.BLUR_ANGLE_VERTICAL)
        elif args.type == 'blur_med_hor':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_MED, configs.BLUR_ANLGE_HORIZONTAL)
        elif args.type == 'blur_med_ver':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_MED, configs.BLUR_ANGLE_VERTICAL)
        elif args.type == 'blur_high_hor':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_HIGH, configs.BLUR_ANLGE_HORIZONTAL)
        elif args.type == 'blur_high_ver':
            degraded_img = apply_motion_blur(img, configs.BLUR_KERNEL_HIGH, configs.BLUR_ANGLE_VERTICAL)

        base_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, base_filename)
        Image.fromarray(degraded_img).save(output_path, format = 'PNG')

    print("\n--- Full dataset degradation complete! ---")
    return

if __name__ == '__main__':
    create_degraded_dataset()