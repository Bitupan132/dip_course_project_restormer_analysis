import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import configs

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

    return Image.fromarray(noisy_image)

def apply_motion_blur(image, kernel_size = configs.BLUR_KERNEL_SIZE, angle = configs.BLUR_ANLGE_HORIZONTAL):
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

    return Image.fromarray(blurred_image)

def main():
    os.makedirs(configs.DEGRADED_NOISE_DIR, exist_ok=True)
    os.makedirs(configs.DEGRADED_BLUR_DIR, exist_ok=True)

    img1 = custom_pil_imread(f"{configs.ORIGINAL_IMG_DIR}/000000000776.png")
    add_gaussian_noise(img1, configs.MEAN, configs.SIGMA_LOW).save(f'{configs.DEGRADED_NOISE_DIR}/noisy_image_low.png')
    add_gaussian_noise(img1, configs.MEAN, configs.SIGMA_MED).save(f'{configs.DEGRADED_NOISE_DIR}/noisy_image_med.png')
    add_gaussian_noise(img1, configs.MEAN, configs.SIGMA_HIGH).save(f'{configs.DEGRADED_NOISE_DIR}/noisy_image_high.png')

    img2 = custom_pil_imread(f"{configs.ORIGINAL_IMG_DIR}/000000000785.png")
    apply_motion_blur(img2, configs.BLUR_KERNEL_SIZE, configs.BLUR_ANLGE_HORIZONTAL).save(f'{configs.DEGRADED_BLUR_DIR}/motion_blurred_image_hor.png')
    apply_motion_blur(img2, configs.BLUR_KERNEL_SIZE, configs.BLUR_ANGLE_VERTICAL).save(f'{configs.DEGRADED_BLUR_DIR}/motion_blurred_image_ver.png')
    apply_motion_blur(img2, configs.BLUR_KERNEL_SIZE, configs.BLUR_ANGLE_DIAGONAL).save(f'{configs.DEGRADED_BLUR_DIR}/motion_blurred_image_dia.png')

    return

if __name__ == '__main__':
    main()