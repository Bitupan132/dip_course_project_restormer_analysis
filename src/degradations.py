import numpy as np
import cv2
import skimage.io
from matplotlib import pyplot as plt
import os

def add_gaussian_noise(image, mean = 0, sigma = 25):
    print(image.shape)
    row, col, ch = image.shape

    gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))

    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def apply_motion_blur(image, kernel_size = 13, angle = 0):
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

def main():
    input_path = '../data/val2017'
    gaussian_degraded_output_dir = '../data/degraded_output/gaussian_noise'
    motion_degraded_output_dir = '../data/degraded_output/motion_blur'
    os.makedirs(gaussian_degraded_output_dir, exist_ok=True)
    os.makedirs(motion_degraded_output_dir, exist_ok=True)

    img1 = skimage.io.imread(f"{input_path}/000000000776.jpg")
    noisy_image_low = add_gaussian_noise(img1, 0, 15)
    noisy_image_med = add_gaussian_noise(img1, 0, 25)
    noisy_image_high = add_gaussian_noise(img1, 0, 50)

    skimage.io.imsave(f'{gaussian_degraded_output_dir}/noisy_image_low.png', noisy_image_low)
    skimage.io.imsave(f'{gaussian_degraded_output_dir}/noisy_image_med.png', noisy_image_med)
    skimage.io.imsave(f'{gaussian_degraded_output_dir}/noisy_image_high.png', noisy_image_high)

    img2 = skimage.io.imread(f"{input_path}/000000000785.jpg")
    motion_blurred_image_hor = apply_motion_blur(img2, 13, 0)
    motion_blurred_image_ver = apply_motion_blur(img2, 13, 90)
    motion_blurred_image_dia = apply_motion_blur(img2, 13, 45)

    skimage.io.imsave(f'{motion_degraded_output_dir}/motion_blurred_image_hor.png', motion_blurred_image_hor)
    skimage.io.imsave(f'{motion_degraded_output_dir}/motion_blurred_image_ver.png', motion_blurred_image_ver)
    skimage.io.imsave(f'{motion_degraded_output_dir}/motion_blurred_image_dia.png', motion_blurred_image_dia)

    return

if __name__ == '__main__':
    main()