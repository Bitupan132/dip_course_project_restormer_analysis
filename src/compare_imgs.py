import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def main():

    degraded_path = '../data/degraded_output/gaussian_noise/noisy_image_med.jpg'
    restored_path = '../data/restored_output/gaussian_denoise/noisy_image_med.png'
    original_path = '../data/val2017/000000000776.jpg'

    degraded_img = skimage.io.imread(degraded_path)
    restored_img = skimage.io.imread(restored_path)
    restored_img = restored_img.astype(degraded_img.dtype)
    original_img = skimage.io.imread(original_path)

    psnr_deg_res = psnr(degraded_img, restored_img, data_range=255)
    ssim_deg_res = ssim(degraded_img, restored_img, 
                data_range=255, 
                multichannel=True,
                channel_axis=2)
    psnr_org_res = psnr(original_img, restored_img, data_range=255)
    ssim_org_res = ssim(original_img, restored_img, 
                data_range=255, 
                multichannel=True,
                channel_axis=2)

    print(f"PSNR (Degraded, Restored): {psnr_deg_res}")
    print(f"SSIM (Degraded, Restored): {ssim_deg_res}")

    print(f"PSNR (Original, Restored): {psnr_org_res}")
    print(f"SSIM (Original, Restored): {ssim_org_res}")

    diff = restored_img - degraded_img
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12,8))

    plt.subplot(131)
    plt.title("degraded")
    plt.imshow(degraded_img)

    plt.subplot(132)
    plt.title("restored")
    plt.imshow(restored_img)

    plt.subplot(133)
    plt.title("diff")
    plt.imshow(diff)

    plt.show()

    return
if __name__ == '__main__':
    main()