import numpy as np
import skimage.io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import configs
import argparse
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from create_degraded_datasets import custom_pil_imread

def main():
    parser = argparse.ArgumentParser(description="Calculate average PSNR/SSIM for a full dataset.")
    parser.add_argument(
        '--type', 
        required=True, 
        type=str,
        choices=[
            'noise_low', 'noise_med', 'noise_high', 
            'blur_low_hor', 'blur_low_ver', 
            'blur_med_hor', 'blur_med_ver', 
            'blur_high_hor', 'blur_high_ver'
        ],
        help="The type of degradation to evaluate."
    )
    args = parser.parse_args()

    type_to_paths = {
        'noise_low': (configs.DEGRADED_NOISE_LOW_DIR, configs.RESTORED_NOISE_LOW_DIR),
        'noise_med': (configs.DEGRADED_NOISE_MED_DIR, configs.RESTORED_NOISE_MED_DIR),
        'noise_high': (configs.DEGRADED_NOISE_HIGH_DIR, configs.RESTORED_NOISE_HIGH_DIR),
        'blur_low_hor': (configs.DEGRADED_BLUR_LOW_HOR_DIR, configs.RESTORED_BLUR_LOW_HOR_DIR),
        'blur_low_ver': (configs.DEGRADED_BLUR_LOW_VER_DIR, configs.RESTORED_BLUR_LOW_VER_DIR),
        'blur_med_hor': (configs.DEGRADED_BLUR_MED_HOR_DIR, configs.RESTORED_BLUR_MED_HOR_DIR),
        'blur_med_ver': (configs.DEGRADED_BLUR_MED_VER_DIR, configs.RESTORED_BLUR_MED_VER_DIR),
        'blur_high_hor': (configs.DEGRADED_BLUR_HIGH_HOR_DIR, configs.RESTORED_BLUR_HIGH_HOR_DIR),
        'blur_high_ver': (configs.DEGRADED_BLUR_HIGH_VER_DIR, configs.RESTORED_BLUR_HIGH_VER_DIR),
    }

    original_dir = configs.ORIGINAL_IMG_DIR
    degraded_dir, restored_dir = type_to_paths[args.type]

    print(f"\n--- Calculating metrics for type: {args.type} ---")
    print(f"Original Dir:  {original_dir}")
    print(f"Degraded Dir:  {degraded_dir}")
    print(f"Restored Dir:  {restored_dir}")


    original_image_paths = glob(os.path.join(original_dir, '*.png'))
    if not original_image_paths:
        print(f"Error: No .png images found in {original_dir}")
        return

    psnr_deg_res_list = []
    ssim_deg_res_list = []
    psnr_org_res_list = []
    ssim_org_res_list = []

    for original_path in tqdm(original_image_paths, desc="Processing images"):
        filename = os.path.basename(original_path)
        degraded_path = os.path.join(degraded_dir, filename)
        restored_path = os.path.join(restored_dir, filename)

        # Load images
        original_img = custom_pil_imread(original_path)
        degraded_img = custom_pil_imread(degraded_path)
        restored_img = custom_pil_imread(restored_path)

        # Handle missing files
        if original_img is None or degraded_img is None or restored_img is None:
            print(f"Warning: Skipping {filename} due to loading error.")
            continue
            
        # Ensure data types match for comparison
        restored_img = restored_img.astype(degraded_img.dtype)
        
        # A. Degraded vs. Restored
        psnr_deg_res_list.append(psnr(degraded_img, restored_img, data_range=255))
        ssim_deg_res_list.append(ssim(degraded_img, restored_img, data_range=255, multichannel=True, channel_axis=2))
        
        # B. Original vs. Restored
        psnr_org_res_list.append(psnr(original_img, restored_img, data_range=255))
        ssim_org_res_list.append(ssim(original_img, restored_img, data_range=255, multichannel=True, channel_axis=2))

    if not psnr_org_res_list:
        print("Error: No images were processed. Check file paths.")
        return

    print("\n--- Average Metrics for 5000 Images ---")
    
    print("\n(Original vs. Restored)")
    print(f"  Avg. PSNR: {np.mean(psnr_org_res_list):.4f}")
    print(f"  Avg. SSIM: {np.mean(ssim_org_res_list):.4f}")
    
    print("\n(Degraded vs. Restored)")
    print(f"  Avg. PSNR: {np.mean(psnr_deg_res_list):.4f}")
    print(f"  Avg. SSIM: {np.mean(ssim_deg_res_list):.4f}")

    return

if __name__ == '__main__':
    main()