import os
import numpy as np
import glob
import skimage.io
from PIL import Image
from tqdm import tqdm
import configs

def convert_val2017_to_png():
    input_dir = configs.VAL_2017_DIR
    output_dir = configs.ORIGINAL_IMG_DIR

    os.makedirs(output_dir, exist_ok=True)

    jpg_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    # jpg_files = glob.glob(os.path.join(input_dir, '000000000785.jpg'))
    
    print(f"\nFound {len(jpg_files)} JPG images to convert.")

    for jpg_path in tqdm(jpg_files, desc="Converting images to PNG"):
        try:
            pil_img = Image.open(jpg_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            base_filename = os.path.basename(jpg_path)
            png_filename = os.path.splitext(base_filename)[0] + '.png'
            output_path = os.path.join(output_dir, png_filename)

            pil_img.save(output_path, 'PNG')
            
        except Exception as e:
            print(f"Error processing {jpg_path}: {e}")

    print("\nConversion complete!")
    print(f"\nConverted {len(glob.glob(os.path.join(output_dir, '*.png')))} images to PNG.")
    print(f"All .png images are saved in: {output_dir}")

if __name__ == "__main__":
    convert_val2017_to_png()