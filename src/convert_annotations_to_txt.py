import json
import os
import argparse
from pathlib import Path
import configs
import glob

# COCO 80 class category IDs (not contiguous!)
COCO_80_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

def convert_coco_to_yolo(images_dir):

    # Load COCO annotations
    coco_json_path = configs.COCO_JSON_PATH
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create category_id to YOLO class_id mapping (0-79)
    coco_id_to_yolo_id = {}
    for yolo_idx, coco_cat_id in enumerate(COCO_80_CATEGORY_IDS):
        coco_id_to_yolo_id[coco_cat_id] = yolo_idx
    
    print(f"Category mapping created: {len(coco_id_to_yolo_id)} classes")
    
    # # Create output directory
    # os.makedirs(images_dir, exist_ok=True)
    # print(f"Output directory: {images_dir}")
    
    # Create mappings
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Get list of images in the images_dir
    image_files = {os.path.basename(f) for f in glob.glob(os.path.join(images_dir, '*.png'))}
    print(f"\nTotal images in input directory: {len(image_files)}")
    
    converted_count = 0
    skipped_count = 0
    
    # Convert each image's annotations
    for img_id, annotations in annotations_by_image.items():
        filename = image_id_to_filename[img_id]

        filename_png = filename.split('.')[0] + '.png'

        if filename_png not in image_files:
            skipped_count += 1
            continue

        img_width, img_height = image_id_to_size[img_id]
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = os.path.join(images_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # COCO format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                # Map COCO category_id to YOLO class_id (0-79)
                coco_cat_id = ann['category_id']
                if coco_cat_id not in coco_id_to_yolo_id:
                    print(f"Warning: Unknown category_id {coco_cat_id} in image {filename}")
                    continue
                
                yolo_class_id = coco_id_to_yolo_id[coco_cat_id]
                
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        converted_count += 1
    
    print(f"Converted: {converted_count} images")
    print(f"Skipped: {skipped_count} images (not found in images directory)")
    print(f"\nConversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO format")
    parser.add_argument('--images_dir', required=True, help='Directory containing the images')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.images_dir)