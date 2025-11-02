from ultralytics import YOLO
import argparse
import configs

def run_evaluation():

    parser = argparse.ArgumentParser(description="Run YOLOv8 evaluation on a specific dataset.")
    parser.add_argument(
        '--dataset', 
        required=True, 
        type=str, 
        choices=['original', 'degraded_noise', 'degraded_blur', 'restored_noise', 'restored_blur'],
        help="The type of dataset to evaluate."
    )
    args = parser.parse_args()

    model = YOLO('yolov8s.pt')

    output_project_dir = configs.OUTPUT_PROJECT_DIR

    if args.dataset == 'original':
        yaml_file = configs.ORIGINAL_YAML
        output_name = 'map_original_clean'
        print("\n--- Running evaluation on ORIGINAL images ---")

    elif args.dataset == 'degraded_noise':
        yaml_file = configs.DEGRADED_NOISE_YAML
        output_name = 'map_degraded_noise'
        print("\n--- Running evaluation on DEGRADED (NOISE) images ---")

    elif args.dataset == 'restored_noise':
        yaml_file = configs.RESTORED_NOISE_YAML
        output_name = 'map_restored_noise'
        print("\n--- Running evaluation on RESTORED (NOISE) images ---")
    
    elif args.dataset == 'degraded_blur':
        yaml_file = configs.DEGRADED_BLUR_YAML
        output_name = 'map_degraded_blur'
        print("\n--- Running evaluation on DEGRADED (BLUR) images ---")
        
    elif args.dataset == 'restored_blur':
        yaml_file = configs.RESTORED_BLUR_YAML
        output_name = 'map_restored_blur'
        print("\n--- Running evaluation on RESTORED (BLUR) images ---")

    print(f"Using config file: {yaml_file}")
    model.val(
        data=yaml_file,
        imgsz=640,
        split='val',
        project=output_project_dir,
        name=output_name
    )

    print("\n--- Evaluation complete! ---")
    print(f"Results are saved in: {output_project_dir}/{output_name}")

if __name__ == "__main__":
    run_evaluation()