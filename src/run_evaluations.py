from ultralytics import YOLO
import argparse
import configs

def run_evaluation():

    parser = argparse.ArgumentParser(description="Run YOLOv8 evaluation on a specific dataset.")
    parser.add_argument(
        '--dataset', 
        required=True, 
        type=str, 
        choices=['original', 
                 'degraded_noise_low', 'degraded_noise_med', 'degraded_noise_high', 'degraded_blur_low_hor', 'degraded_blur_low_ver', 'degraded_blur_med_hor', 'degraded_blur_med_ver', 'degraded_blur_high_hor', 'degraded_blur_high_ver', 
                 'restored_noise_low', 'restored_noise_med', 'restored_noise_high', 'restored_blur_low_hor', 'restored_blur_low_ver', 'restored_blur_med_hor', 'restored_blur_med_ver', 'restored_blur_high_hor', 'restored_blur_high_ver'],
        help="The type of dataset to evaluate."
    )
    parser.add_argument('--yaml', required=True, type=str, help='Path to the dataset yaml file')
    args = parser.parse_args()

    output_project_dir = configs.OUTPUT_PROJECT_DIR
    yaml_file = args.yaml
    print(yaml_file)
    output_name = 'map_' + args.dataset
    print(output_name)

    print(f"\n--- Running evaluation on {args.dataset} images ---")

    model = YOLO('yolov8s.pt')

    # if args.dataset == 'degraded_blur':
    #     yaml_file = configs.DEGRADED_BLUR_YAML
    #     output_name = 'map_degraded_blur'
    #     print("\n--- Running evaluation on DEGRADED (BLUR) images ---")

    print(f"Using config file: {yaml_file}")
    results = model.val(
        data=yaml_file,
        project=output_project_dir,
        name=output_name,
        imgsz=640,
        split='val',
        save_json=True,
        plots=True,
        verbose=True
    )

    print("\n--- Evaluation complete! ---")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Results are saved in: {output_project_dir}/{output_name}")

if __name__ == "__main__":
    run_evaluation()