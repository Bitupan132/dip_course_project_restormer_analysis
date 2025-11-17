from ultralytics import YOLO
import argparse
import configs
import os

def run_prediction():
    parser = argparse.ArgumentParser(description="Run YOLOv8 prediction on a single image.")
    
    parser.add_argument(
        '--source', 
        required=True, 
        type=str, 
        help="Path to the single source image (e.g., ../data/original_images/000000000776.jpg)"
    )
    
    parser.add_argument(
        '--name', 
        required=True, 
        type=str,
        help="A unique name for the output folder (e.g., 'original_teddy_bear')"
    )
    
    args = parser.parse_args()

    print(f"Loading YOLOv8s model...")
    model = YOLO('yolov8s.pt')

    output_project_dir = configs.OUTPUT_PREDICT_DIR
    
    print(f"\n--- Running prediction on: {args.source} ---")

    # run the model and save an annotated image
    model.predict(
        source=args.source,
        project=output_project_dir,
        name=args.name,
        imgsz=640,
        save=True,
        verbose=True
    )

    print("\n--- Prediction complete! ---")
    print(f"Result saved in: {output_project_dir}/{args.name}")

if __name__ == "__main__":
    run_prediction()