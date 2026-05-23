import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Driver Distraction Dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default is 3 for quick local test)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (lower for CPU/memory safety)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (e.g. cpu, 0, cuda)")
    args = parser.parse_args()

    # Load a pretrained YOLOv8 nano model
    print("Loading YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")

    # Resolve absolute path to data.yaml
    data_yaml_path = os.path.abspath("data.yaml")
    print(f"Dataset config path: {data_yaml_path}")

    # Start training
    print(f"Starting training on device: {args.device} for {args.epochs} epochs...")
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # Avoid multi-processing issues on Windows CPU training
        project="driver_distraction",
        name="yolov8n_local"
    )
    
    print("\nTraining completed successfully!")
    print("Local model weights saved at: driver_distraction/yolov8n_local/weights/best.pt")

if __name__ == "__main__":
    main()
