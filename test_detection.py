
from ultralytics import YOLO
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())

    # Load a small pretrained object detection model.
    model = YOLO("yolo11n.pt")

    print("Running a sample prediction...")

    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=0,
        imgsz=640,
        conf=0.35,
        save=True,
        verbose=False,
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            print(f"{label}: {confidence:.2%}")

    print("\nDetection test complete.")
    print("Annotated output saved by Ultralytics.")

if __name__ == "__main__":
    main()
