from ultralytics import YOLO

# Load the best trained model
model = YOLO(
    "runs/detect/runs/license_plate_yolo11n-3/weights/best.pt"
)

# Run inference on an image
results = model(
    source="dataset/test/images/00d9db3d2c186504_jpg.rf.5a493e083834aa4b4748f09a073cc200.jpg",
    device=0,
    conf=0.25,
    save=True,
    project="runs/inference",
    name="license_plate"
)

# Display detection information
for result in results:
    boxes = result.boxes

    print(f"\nDetected license plates: {len(boxes)}")

    for i, box in enumerate(boxes):
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print(
            f"Plate {i + 1}: "
            f"confidence={confidence:.3f}, "
            f"box={coordinates}"
        )