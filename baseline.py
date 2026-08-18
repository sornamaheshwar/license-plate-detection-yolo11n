from ultralytics import YOLO
from pathlib import Path

# Load the untouched pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Test image directory
test_images = Path("dataset/test/images")

# Run inference on the test set
results = model(
    source=str(test_images),
    device=0,
    conf=0.25,
    save=True,
    project="runs/baseline",
    name="pretrained"
)

print("\nBaseline inference completed.")
print(f"Images evaluated: {len(results)}")
print("Results saved to: runs/baseline/pretrained")