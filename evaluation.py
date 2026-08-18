from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/license_plate_yolo11n-3/weights/best.pt"
)

metrics = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0
)

print("\nTEST SET RESULTS")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")