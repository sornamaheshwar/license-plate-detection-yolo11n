from ultralytics import YOLO


def main():
    # Load pretrained YOLO11n
    model = YOLO("yolo11n.pt")

    # Fine-tune on license plate dataset
    results = model.train(
        data="dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        project="runs",
        name="license_plate_yolo11n",
        patience=8,
        workers=0
    )

    print("Training completed!")


if __name__ == "__main__":
    main()