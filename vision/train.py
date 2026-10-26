from ultralytics import YOLO

DATA_YAML = "/home/lucas/vira/vision/datasets_ppe6/data.yaml"
PROJECT = "/home/lucas/vira/vision/runs"


def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=100,
        patience=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        project=PROJECT,
        name="ppe6_yolo11n",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
