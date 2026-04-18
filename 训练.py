#!/usr/bin/env python3
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


DATA = str(Path("datasets/data.yaml").resolve())
MODEL = "yolo26n-seg.pt"
EPOCHS = 100
IMGSZ = 640
BATCH = 16
PROJECT = ROOT / "runs"
NAME = "seg"


def auto_device():
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return 0

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main() -> None:


    device = auto_device()
    model = YOLO(MODEL)
    results = model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        project=str(PROJECT),
        name=NAME,
        workers=4,
    )

    print(f"使用设备: {device}")
    print(f"训练完成: {Path(results.save_dir).resolve()}")


if __name__ == "__main__":
    main()
