"""Sanity check — sprawdza czy YOLO ładuje się i odpala inference.

Uruchamia pretrained yolov8n na obrazku testowym z Ultralytics i zapisuje
wynik do results/predictions/sanity/. Powinien wykryć autobus, osoby itp.
"""
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_path = Path("models/pretrained/yolov8n.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    if not model_path.exists():
        Path("yolov8n.pt").rename(model_path)

    out_dir = (Path.cwd() / "results" / "predictions" / "sanity").resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    for r in results:
        names = r.names
        detected = [names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
        print(f"Wykryto {len(detected)} obiektów: {detected}")


if __name__ == "__main__":
    main()
