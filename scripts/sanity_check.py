"""Sanity check — sprawdza środowisko i odpala YOLOv8n na obrazku testowym.

Demo systemu console + report, który będziemy używać w treningu i ewaluacji.
"""
import sys
import time
from pathlib import Path

# Pozwól importować src/ z roota projektu
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO

from src import console as cn
from src.report import (
    format_inference_section,
    write_report,
)


def main() -> None:
    cn.phase(
        "Sanity check — YOLOv8n inference",
        "Weryfikacja środowiska + pretrained inference na obrazku testowym",
    )

    cn.info("Środowisko:")
    cn.kv("PyTorch", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    cn.kv("CUDA available", cuda_ok)
    if cuda_ok:
        cn.kv("CUDA device", torch.cuda.get_device_name(0))
        cn.kv("VRAM total", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_path = Path("models/pretrained/yolov8n.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with cn.step(f"Ładuję pretrained model ({model_path.name})"):
        model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
        leftover = Path("yolov8n.pt")
        if leftover.exists() and leftover.resolve() != model_path.resolve():
            leftover.rename(model_path)

    out_dir = (Path.cwd() / "results" / "predictions" / "sanity").resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with cn.step("Inference na obrazku testowym (Ultralytics bus.jpg)"):
        t0 = time.perf_counter()
        results = model.predict(
            source="https://ultralytics.com/images/bus.jpg",
            save=True,
            project=str(out_dir.parent),
            name=out_dir.name,
            exist_ok=True,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

    r = results[0]
    detected = [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []

    cn.console.print()
    cn.console.print(
        cn.metrics_table(
            "Wyniki detekcji",
            ["Obiekt", "Liczba"],
            [[name, detected.count(name)] for name in sorted(set(detected))]
            + [["RAZEM", len(detected)]],
        )
    )

    inference_stats = {
        "avg_ms": elapsed_ms,
        "fps": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0,
        "resolution": "640x480",
        "num_images": 1,
        "num_detections": len(detected),
    }

    report_path = Path("results/sanity_report.txt")
    with cn.step(f"Zapisuję raport do {report_path}"):
        write_report(
            output_path=report_path,
            title="YOLO Sign Detection — Sanity Check Report",
            metadata={
                "Model": "YOLOv8n (pretrained, COCO)",
                "Hardware": torch.cuda.get_device_name(0) if cuda_ok else "CPU",
                "PyTorch": torch.__version__,
                "Plik wyjściowy": str(out_dir / "bus.jpg"),
            },
            sections=[
                ("WYDAJNOŚĆ INFERENCE", format_inference_section(inference_stats)),
                (
                    "WYKRYTE OBIEKTY",
                    "\n".join(f"  - {name}: {detected.count(name)}" for name in sorted(set(detected)))
                    + f"\n\nRAZEM: {len(detected)} obiektów",
                ),
            ],
        )

    cn.console.print()
    cn.success(f"Sanity check przeszedł — {len(detected)} obiektów wykrytych w {elapsed_ms:.1f} ms")
    cn.info(f"Output: {out_dir / 'bus.jpg'}")
    cn.info(f"Raport: {report_path}")


if __name__ == "__main__":
    main()
