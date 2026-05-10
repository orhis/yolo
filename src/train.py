"""Trening YOLOv8 na własnym dataset (Faza 4).

Użycie:
    python -m src.train                       # default: yolov8n, 50 epok
    python -m src.train --epochs 100
    python -m src.train --model yolov8s.pt --epochs 80
"""
import argparse
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from src import console as cn
from src.dataset_stats import collect_dataset_stats
from src.report import (
    format_dataset_section,
    format_training_section,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trening YOLOv8 na własnym dataset")
    p.add_argument("--model", default="yolov8n.pt", help="Bazowy model (yolov8n.pt, yolov8s.pt, ...)")
    p.add_argument("--data", default="data/dataset.yaml", help="Ścieżka do dataset.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="signs", help="Nazwa run-a (results/runs/<name>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cn.phase("Faza 4 — Trening YOLOv8", f"Model: {args.model} • Epoki: {args.epochs} • imgsz: {args.imgsz}")

    cn.info("Środowisko:")
    cn.kv("PyTorch", torch.__version__)
    cn.kv("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        cn.kv("Device", torch.cuda.get_device_name(0))
    cn.kv("Dataset config", args.data)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        cn.error(f"Nie znaleziono {data_yaml}. Najpierw wykonaj Fazę 3 (anotacja + split).")
        raise SystemExit(1)

    with cn.step("Wczytuję statystyki datasetu"):
        ds_stats = collect_dataset_stats(data_yaml)

    cn.console.print(cn.metrics_table(
        "Dataset",
        ["Klasa", "Train", "Val", "Test", "Razem"],
        [[pc["name"], pc["train"], pc["val"], pc["test"], pc["total"]] for pc in ds_stats["per_class"]]
        + [["RAZEM", ds_stats["splits"]["train"], ds_stats["splits"]["val"], ds_stats["splits"]["test"], ds_stats["total_instances"]]],
    ))

    cn.console.print()
    cn.info("Startuję trening — Ultralytics pokaże progress per epoka...")
    cn.console.print()

    model = YOLO(args.model)
    t0 = time.perf_counter()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="results/runs",
        name=args.name,
        exist_ok=False,
    )
    duration_s = time.perf_counter() - t0

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("results/runs") / args.name
    best_pt = run_dir / "weights" / "best.pt"
    target_pt = Path("models/trained") / f"{args.name}_best.pt"
    if best_pt.exists():
        target_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, target_pt)
        cn.success(f"Skopiowano best.pt → {target_pt}")

    metrics = getattr(results, "results_dict", {}) or {}
    train_stats = {
        "epochs": args.epochs,
        "best_epoch": metrics.get("best_epoch", "?"),
        "duration": _fmt_duration(duration_s),
        "train_loss": _fmt(metrics.get("train/box_loss")),
        "val_loss": _fmt(metrics.get("val/box_loss")),
        "best_map50": _fmt(metrics.get("metrics/mAP50(B)")),
        "best_map5095": _fmt(metrics.get("metrics/mAP50-95(B)")),
    }

    cn.console.print()
    cn.console.print(cn.stats_table("Wyniki treningu", [(k, v) for k, v in train_stats.items()]))

    report_path = Path("results") / f"{args.name}_train_report.txt"
    write_report(
        output_path=report_path,
        title=f"YOLO Sign Detection — Training Report ({args.name})",
        metadata={
            "Model": args.model,
            "Epoki": args.epochs,
            "imgsz": args.imgsz,
            "Batch": args.batch,
            "Hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "Run dir": str(run_dir),
        },
        sections=[
            ("DATASET", format_dataset_section(ds_stats)),
            ("TRENING", format_training_section(train_stats)),
        ],
    )
    cn.success(f"Raport: {report_path}")


def _fmt(x: object) -> object:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "?"


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


if __name__ == "__main__":
    main()
