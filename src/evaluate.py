"""Ewaluacja wytrenowanego modelu na test set + generowanie raportu (Faza 5).

Użycie:
    python -m src.evaluate                                  # default best
    python -m src.evaluate --weights models/trained/signs_best.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src import console as cn
from src.dataset_stats import collect_dataset_stats
from src.report import (
    format_confusion_matrix,
    format_dataset_section,
    format_eval_section,
    format_inference_section,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ewaluacja modelu YOLO na test set")
    p.add_argument("--weights", default="models/trained/signs_best.pt")
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--name", default="signs_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cn.phase("Faza 5 — Ewaluacja", f"Weights: {args.weights} • Split: {args.split}")

    weights = Path(args.weights)
    if not weights.exists():
        cn.error(f"Brak wag: {weights}. Najpierw odpal trening (src.train).")
        raise SystemExit(1)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        cn.error(f"Brak {data_yaml}.")
        raise SystemExit(1)

    cn.info("Środowisko:")
    cn.kv("Hardware", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    cn.kv("Weights size", f"{weights.stat().st_size / 1e6:.1f} MB")

    with cn.step("Wczytuję statystyki datasetu"):
        ds_stats = collect_dataset_stats(data_yaml)

    with cn.step(f"Ładuję model {weights.name}"):
        model = YOLO(str(weights))

    cn.console.print()
    cn.info(f"Walidacja na split={args.split} — Ultralytics policzy mAP, PR curve, confusion matrix...")
    cn.console.print()

    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        split=args.split,
        project="results/runs",
        name=args.name,
        exist_ok=False,
        verbose=False,
    )
    duration_s = time.perf_counter() - t0

    box = metrics.box
    class_names = ds_stats["class_names"]
    per_class_eval = []
    for i, name in enumerate(class_names):
        per_class_eval.append({
            "name": name,
            "precision": float(box.p[i]) if i < len(box.p) else 0.0,
            "recall": float(box.r[i]) if i < len(box.r) else 0.0,
            "map50": float(box.ap50[i]) if i < len(box.ap50) else 0.0,
            "map5095": float(box.ap[i]) if i < len(box.ap) else 0.0,
        })
    eval_stats = {
        "per_class": per_class_eval,
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map5095": float(box.map),
        },
    }

    cn.console.print(cn.metrics_table(
        "Ewaluacja",
        ["Klasa", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        [[pc["name"], f"{pc['precision']:.3f}", f"{pc['recall']:.3f}", f"{pc['map50']:.3f}", f"{pc['map5095']:.3f}"] for pc in per_class_eval]
        + [["WSZYSTKIE", f"{eval_stats['overall']['precision']:.3f}", f"{eval_stats['overall']['recall']:.3f}", f"{eval_stats['overall']['map50']:.3f}", f"{eval_stats['overall']['map5095']:.3f}"]],
    ))

    confusion = _extract_confusion(metrics, class_names)

    speed = metrics.speed if hasattr(metrics, "speed") else {}
    avg_ms = sum(speed.values()) if speed else 0.0
    inference_stats = {
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms if avg_ms > 0 else 0,
        "resolution": f"{args.imgsz}x{args.imgsz}",
        "num_images": ds_stats["splits"].get(args.split, "?"),
        "num_detections": "?",
    }

    cn.console.print()
    cn.console.print(cn.stats_table("Wydajność", [
        ("Avg time (ms/img)", f"{avg_ms:.1f}"),
        ("FPS", f"{1000.0 / avg_ms:.1f}" if avg_ms > 0 else "?"),
        ("Eval duration", f"{duration_s:.1f} s"),
    ]))

    report_path = Path("results") / f"{args.name}_report.txt"
    write_report(
        output_path=report_path,
        title=f"YOLO Sign Detection — Evaluation Report ({args.name})",
        metadata={
            "Weights": str(weights),
            "Split": args.split,
            "imgsz": args.imgsz,
            "Hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        sections=[
            ("DATASET", format_dataset_section(ds_stats)),
            ("EWALUACJA", format_eval_section(eval_stats)),
            *([("CONFUSION MATRIX", format_confusion_matrix(confusion))] if confusion else []),
            ("WYDAJNOŚĆ INFERENCE", format_inference_section(inference_stats)),
        ],
    )
    cn.console.print()
    cn.success(f"Raport: {report_path}")


def _extract_confusion(metrics: object, class_names: list[str]) -> dict | None:
    """Próbuje wyciągnąć confusion matrix z wyników Ultralytics."""
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None:
        return None
    matrix = getattr(cm, "matrix", None)
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=int)
    classes = list(class_names) + ["background"]
    if arr.shape[0] != len(classes):
        return None
    return {"classes": classes, "data": arr.tolist()}


if __name__ == "__main__":
    main()
