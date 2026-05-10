"""Baseline inference — pretrained YOLOv8n na wszystkich obrazach z data/raw/.

Cel: wstępna ocena datasetu PRZED anotacją.
- Pretrained YOLO zna 'stop sign' z COCO → testuje klasę stop bez treningu
- Brak detekcji w obrazku ≈ thumbnail / catalog / dziwne ujęcie
- Detekcja person/car/traffic light → realna scena uliczna

Output:
- results/predictions/baseline/img_NNN.jpg — adnotowane obrazki
- results/baseline_report.txt — pełny raport per-image + statystyki
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from src import console as cn
from src.report import write_report


def main() -> None:
    cn.phase("Baseline inference", "Pretrained YOLOv8n na wszystkich obrazach z data/raw/")

    raw_dir = Path("data/raw")
    out_dir = (Path.cwd() / "results" / "predictions" / "baseline").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(raw_dir.glob("img_*.jpg"))
    cn.kv("Obrazów do przetworzenia", len(images))
    cn.kv("Output dir", out_dir.relative_to(Path.cwd()))

    if not images:
        cn.error("Brak obrazów w data/raw/. Najpierw uruchom scripts/import_images.py.")
        raise SystemExit(1)

    class_hints = _load_manifest(raw_dir / "_manifest.csv")

    with cn.step("Ładuję pretrained YOLOv8n"):
        model = YOLO("models/pretrained/yolov8n.pt")

    per_image: list[dict] = []
    coco_counter: Counter[str] = Counter()
    by_hint: dict[str, dict[str, int]] = defaultdict(lambda: {"with": 0, "without": 0})

    cn.console.print()
    with cn.make_progress() as prog:
        task = prog.add_task("Inference", total=len(images))
        for img_path in images:
            results = model.predict(source=str(img_path), verbose=False, save=False)
            r = results[0]
            hint = class_hints.get(img_path.name, "?")

            detections: list[tuple[str, float]] = []
            if r.boxes is not None and len(r.boxes) > 0:
                for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = r.names[int(cls_id)]
                    detections.append((name, float(conf)))
                    coco_counter[name] += 1
                by_hint[hint]["with"] += 1
            else:
                by_hint[hint]["without"] += 1

            annotated = r.plot()
            cv2.imwrite(str(out_dir / img_path.name), annotated)

            per_image.append({
                "name": img_path.name,
                "hint": hint,
                "detections": detections,
            })
            prog.advance(task)

    _print_summary(per_image, coco_counter, by_hint)
    _write_report(per_image, coco_counter, by_hint, out_dir)


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["new_name"]: row["note"] for row in csv.DictReader(f) if row["new_name"]}


def _print_summary(per_image, coco_counter, by_hint) -> None:
    total = len(per_image)
    with_det = sum(1 for r in per_image if r["detections"])
    cn.console.print()
    cn.console.print(cn.stats_table("Podsumowanie", [
        ("Obrazów przetworzonych", total),
        ("Z jakąkolwiek detekcją", f"{with_det} ({with_det / total * 100:.0f}%)"),
        ("Łączna liczba detekcji", sum(coco_counter.values())),
        ("Unikalnych klas COCO", len(coco_counter)),
    ]))

    cn.console.print()
    cn.console.print(cn.metrics_table(
        "Top wykrytych klas COCO",
        ["Klasa COCO", "Detekcje"],
        [[name, count] for name, count in coco_counter.most_common(10)] or [["(brak detekcji)", 0]],
    ))

    cn.console.print()
    cn.console.print(cn.metrics_table(
        "Per nasza klasa (z manifestu)",
        ["Klasa", "Z detekcją", "Bez detekcji", "Razem"],
        [[cls, v["with"], v["without"], v["with"] + v["without"]] for cls, v in sorted(by_hint.items())],
    ))

    stop_imgs = [r["name"] for r in per_image if any(d[0] == "stop sign" for d in r["detections"])]
    cn.console.print()
    if stop_imgs:
        cn.info(f"Pretrained YOLO znalazło 'stop sign' w [bold]{len(stop_imgs)}[/bold] obrazach z {sum(1 for r in per_image if r['hint'] == 'stop')} oznaczonych jako stop")
    else:
        cn.warning("Pretrained YOLO nie znalazło 'stop sign' w żadnym obrazie — to mocny sygnał że zdjęcia w klasie stop są problematyczne (thumbnaile / nieostre)")


def _write_report(per_image, coco_counter, by_hint, out_dir: Path) -> None:
    total = len(per_image)
    with_det = sum(1 for r in per_image if r["detections"])

    sections: list[tuple[str, str]] = []

    overview = "\n".join([
        f"Obrazów przetworzonych:    {total}",
        f"Z jakąkolwiek detekcją:    {with_det} ({with_det / total * 100:.0f}%)",
        f"Bez detekcji:              {total - with_det}",
        f"Łączna liczba detekcji:    {sum(coco_counter.values())}",
        f"Unikalnych klas COCO:      {len(coco_counter)}",
    ])
    sections.append(("PRZEGLĄD", overview))

    classes_block = "\n".join(f"  {name:<25} {count}" for name, count in coco_counter.most_common())
    sections.append(("WYKRYTE KLASY (COCO)", classes_block or "  (brak)"))

    hint_lines = []
    for cls, v in sorted(by_hint.items()):
        total_cls = v["with"] + v["without"]
        pct = v["with"] / total_cls * 100 if total_cls else 0
        hint_lines.append(f"  {cls:<25} z detekcją: {v['with']:>3} / {total_cls:<3}  ({pct:>3.0f}%)")
    sections.append(("PER NASZA KLASA", "\n".join(hint_lines)))

    stop_imgs = [r["name"] for r in per_image if any(d[0] == "stop sign" for d in r["detections"])]
    stop_block = (
        f"Pretrained YOLOv8n zna 'stop sign' z COCO. Wykrył w {len(stop_imgs)} obrazach:\n"
        + ("\n".join(f"  - {n}" for n in stop_imgs) if stop_imgs else "  (brak)")
    )
    sections.append(("STOP — BASELINE (COCO pretrained)", stop_block))

    no_det = [r["name"] + f" [{r['hint']}]" for r in per_image if not r["detections"]]
    no_det_block = (
        f"YOLO nie wykrył NICZEGO w {len(no_det)} obrazach. To prawdopodobnie thumbnaile,\n"
        f"crop'y na białym tle, lub bardzo nietypowe ujęcia.\n\n"
        + "\n".join(f"  - {n}" for n in no_det)
    )
    sections.append(("OBRAZY BEZ DETEKCJI (kandydaci do cull)", no_det_block))

    detail_lines = []
    for r in per_image:
        dets_str = ", ".join(f"{n}({c:.2f})" for n, c in r["detections"]) if r["detections"] else "(brak)"
        detail_lines.append(f"  {r['name']:<14} [{r['hint']:<22}] {dets_str}")
    sections.append(("PER OBRAZ", "\n".join(detail_lines)))

    report_path = Path("results/baseline_report.txt")
    write_report(
        output_path=report_path,
        title="YOLO Sign Detection — Baseline Pretrained Inference",
        metadata={
            "Model": "YOLOv8n (pretrained, COCO)",
            "Liczba obrazów": total,
            "Output annotated": str(out_dir.relative_to(Path.cwd())),
        },
        sections=sections,
    )

    cn.console.print()
    cn.success(f"Raport: {report_path}")
    cn.info(f"Annotated images: {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
