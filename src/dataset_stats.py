"""Liczenie statystyk datasetu YOLO (ile obrazów, ile instancji per klasa)."""

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def collect_dataset_stats(dataset_yaml: Path) -> dict[str, Any]:
    """Czyta dataset.yaml i liczy: total_images, splits, total_instances, per_class."""
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    base = (dataset_yaml.parent / cfg.get("path", ".")).resolve()
    names: dict[int, str] = cfg["names"]

    splits: dict[str, int] = {}
    instances: dict[str, Counter] = {split: Counter() for split in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel is None:
            continue
        img_dir = (base / rel).resolve()
        label_dir = Path(str(img_dir).replace("images", "labels", 1))
        if not img_dir.exists():
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        splits[split] = len(images)

        for img_path in images:
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                instances[split][cls_id] += 1

    per_class = []
    for cls_id, name in sorted(names.items()):
        train_n = instances["train"][cls_id]
        val_n = instances["val"][cls_id]
        test_n = instances["test"][cls_id]
        per_class.append({
            "name": name,
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "total": train_n + val_n + test_n,
        })

    return {
        "total_images": sum(splits.values()),
        "splits": splits,
        "total_instances": sum(pc["total"] for pc in per_class),
        "per_class": per_class,
        "class_names": [names[i] for i in sorted(names.keys())],
    }
