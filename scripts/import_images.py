"""Import obrazów z dowolnego folderu do data/raw/ — standardyzacja.

Konwertuje wszystkie obsługiwane formaty (jpg/jpeg/png/webp/avif/bmp) na JPG,
resizuje do max 1280px po dłuższym boku, naprawia orientację z EXIF,
przemianowuje na img_NNN.jpg. Numeracja kontynuuje istniejące pliki.

Manifest (mapping stara nazwa → nowa) ląduje w data/raw/_manifest.csv.

Użycie:
    python scripts/import_images.py "C:\\Users\\X\\Desktop\\przejscie" --note pedestrian_crossing
"""
import argparse
import csv
import sys
from pathlib import Path

import pillow_avif  # noqa: F401  - rejestruje plugin AVIF w Pillow
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import console as cn

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
MAX_DIM = 1280
JPG_QUALITY = 92


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardyzacja obrazów do data/raw/")
    p.add_argument("source", type=Path, help="Folder źródłowy z obrazami")
    p.add_argument("--dest", type=Path, default=Path("data/raw"), help="Folder docelowy")
    p.add_argument("--prefix", default="img_", help="Prefiks nazwy")
    p.add_argument("--note", default="", help="Notatka do manifestu (np. nazwa klasy)")
    return p.parse_args()


def next_id(dest: Path, prefix: str) -> int:
    nums = []
    for p in dest.glob(f"{prefix}*.jpg"):
        stem = p.stem.removeprefix(prefix)
        if stem.isdigit():
            nums.append(int(stem))
    return max(nums, default=0) + 1


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        cn.error(f"Source nie istnieje: {args.source}")
        raise SystemExit(1)

    args.dest.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)

    cn.phase("Import obrazów", f"{args.source} → {args.dest}")
    cn.kv("Znaleziono plików", len(files))
    cn.kv("Notatka", args.note or "(brak)")
    start = next_id(args.dest, args.prefix)
    cn.kv("Numeracja od", f"{args.prefix}{start:03d}.jpg")

    if not files:
        cn.warning("Brak obrazów do importu — wychodzę.")
        return

    converted = 0
    skipped = 0
    resized = 0
    rows: list[dict] = []

    with cn.make_progress() as prog:
        task = prog.add_task("Konwersja", total=len(files))
        for i, src in enumerate(files):
            new_id = start + i
            new_name = f"{args.prefix}{new_id:03d}.jpg"
            new_path = args.dest / new_name
            row = {"new_name": "", "old_name": src.name, "size": "", "note": args.note, "status": ""}

            try:
                with Image.open(src) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    w, h = img.size
                    long_side = max(w, h)
                    if long_side > MAX_DIM:
                        scale = MAX_DIM / long_side
                        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        resized += 1
                        row["status"] = "resized"
                    else:
                        row["status"] = "ok"

                    img.save(new_path, "JPEG", quality=JPG_QUALITY, optimize=True)
                    row["new_name"] = new_name
                    row["size"] = f"{img.size[0]}x{img.size[1]}"
                converted += 1
            except (OSError, UnidentifiedImageError, ValueError) as e:
                row["status"] = f"error: {e}"
                skipped += 1

            rows.append(row)
            prog.advance(task)

    manifest = args.dest / "_manifest.csv"
    write_header = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["new_name", "old_name", "size", "note", "status"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    cn.console.print()
    cn.console.print(cn.stats_table("Podsumowanie", [
        ("Znalezione pliki", len(files)),
        ("Zaimportowane", converted),
        ("Pominięte (błąd)", skipped),
        ("Przeskalowane (>1280px)", resized),
        ("Manifest", manifest),
    ]))
    cn.console.print()
    total_in_dest = len(list(args.dest.glob(f"{args.prefix}*.jpg")))
    cn.success(f"Gotowe — w {args.dest} jest teraz {total_in_dest} obrazów")
    if skipped:
        cn.warning(f"{skipped} plików pominiętych — sprawdź manifest")


if __name__ == "__main__":
    main()
