"""Generator raportu TXT z wynikami treningu i ewaluacji.

Format: prosty plain-text z sekcjami i tabelami ASCII. Cel — czytelność
w terminalu i w edytorze, łatwe dołączenie do prezentacji.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

WIDTH = 80
DIVIDER = "=" * WIDTH
SUBDIV = "-" * WIDTH


def _kv_block(rows: Sequence[tuple[str, object]], width: int = 28) -> str:
    return "\n".join(f"{k:<{width}}{v}" for k, v in rows)


def _ascii_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    widths: Sequence[int] | None = None,
) -> str:
    if widths is None:
        cols = list(zip(*([headers] + [list(r) for r in rows])))
        widths = [max(len(str(c)) for c in col) + 2 for col in cols]

    def fmt_row(row: Sequence[object]) -> str:
        return "".join(f"{str(c):<{w}}" for c, w in zip(row, widths))

    lines = [fmt_row(headers), "-" * sum(widths), *[fmt_row(r) for r in rows]]
    return "\n".join(lines)


def _section(title: str, body: str) -> str:
    return f"{title}\n{SUBDIV}\n{body}"


def format_dataset_section(stats: dict[str, Any]) -> str:
    """stats: total_images, splits{train,val,test}, total_instances, per_class[{name,train,val,test,total}]."""
    lines = [
        _kv_block([
            ("Łączna liczba obrazów:", stats.get("total_images", "?")),
            ("  - train (70%):", stats.get("splits", {}).get("train", "?")),
            ("  - val (20%):", stats.get("splits", {}).get("val", "?")),
            ("  - test (10%):", stats.get("splits", {}).get("test", "?")),
        ]),
        "",
        f"Łączna liczba instancji:    {stats.get('total_instances', '?')}",
    ]
    per_class = stats.get("per_class", [])
    if per_class:
        rows = [[pc["name"], pc.get("train", 0), pc.get("val", 0), pc.get("test", 0), pc.get("total", 0)] for pc in per_class]
        totals = ["RAZEM", sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows)]
        rows.append(totals)
        lines += [
            "",
            "Rozkład klas:",
            _ascii_table(["Klasa", "Train", "Val", "Test", "Razem"], rows, [25, 8, 8, 8, 8]),
        ]
    return "\n".join(lines)


def format_training_section(stats: dict[str, Any]) -> str:
    """stats: epochs, best_epoch, duration, train_loss, val_loss, best_map50, best_map5095."""
    return _kv_block([
        ("Liczba epok:", stats.get("epochs", "?")),
        ("Best epoch:", stats.get("best_epoch", "?")),
        ("Czas treningu:", stats.get("duration", "?")),
        ("Final train loss:", stats.get("train_loss", "?")),
        ("Final val loss:", stats.get("val_loss", "?")),
        ("Best mAP@0.5:", stats.get("best_map50", "?")),
        ("Best mAP@0.5:0.95:", stats.get("best_map5095", "?")),
    ])


def format_eval_section(stats: dict[str, Any]) -> str:
    """stats: per_class[{name,precision,recall,map50,map5095}], overall{precision,recall,map50,map5095}."""
    per_class = stats.get("per_class", [])
    if not per_class:
        return "(brak danych)"

    def fmt(x: object) -> str:
        return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

    rows = [[pc["name"], fmt(pc.get("precision")), fmt(pc.get("recall")), fmt(pc.get("map50")), fmt(pc.get("map5095"))] for pc in per_class]
    if (overall := stats.get("overall")):
        rows.append(["WSZYSTKIE", fmt(overall.get("precision")), fmt(overall.get("recall")), fmt(overall.get("map50")), fmt(overall.get("map5095"))])
    return _ascii_table(["Klasa", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"], rows, [25, 12, 10, 11, 16])


def format_confusion_matrix(matrix: dict[str, Any]) -> str:
    """matrix: classes (list), data (NxN ints, rows=true, cols=pred), include 'background' as last."""
    classes = matrix.get("classes", [])
    data = matrix.get("data", [])
    if not classes or not data:
        return "(brak danych)"

    headers = ["true \\ pred"] + [c[:8] for c in classes]
    rows = [[classes[i][:20]] + list(data[i]) for i in range(len(classes))]
    widths = [22] + [10] * len(classes)
    return _ascii_table(headers, rows, widths)


def format_inference_section(stats: dict[str, Any]) -> str:
    """stats: avg_ms, fps, resolution, num_images, num_detections."""
    return _kv_block([
        ("Średni czas na obraz:", f"{stats['avg_ms']:.1f} ms" if "avg_ms" in stats else "?"),
        ("FPS:", f"{stats['fps']:.1f}" if "fps" in stats else "?"),
        ("Rozdzielczość:", stats.get("resolution", "?")),
        ("Liczba obrazów:", stats.get("num_images", "?")),
        ("Liczba detekcji:", stats.get("num_detections", "?")),
    ])


def write_report(
    output_path: Path,
    title: str,
    metadata: dict[str, Any] | None = None,
    sections: Sequence[tuple[str, str]] = (),
) -> str:
    """Składa raport i zapisuje do pliku. Zwraca treść."""
    parts: list[str] = [DIVIDER, f"  {title}", DIVIDER]

    base_meta = {"Wygenerowano": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    base_meta.update(metadata or {})
    parts += ["", _kv_block([(f"{k}:", v) for k, v in base_meta.items()])]

    for section_title, body in sections:
        parts += ["", _section(section_title, body)]

    parts += ["", DIVIDER, ""]
    report = "\n".join(parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
