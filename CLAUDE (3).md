# YOLO Sign Detection — projekt na zajęcia

## Cel
Detekcja obiektów (głównie znaków drogowych i elementów infrastruktury) na zdjęciach z Google Earth / Street View przy użyciu modelu YOLO. Wykorzystujemy pretrained model + finetuning na własnym, małym datasetcie.

## Stack
- **Python 3.10+**
- **Ultralytics YOLO** (v8 lub v11) — `pip install ultralytics`
- **PyTorch** — instaluje się razem z ultralytics
- **OpenCV** + **Pillow** — obróbka obrazu
- **Jupyter** — eksperymenty i wizualizacje
- **LabelImg** lub **Roboflow** — anotacja (do wyboru)

## Struktura projektu
```
yolo-signs/
├── CLAUDE.md              # ten plik
├── TODO.md                # lista zadań
├── README.md              # opis projektu (do oddania)
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               # surowe screeny ze Street View
│   ├── processed/         # przeskalowane do 640x640
│   ├── annotations/       # etykiety YOLO (.txt)
│   ├── splits/            # train / val / test
│   └── dataset.yaml       # config datasetu dla YOLO
├── models/
│   ├── pretrained/        # yolov8n.pt itd.
│   └── trained/           # nasze wagi (best.pt)
├── notebooks/
│   ├── 01_explore.ipynb       # eksploracja danych
│   ├── 02_pretrained_demo.ipynb  # test pretrained na naszych obrazkach
│   ├── 03_train.ipynb         # finetuning
│   └── 04_evaluate.ipynb      # mAP, confusion matrix, błędy
├── src/
│   ├── __init__.py
│   ├── detect.py          # inference na obrazie/katalogu
│   ├── train.py           # trening
│   ├── evaluate.py        # ewaluacja
│   └── utils.py           # helpery (rysowanie boxów, konwersje)
├── scripts/
│   └── prepare_dataset.py # podział train/val/test
├── results/
│   ├── predictions/       # obrazki z narysowanymi boxami
│   └── runs/              # output treningu (Ultralytics tworzy sam)
└── presentation/
    └── slides.md
```

## Konwencje
- **Format anotacji**: YOLO (`class_id x_center y_center width height`, znormalizowane 0-1)
- **Klasy**: zdefiniowane w `data/dataset.yaml`
- **Bazowy model**: `yolov8n.pt` (nano — szybki, do prototypowania), później `yolov8s.pt` jak będzie czas
- **Rozmiar wejścia**: 640x640
- **Naming**: snake_case

## Komendy
```bash
# Setup
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Inference z pretrained (sanity check)
yolo predict model=yolov8n.pt source=data/raw/test.jpg

# Trening
yolo train model=yolov8n.pt data=data/dataset.yaml epochs=50 imgsz=640

# Ewaluacja
yolo val model=models/trained/best.pt data=data/dataset.yaml
```

## Aktualny status
- [ ] Setup
- [ ] Dane
- [ ] Anotacja
- [ ] Trening
- [ ] Ewaluacja
- [ ] Prezentacja

## Decyzje projektowe
*(tu wpisuj kluczowe decyzje, żeby pamiętać dlaczego coś zrobiliśmy tak a nie inaczej)*

- Wersja YOLO: **YOLOv8n** (Ultralytics — najlepiej udokumentowana, dużo tutoriali)
- Klasy do detekcji: **5 klas** — `stop` (B-20), `yield` (A-7), `speed_limit` (B-33), `pedestrian_crossing` (D-6), `no_entry` (B-2). Pełen przewodnik anotacji: [CLASSES.md](CLASSES.md)
- Źródło danych: Google Earth / Street View (screeny ręczne, pełne sceny — nie close-upy)

## Pułapki i uwagi
- Google ma TOS — screeny do projektu studenckiego są zwykle OK, ale **nie publikujemy datasetu**, tylko używamy lokalnie
- Anotacja zajmuje najwięcej czasu — nie oszukujmy się, 100 obrazków = jakieś 2-4h roboty
- CPU starczy do prototypu, ale trening na GPU jest 10-50x szybszy. Jak nie ma GPU — Google Colab (free tier z T4) ratuje sytuację
- YOLO format != COCO format != Pascal VOC — uważać przy konwersjach
