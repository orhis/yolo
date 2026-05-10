# TODO — YOLO Sign Detection

## Faza 1: Setup (≈1-2h)
- [ ] Stworzyć venv i zainstalować `ultralytics`, `opencv-python`, `pillow`, `jupyter`
- [ ] Wygenerować `requirements.txt` (`pip freeze > requirements.txt`)
- [ ] Sprawdzić czy CUDA działa: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Pobrać pretrained: `yolov8n.pt`
- [ ] Sanity check — inference na losowym obrazku, zobaczyć czy rysuje boxy
- [ ] `git init` i pierwszy commit

## Faza 2: Dane (≈2-4h)
- [ ] Zdefiniować listę klas (np. stop, ograniczenie prędkości, ustąp pierwszeństwa, sygnalizacja, przejście dla pieszych — albo szerzej, albo węziej)
- [ ] Zebrać 80-150 screenów ze Street View / Google Earth
  - [ ] Różne lokalizacje (Polska — Poznań, Warszawa, mniejsze miasta)
  - [ ] Różne kąty / odległości / oświetlenie
  - [ ] Każda klasa minimum ~20 instancji, najlepiej więcej
- [ ] Zapisać surowe obrazki w `data/raw/` (jednolite nazwy: `img_001.jpg`, ...)
- [ ] Sprawdzić rozdzielczości — przeskalować do max 1280px po dłuższym boku jak są ogromne

## Faza 3: Anotacja (≈3-6h, najmniej zabawna)
- [ ] Wybrać tool: **LabelImg** (offline, prosty) albo **Roboflow** (online, ale daje augmentację za darmo)
- [ ] Anotować obrazki — każdy znak w bounding box
- [ ] Eksport w formacie YOLO (jeden .txt na obraz)
- [ ] Podział train/val/test = 70/20/10 (skrypt `scripts/prepare_dataset.py`)
- [ ] Stworzyć `data/dataset.yaml`:
  ```yaml
  path: ../data
  train: splits/train/images
  val: splits/val/images
  test: splits/test/images
  names:
    0: stop
    1: speed_limit
    # ...
  ```

## Faza 4: Trening (≈1-3h, zależnie od sprzętu)
- [ ] Pierwszy trening: `yolov8n.pt`, 50 epok, `imgsz=640`
- [ ] Obejrzeć wykresy w `runs/detect/train/results.png` (loss, mAP)
- [ ] Sprawdzić overfitting: val_loss vs train_loss
- [ ] Zapisać `best.pt` do `models/trained/`
- [ ] (opcja) Powtórzyć z `yolov8s.pt` lub większą liczbą epok dla porównania

## Faza 5: Ewaluacja (≈1-2h)
- [ ] `yolo val` na test set → zapisać metryki
- [ ] Confusion matrix, PR curve (Ultralytics generuje sam)
- [ ] Inference na 5-10 nowych screenach (poza datasetem) — sprawdzić generalizację
- [ ] Analiza błędów: kiedy model fail-uje? (mały znak, częściowo zasłonięty, dziwny kąt?)
- [ ] Zapisać przykłady udane i nieudane do `results/predictions/`

## Faza 6: Prezentacja (≈2-3h)
- [ ] Slajdy: motywacja → dataset → metoda (YOLO architektura w skrócie) → wyniki → ograniczenia → wnioski
- [ ] Demo: na żywo lub nagrane (5-10s gif z detekcjami)
- [ ] Przykłady: 2-3 udane + 1-2 nieudane (z analizą czemu)
- [ ] README.md w repo opisujący jak odpalić

## Bonus / nice-to-have
- [ ] Streamlit / Gradio app — wrzucasz obrazek, dostajesz boxy
- [ ] Porównanie YOLOv8 vs YOLOv11 na tym samym datasetcie
- [ ] Test na fragmencie wideo (przejazd Street View)
- [ ] Augmentacja danych (Albumentations) i porównanie z/bez
