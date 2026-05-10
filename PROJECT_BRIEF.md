# Project Brief: YOLO Sign Detection
**Dokument do recenzji przez zewnętrznego specjalistę (computer vision / ML)**

---

## 0. Kontekst dla recenzenta

Cześć! To krótki opis projektu, który robię na zajęcia uniwersyteckie (przedmiot związany z rozpoznawaniem obrazu). Chciał(a)bym prosić Cię o krytyczne spojrzenie — czy podejście ma sens, czego brakuje, gdzie są ryzyka. Konkretne pytania na końcu dokumentu (sekcja 9). Nie musisz odpowiadać na wszystkie — wystarczy te, do których masz mocne zdanie.

**Skala projektu**: ~30-50h pracy, jedna osoba, deadline koniec semestru.
**Cel akademicki**: pokazać że rozumiem pipeline detekcji obiektów end-to-end.
**Cel praktyczny**: działający demo + prezentacja.

---

## 1. Problem i motywacja

Detekcja znaków drogowych i wybranych elementów infrastruktury na zdjęciach pochodzących z **Google Street View / Google Earth**. Dataset budowany ręcznie przez zbieranie screenów z różnych lokalizacji (głównie Polska).

**Dlaczego to jest ciekawe**:
- Dane Street View mają specyficzne cechy (rozdzielczość, kąt kamery, kompresja, czasem rozmycie twarzy/tablic), które różnią się od typowych datasetów (KITTI, BDD100K).
- Pozwala zbadać generalizację pretrenowanych modeli na "real-world but unconventional" źródło.
- Skala daje się zamknąć w semestrze.

**Dlaczego to NIE jest super oryginalne**: detekcja znaków to klasyczny benchmark CV. Wartość projektu leży w wykonaniu (czysty pipeline, dobra ewaluacja, świadoma analiza błędów), nie w nowatorstwie metody.

---

## 2. Zakres

### W zakresie (MVP)
- Wybór 3-5 klas znaków drogowych (kandydaci: stop, ograniczenie prędkości, ustąp pierwszeństwa, przejście dla pieszych, sygnalizacja)
- Dataset: 80-150 obrazów własnoręcznie zebranych i zanotowanych
- Finetuning pretrenowanego YOLOv8n na tym datasetcie
- Ewaluacja: mAP@0.5, mAP@0.5:0.95, confusion matrix, jakościowa analiza błędów
- Demo: skrypt CLI + ewentualnie minimalny Streamlit/Gradio

### Poza zakresem (świadomie)
- Klasyfikacja wszystkich znaków PL (zbyt szeroka klasoznaczność)
- Detekcja w czasie rzeczywistym z wideo
- Custom architektura — bierzemy gotowy YOLO
- Deployment produkcyjny

---

## 3. Plan techniczny

### Architektura
- **Model bazowy**: Ultralytics YOLOv8n (nano, 3.2M parametrów) — wybór podyktowany szybkością prototypowania na CPU/słabym GPU. Plan B: YOLOv8s jeśli starczy czasu.
- **Strategia treningu**: transfer learning — ładujemy wagi pretrenowane na COCO i finetunujemy ostatnie warstwy + całą sieć z niskim learning rate.
- **Input size**: 640x640 (default Ultralytics).
- **Augmentacje**: domyślne YOLO (mosaic, flip, HSV, scaling); rozważam wyłączenie mosaic w ostatnich epokach.

### Pipeline
```
Street View screenshot
        ↓
   resize / pad do 640x640
        ↓
       YOLO
        ↓
boxes + class_ids + confidences
        ↓
NMS (zrobione przez YOLO)
        ↓
wizualizacja / metryki
```

### Stack
Python 3.10+ · `ultralytics` · PyTorch · OpenCV · LabelImg lub Roboflow do anotacji · Jupyter do eksperymentów.

---

## 4. Dataset

### Źródło
Ręczne screeny z Google Street View i Google Earth, różne lokalizacje (planowo: Poznań + 2-3 inne miasta PL dla różnorodności).

### Wielkość docelowa
- 80-150 obrazów łącznie
- Min. ~20 instancji na klasę (świadom, że to mało; alternatywa to suplementacja datasetem GTSRB lub Mapillary)

### Podział
70 / 20 / 10 (train / val / test) — randomized split, ale ze świadomą stratyfikacją po klasach.

### Anotacja
- Format YOLO (`class_id x_c y_c w h`, znormalizowane)
- Tool: LabelImg (offline) lub Roboflow free tier
- Estymowany czas: 2-4h dla 100 obrazów

### Ryzyka datasetowe
- **Mała wielkość** → ryzyko overfittingu, ograniczona generalizacja
- **Bias geograficzny** → znaki PL mogą się różnić od innych krajów (kolory, kształty), model nie będzie generalizował poza PL
- **Long tail** → niektóre klasy mogą mieć dramatycznie mniej przykładów
- **TOS Google** → screenów nie publikujemy, używamy lokalnie do celów edukacyjnych

---

## 5. Plan ewaluacji

### Metryki ilościowe
- **mAP@0.5** — główna metryka, standardowa dla detekcji
- **mAP@0.5:0.95** — restrykcyjna, COCO-style
- **Precision / Recall** per klasa
- **Confusion matrix** — żeby zobaczyć które klasy się mylą

### Metryki jakościowe
- 5-10 obrazów spoza datasetu — sanity check generalizacji
- Manualne przeglądanie błędów: false positives, false negatives, mylone klasy
- Identyfikacja systematycznych failure modes (mały rozmiar znaku, częściowe zasłonięcie, dziwny kąt, słabe oświetlenie)

### Baseline
Pretrained YOLOv8n bez finetuningu (rozpoznaje "stop sign" w COCO) — porównanie pokazuje wartość finetuningu na własnych klasach.

---

## 6. Anticipated challenges

| Ryzyko | Prawdopodobieństwo | Wpływ | Mitigation |
|---|---|---|---|
| Za mały dataset → overfit | wysokie | wysoki | Augmentacja, ew. suplementacja zewnętrznym datasetem |
| Niezbalansowane klasy | średnie | średni | Świadoma kuracja przy zbieraniu, class weights w lossie |
| Słaba generalizacja na nowe lokalizacje | wysokie | średni | Jawne uznanie w wynikach, świadomy zakres |
| Brak GPU → długi trening | średnie | średni | Google Colab (T4 free tier) jako fallback |
| Anotacja błędów / niespójności | wysokie | wysoki | Reguły anotacji spisane przed startem, self-review |
| Ambiguity klasy (np. czy znak częściowo zasłonięty to instancja?) | wysokie | średni | Definicja w guidelines: "≥50% widoczne = anotujemy" |

---

## 7. Timeline (orientacyjny)

| Faza | Czas | Deliverable |
|---|---|---|
| Setup | 1-2h | Działający `yolo predict` na sample |
| Dane | 2-4h | 100+ obrazów w `data/raw/` |
| Anotacja | 3-6h | Pełny `data/annotations/` + dataset.yaml |
| Trening | 1-3h | `best.pt` + wykresy treningu |
| Ewaluacja | 1-2h | Metryki, confusion matrix, analiza błędów |
| Prezentacja | 2-3h | Slajdy + demo |
| **Total** | **10-20h** | |

Bufor czasowy ~30% na nieprzewidziane.

---

## 8. Decyzje techniczne — uzasadnienia

| Decyzja | Alternatywa | Dlaczego ta a nie tamta |
|---|---|---|
| YOLOv8 (Ultralytics) | YOLOv11, YOLOv5, DETR, Faster R-CNN | Najlepsza dokumentacja, niski próg wejścia, aktywna społeczność, łatwy CLI. Wydajność wystarczająca dla projektu studenckiego. |
| YOLOv8**n** | v8s/m/l/x | Szybki trening na słabym sprzęcie, wystarczy dla MVP. Plan B: większy model jeśli starczy czasu. |
| Finetuning całej sieci | Tylko head | Mały dataset → finetuning całości z niskim LR powinien działać lepiej dla *nowych* klas (znaki PL ≠ COCO). |
| Format YOLO (txt) | COCO JSON, Pascal VOC | Natywny dla Ultralytics, brak konwersji. |
| Manual collection | GTSRB / Mapillary | Świadoma decyzja: chcę przetestować na danych Street View, nie na "studyjnych" zdjęciach znaków. Może hybryda jeśli czas pozwoli. |

---

## 9. Pytania do recenzenta

Najbardziej zależy mi na opinii w tych punktach:

1. **Wielkość datasetu**: Czy 80-150 obrazów ma w ogóle sens dla 3-5 klas, czy to skazane na overfit? Czy hybryda (mały custom + większy zewnętrzny GTSRB/Mapillary jako pretraining-stage-2) byłaby mądrzejsza?

2. **Wybór YOLOv8 vs nowsze (v11, v12)**: Czy warto się męczyć z nowszymi, czy v8 to nadal solidny wybór dla projektu edukacyjnego w 2026?

3. **Strategia ewaluacji**: Czy mAP@0.5 + mAP@0.5:0.95 + confusion matrix to wystarczające minimum? Co jeszcze powinienem mierzyć, żeby projekt wyglądał profesjonalnie?

4. **Anotacja**: LabelImg vs Roboflow vs CVAT — Twoja rekomendacja dla jednoosobowego projektu studenckiego?

5. **Failure analysis**: Jak ustrukturyzować analizę błędów, żeby nie była tylko "popatrz na obrazki"? Jakieś konkretne kategoryzacje błędów, które zwykle robisz?

6. **Generalizacja**: Czy zaplanować jawny "out-of-distribution test" (np. screeny z innego kraju), czy to przesada na poziomie projektu studenckiego?

7. **Co bym zrobił/a, gdybym to robił/a Ty?** Co jest "obviously missing" z perspektywy kogoś z doświadczeniem? Czego studenci typowo zapominają?

8. **Red flags**: Czy widzisz coś, co prawdopodobnie się rozwali? Pułapki specyficzne dla Street View jako źródła danych?

---

## 10. Co już mam gotowe

- [x] Plan projektu (CLAUDE.md, TODO.md)
- [x] Struktura repo (skeleton)
- [x] requirements.txt
- [x] Starter script `src/detect.py`
- [ ] Wszystko poniżej — patrz TODO.md

---

*Dziękuję za poświęcony czas. Każdy feedback — od jednej linijki po szczegółową krytykę — jest mile widziany.*
