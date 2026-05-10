# Klasy do detekcji — przewodnik zbierania i anotacji

## Cel datasetu
- **~120 obrazów** ze Street View / Google Earth
- **~150-200 instancji** łącznie
- **Min. 25-40 instancji na klasę**
- Lokalizacje: głównie Polska (Poznań + 2-3 inne miasta dla różnorodności)

---

## Lista klas (5)

### `0: stop` — znak B-20 STOP
- **Wygląd:** ośmiokąt, czerwony, biały napis "STOP"
- **Gdzie szukać:** skrzyżowania bez sygnalizacji, wyjazdy z dróg podporządkowanych, wyjazdy z parkingów
- **Cel:** ~25-30 instancji
- **Uwagi:** unikalny kształt → łatwy dla modelu. Występuje rzadziej niż inne, szukaj specjalnie.

### `1: yield` — znak A-7 ustąp pierwszeństwa
- **Wygląd:** trójkąt **wierzchołkiem w dół**, biały środek, czerwona obwódka
- **Gdzie szukać:** wjazdy na ronda, skrzyżowania z drogą z pierwszeństwem
- **Cel:** ~30-40 instancji
- **Uwagi:** **uważaj** — A-1, A-2 itd. (ostrzegawcze) to też trójkąty, ale **wierzchołkiem w górę** i z piktogramem. Nie myl.

### `2: speed_limit` — znak B-33 ograniczenie prędkości
- **Wygląd:** koło, biały środek, czerwona obwódka, czarna liczba (20/30/40/50/60/70/80/90/100/120)
- **Gdzie szukać:** wjazdy do miejscowości, strefy zamieszkania, autostrady, drogi szybkiego ruchu
- **Cel:** ~40-50 instancji (najczęstszy, łatwo nazbierać)
- **Uwagi:** **wszystkie wartości liczbowe = jedna klasa.** Nie rozdzielaj na "speed_30" i "speed_50" — zabraknie instancji.

### `3: pedestrian_crossing` — znak D-6 przejście dla pieszych
- **Wygląd:** **niebieski kwadrat**, biały trójkąt w środku, czarna sylwetka pieszego na pasach
- **Gdzie szukać:** każde miasto, przy każdej zebrze
- **Cel:** ~30-40 instancji (bardzo częsty)
- **Uwagi:** **nie myl z A-17** (ostrzegawczy — żółty trójkąt). D-6 = niebieski kwadrat informacyjny, A-17 = trójkąt ostrzegawczy.

### `4: no_entry` — znak B-2 zakaz wjazdu
- **Wygląd:** koło, czerwone z białą poziomą belką w środku
- **Gdzie szukać:** ulice jednokierunkowe (od strony zakazanej), wjazdy "tylko dla autobusów", zamknięte uliczki w starówkach
- **Cel:** ~25-30 instancji
- **Uwagi:** najłatwiej trafić w centrach miast (Poznań Stare Miasto, Warszawa centrum, Kraków).

---

## Reguły anotacji

Spisane **przed** startem anotacji — żeby był spójny standard.

| Sytuacja | Decyzja |
|---|---|
| Widoczność ≥50% | **Anotujemy** |
| Widoczność <50% (mocno zasłonięty) | **Pomijamy** |
| Rozmiar ≥15×15 px | **Anotujemy** |
| Rozmiar <15×15 px | **Pomijamy** (model się nie nauczy) |
| Kilka znaków na jednym słupie | **Każdy osobny box** |
| Znak od tyłu (rewers) | **Nie anotujemy** (nie da się rozpoznać klasy) |
| Wyblakły / nieczytelny | **Pomijamy** (nie zgadujemy) |
| Znak częściowo poza kadrem | Anotujemy widoczną część jeśli ≥50% widać |

**Box rysujemy ciasno wokół znaku** — nie wokół słupa, nie z marginesem.

---

## Strategia zbierania ze Street View

1. **Wybierz lokalizację** — zacznij od Poznania (znajomy teren), potem dwa inne miasta dla różnorodności
2. **"Spaceruj" co 50-100m**, screen tylko gdy widzisz znak z listy
3. **Mieszaj warunki:**
   - Różne kąty (z przodu, z boku, z dystansu)
   - Różne pory roku jeśli Street View ma kilka przejazdów
   - Centrum miasta + obrzeża + droga poza miastem
4. **1-3 znaki na obraz** — idealnie. Czasem 1 znak wystarczy, czasem złap kilka naraz przy skrzyżowaniu.
5. **Naming:** `img_001.jpg`, `img_002.jpg`, ...
6. **Format:** JPG, max 1280px po dłuższym boku

## Czego unikać przy zbieraniu

- Close-upy znaku wypełniające cały kadr → to klasyfikacja, nie detekcja
- Zdjęcia z plansz / tablic / ulotek edukacyjnych → domain gap, model się nie nauczy
- Same identyczne lokalizacje → bias, brak generalizacji
- Sceny bez żadnego znaku z listy → marnujesz miejsce w datasetcie

---

## dataset.yaml (do wygenerowania)

```yaml
path: ../data
train: splits/train/images
val: splits/val/images
test: splits/test/images
nc: 5
names:
  0: stop
  1: yield
  2: speed_limit
  3: pedestrian_crossing
  4: no_entry
```
