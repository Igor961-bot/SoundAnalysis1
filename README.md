# Projekt 1 - Cechy sygnalu audio w dziedzinie czasu

Prosta aplikacja okienkowa w Pythonie do analizy plikow WAV.

Projekt realizuje:

- wczytanie pliku `*.wav`
- wyswietlenie przebiegu czasowego
- obliczenie cech frame-level
- obliczenie cech clip-level
- detekcje ciszy
- estymacje F0 metodami autokorelacji i AMDF
- podzial na fragmenty `voiced / unvoiced / silence`
- prosty podzial na fragmenty `speech / music / silence`
- eksport wynikow do `csv` i `txt`
- dodatkowo: dominujaca czestotliwosc z FFT

## Zaleznosci

W logice audio uzyty zostal tylko jeden zewnetrzny pakiet:

- `numpy` - tablice numeryczne i FFT

Do zbudowania okna uzyta jest biblioteka:

- `PyQt5`

## Uruchomienie

Najprostsza forma:

```bash
python3 main.py
```

Mozna tez od razu podac plik WAV:

```bash
python3 main.py sciezka/do/pliku.wav
```

## Uwagi implementacyjne

- Odczyt pliku WAV jest zrobiony przez standardowa biblioteke `wave`.
- Cechy z PDF sa liczone recznie we wlasnych funkcjach.
- FFT jest liczone przez `numpy.fft.rfft`.
- Po wczytaniu pliku nalezy kliknac `Analizuj`.
- Dla szybszego liczenia sygnal do analizy moze zostac prostym sposobem zmniejszony do ok. `16 kHz`.
- Progi ciszy oraz podzial `speech / music` sa oparte o proste heurystyki, bo w materiale nie bylo podanych sztywnych progow liczbowych.
- Aplikacja najlepiej dziala dla nieskompresowanych plikow WAV PCM.
