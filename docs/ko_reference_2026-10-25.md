# Knock-out-Referenz: Messung gegen echte Kurse (2026-10-25)

**Nichtinvasiv.** Kein Produktionsmodell, kein Risikolimit, kein
ACTIONABLE-Kriterium, kein Promotion-Gate geändert. Nichts in
`src/turboedge` außerhalb von `research/` importiert
`research/ko_reference.py`; ein Test erzwingt das.

## Aufbau der Messung

Drei Verfahren, dieselbe Barriere, dieselbe kausale Volatilität, dieselben
Fenster:

- **realisiert** — Grundwahrheit aus echten Tageshochs/-tiefs
  (yfinance, 2010-01-01 bis 2026-10-23). σ_t aus `features/volatility.
  ewma_volatility` (λ=0,94) ausschließlich aus Schlusskursen bis
  einschließlich t (Regel 5). Barriere bei `S_t·exp(∓k·σ_t·√h)`, geprüft
  gegen das echte Tief (Long) bzw. Hoch (Short) über t+1..t+h.
- **sim_default** — `simulation/paths.simulate_paths` mit den
  Produktionsvorgaben (`vol_scaled_bootstrap`, 400 Tage Lookback,
  2.000 Pfade), Treffer über `simulation/barrier.first_hit_index`.
- **sim_sigma** — dieselbe Simulation, aber mit `target_sigma = σ_t·√h`.
  Der Parameter **existiert bereits** in `simulate_paths`; die produktive
  Pipeline übergibt ihn nicht (vgl. `measured_results.md` §6.14).
- **analytisch** — `research/ko_reference.analytic_touch_probability`,
  Log-Drift null, also exakt `2Φ(−k)`.

Gitter: 4 Basiswerte (DAX, S&P 500, EUR/USD, Gold) × h ∈ {3, 5, 10, 14} ×
k ∈ {1,5, 2,0, 2,5, 3,0} × {long, short} = **128 Zellen**, je 771–773
Fenster (Schrittweite 5 Handelstage). Konfidenzintervalle über einen
Moving-Block-Bootstrap mit Blocklänge ≥ Horizont, weil aufeinanderfolgende
Fenster bis zu h−1 Tage teilen.

**Validierung der Messung:** für DAX, h=10, k=2 ergibt die Messung
realisiert 0,0751 (long) und 0,0324 (short). `measured_results.md` §3
dokumentiert 0,072 und 0,036. Die Grundwahrheit ist also unabhängig
reproduziert.

## Ergebnis 1 — die analytische Referenz ist nicht besser, sondern symmetrisch

| Verfahren | MAE | max | im 95%-CI | überschätzt |
|---|---:|---:|---:|---:|
| sim_default | **0,01404** | 0,05296 | **86/128 (67 %)** | 113/128 |
| analytisch | 0,01749 | 0,05186 | 46/128 (36 %) | 38/128 |

**Das korrigiert meinen eigenen früheren Befund.** Am 2026-10-25 hatte ich
aus den zwei dokumentierten DAX-Zellen geschlossen, die analytische
Referenz sei der Realität um Faktor ~2,6 näher als die Simulation. Über das
volle Gitter ist das falsch: die Simulation hat den kleineren mittleren
Fehler und trifft das Konfidenzintervall fast doppelt so oft. Selbst wenn
man nur k=2 betrachtet, aber über alle vier Basiswerte (32 Zellen), führt
die Simulation (MAE 0,01496 gegen 0,02076; 69 % gegen 38 % CI-Abdeckung).
Der frühere Schluss hing an genau zwei Zellen eines Basiswerts.

Der Grund ist strukturell, nicht numerisch. Die realisierte
Long/Short-Trefferquote ist nicht symmetrisch, und die Asymmetrie wächst
mit dem Barriereabstand:

| k | long | short | long/short | analytisch |
|---|---:|---:|---:|---:|
| 1,5 | 0,1428 | 0,1196 | 1,19 | 0,1336 |
| 2,0 | 0,0677 | 0,0493 | 1,37 | 0,0455 |
| 2,5 | 0,0335 | 0,0191 | 1,75 | 0,0124 |
| 3,0 | 0,0176 | 0,0087 | 2,01 | 0,0027 |

Und sie tritt genau dort auf, wo die Theorie sie erwartet — das ist die
Falsifikationsprobe, nicht nur eine Beschreibung:

| Basiswert | long/short bei k=2 |
|---|---:|
| S&P 500 | 2,75 |
| DAX | 2,10 |
| EUR/USD | 1,09 |
| Gold | 0,78 |

Aktienindizes zeigen den Leverage-Effekt (negative Schiefe, Abwärtsbarriere
wird häufiger berührt), EUR/USD ist praktisch symmetrisch, Gold leicht
invertiert. Eine driftfreie Brown'sche Bewegung hat für Schiefe **keinen
Parameter**; sie kann das bei keiner Parametrisierung abbilden. Die
Schlussfolgerung ist deshalb nicht „die einfachere Methode ist besser",
sondern: **ein geschlossener Ausdruck ist das richtige Werkzeug, um
Numerik zu prüfen, und das falsche, um einen Index zu prognostizieren.**

## Ergebnis 2 — die Simulation überstreut, und der Parameter dagegen existiert schon

Gemessenes Verhältnis von realisierter Horizont-σ der simulierten Pfade zur
Volatilität, mit der die Barriere platziert wurde:

| Basiswert | Standard | mit `target_sigma` |
|---|---:|---:|
| DAX | 1,358 | 1,089 |
| S&P 500 | 1,223 | 1,022 |
| Gold | 1,332 | 1,044 |
| EUR/USD | 1,707 | 1,610 |
| **Mittel** | **1,405** | 1,191 |

Die Ursache ist eine Regime-Inkonsistenz, kein Fehler: der Bootstrap zieht
aus 400 Tagen, die Barriere wird mit einer EWMA-Schätzung von etwa 16 Tagen
Halbwertszeit platziert. Volatilität ist rechtsschief, also liegt die
Streuung eines langen Fensters meist über der heutigen. Das genügt, um die
dokumentierte Überschätzung zu erklären — die Simulation liest in 113 von
128 Zellen zu hoch.

Auf Aktienindizes (64 Zellen) ist die σ-Angleichung eine **messbare
Verbesserung**:

| Verfahren | MAE | im 95%-CI |
|---|---:|---:|
| sim_sigma | **0,00615** | **58/64 (91 %)** |
| sim_default | 0,01224 | 44/64 (69 %) |
| analytisch | 0,01723 | 22/64 (34 %) |

Der mittlere Fehler halbiert sich, und die CI-Abdeckung steigt von 69 % auf
91 % bei einem Nominalniveau von 95 %. Gold bleibt neutral (0,0119 →
0,0125).

**Die Falsifikation: EUR/USD.** Dort verschlechtert die σ-Angleichung das
Ergebnis drastisch (MAE 0,0198 → 0,1553), und das Verhältnis bleibt bei
1,610 — das Ziel ist unerreichbar, weil die Gap-Komponente allein darüber
liegt. Ob das am Modell liegt oder an den synthetischen Tages-OHLC, die
yfinance für FX liefert, ist hier **nicht** entschieden. Deshalb:

- Empfehlung **WEITER VALIDIEREN**, nicht `IMPLEMENTIEREN`.
- Keine produktive Umstellung. Dieser Auftrag gibt Modellwechsel nicht frei,
  und die Richtung der Korrektur verlangt ohnehin eine Betreiberentscheidung:
  eine zu hohe P(KO) macht das System **zu konservativ**. Sie zu senken
  würde ACTIONABLE leichter erreichbar machen. Das ist eine Lockerung mit
  wirtschaftlicher Wirkung und gehört nicht in einen Forschungsauftrag.

## Ergebnis 3 — ein Prozentabstand ist kein Log-Abstand, und der Fehler ist asymmetrisch

`simulation/ko_calibration.py` platziert Barrieren bei `spot·(1 ± a)` mit
`a = k·σ·√h`. Weil `exp(−a) > 1−a`, liegt die Long-Barriere in Log-Maßen
**weiter** weg als `a` und die Short-Barriere **näher** — ein nominelles
`k` bedeutet zwei verschiedene Abstände. `linear_offset_bias` rechnet es
aus (σ=0,011, h=10):

| k | k_eff long | k_eff short | p_short/p_long |
|---|---:|---:|---:|
| 1,5 | 1,541 | 1,462 | 1,16 |
| 2,0 | 2,073 | 1,933 | **1,39** |
| 2,5 | 2,615 | 2,397 | 1,85 |
| 3,0 | 3,168 | 2,854 | 2,82 |
| 4,0 | 4,307 | 3,745 | **10,90** |

`DEFAULT_SIGMA_LEVELS` reicht bis 4,0, das liegt also im gemessenen Gitter.
Empirisch bestätigt (DAX, h=10, k=3): lineare Platzierung ergibt
long/short 7,92, logarithmische 11,20 — die lineare Variante drückt die
echte Asymmetrie zusammen, weil sie Short künstlich begünstigt.

**Tragweite, nüchtern:** keine Handelsentscheidung liest diese Barrieren.
Produktive Kandidaten tragen die echte Emittentenbarriere. Verzerrt wird
eine **Messung**, nicht ein Trade. Es bedeutet aber, dass der Long/Short-
Anteil der in §3 dokumentierten Überschätzung teilweise Geometrie ist. Da
W10 mit `NO PROMOTION` endete, wurde auf dieser Grundlage kein Kalibrator
befördert — es ist kein Schaden entstanden, nur eine unsauber vergleichbare
Zahl.

## Einordnung nach Phase 13G

| Befund | Klassifikation | Begründung |
|---|---|---|
| Analytische Referenz als Numerikprüfung | `NUR RESEARCH` | exakt und nützlich, aber als Marktmodell strukturell unfähig zu Schiefe |
| σ-Angleichung (`target_sigma`) | `WEITER VALIDIEREN` | MAE halbiert auf Aktien, scheitert auf FX; Richtung lockert Gates |
| Lineare Barriereplatzierung | `IMPLEMENTIEREN` (Messpfad) | mathematisch eindeutig, betrifft nur Research-Messung |
| Brownian-Bridge-Korrektur | `VERWERFEN` | siehe `mathematical_validation_2026.md`: Vorzeichen falsch |

---

*Reproduzierbar mit festen Seeds (20260125 Simulation, 11 Bootstrap). Die
Testsuite prüft Mechanismus und Richtung, nicht die dritte Dezimale.*
