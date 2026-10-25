# Scientific Alpha Discovery 2026 — Abschlussbericht

**Stand 2026-10-25.** Nichtinvasiv: kein Risikolimit, kein ACTIONABLE-Kriterium,
kein Promotion-Gate, kein Forschungsbudget und kein historischer Messwert
geändert. `PROMOTION_LOCKED` steht. Kein Signal aktiviert.

---

## Vorbemerkung: was ich geprüft habe und was nicht

Mein Wissensstand endet im Mai 2026. Von den 17 Arbeiten habe ich **sechs**
gegen Originalquellen geprüft und berichte unten, wo die Angaben des
Auftrags von dem abweichen, was ich gefunden habe. Die übrigen elf habe ich
**nicht** verifiziert und markiere sie entsprechend — nicht als „geprüft und
plausibel", sondern als ungeprüft.

**Zwei Korrekturen, die den Auftrag selbst betreffen**, und beide treffen
genau die Arbeiten, auf denen Experiment 1 beruhen sollte:

- **A06/P06 (Gao, Han, Li, Zhou 2018, JFE)** ist verifiziert, aber die
  Behauptung lautet: die **erste halbe Stunde** prognostiziert die **letzte
  halbe Stunde** auf SPY, 1993–2013, In-Sample-R² etwa 0,02. Ein Tagesbalken
  enthält weder die eine noch die andere. Gemessen 2026-10-25: 30-Minuten-
  Historie reicht **60 Tage**, Stundenhistorie 730. Die Hypothese ist damit
  nicht testbar — `AWAIT_DATA` aus gemessenem Grund, nicht aus Desinteresse.
- **A12/P12 (Barardehi, Bogousslavsky, Muravyev)** ist **nicht** als
  RFS-2026-Artikel bestätigt. Ich finde ein Working Paper von März 2022
  (Stichprobe 1926–2019) und einen Autoren-CV-Eintrag „RFS, Revise and
  Resubmit". Wichtiger als das Datum ist der Inhalt: die Arbeit bildet
  **querschnittliche Portfolios** aus US-Einzelaktien, sortiert nach
  Intraday- bzw. Overnight-Renditen. Intraday-Portfolios zeigen
  kurzfristige Umkehr und mittelfristiges Momentum, Overnight-Portfolios nur
  langfristige Umkehr. TurboEdge handelt vier Basiswerte (DAX, NDX, EURUSD,
  XAU) und hat damit **keinen Querschnitt**. Der Mechanismus der Arbeit steht
  diesem System nicht zur Verfügung.

Experiment 1 hat deshalb das nächstliegende **Zeitreihen**-Analogon getestet
— eine schwächere und andere Behauptung als die der Publikationen. Das ist
im Ergebnis unten ausdrücklich so ausgewiesen.

---

## A. Literaturprüfung

| ID | Verifiziert | Mechanismus | Übertragbarkeit | Daten vorhanden | Entscheidung |
|---|---|---|---|---|---|
| A01/P01 Bianchi, Ludvigson, Ma | **nein** | Überreaktion auf Makro-News-Komponenten | braucht zeitpunktgetreue Konsensdaten | **nein** | `AWAIT_DATA` |
| A02/P02 Driessen, Ebert, Koëter (Π-CAPM) | **nein** | Probability Weighting: Lotterie-Payoffs zu teuer | hoch — Turbos *sind* Lotterie-Payoffs | teilweise (Snapshot-Archiv) | `READY`, nicht ausgeführt |
| A03/P03 Varianzprämie, saisonal | **nein** | saisonale Vola-Zyklen unvollständig eingepreist | braucht Optionsdaten | **nein** | `AWAIT_DATA` |
| A04/P04 Retail-Optionsnachfrage | **nein** | Retail-Flow verzerrt IV-Oberfläche | braucht Optionsflussdaten | **nein** | `AWAIT_DATA` |
| A05/P05 Time-Varying Skewness | **nein** | bedingte Schiefe als Momentum-Crash-Filter | mittel, als Risiko-Challenger | Bars vorhanden | `READY`, nicht ausgeführt |
| A06/P06 Gao et al. 2018 | **ja** | erste → letzte halbe Stunde | **nein** bei Tagesdaten | **nein** (30 min = 60 Tage) | `AWAIT_DATA`, gemessen |
| A07/P07 Market-Maker-Hedging | **nein** | Delta-Hedging-Druck und Reversal | braucht Options-Orderflow | **nein** | `AWAIT_DATA` |
| A08/P08 Passive Flows | **nein** | Index-Rebalancing-Verzerrungen | braucht Flow-/Indexdaten | **nein** | `AWAIT_DATA` |
| A09/P09 Crypto-Momentum, realistisch | **nein** | Methodik: Pfadabhängigkeit tötet Scheinalpha | **hoch als Methode** | n/a | bereits umgesetzt (KO-Pfad, Kosten) |
| A10/P10 Duarte et al., Look-Ahead | **nein** | Look-Ahead-Bias in Optionsforschung | **hoch als Methode** | n/a | **umgesetzt** — siehe D |
| A11/P11 Lee & Wang, FOMC-FX | **nein** | Reversal nach FOMC in EUR/USD | braucht Intraday-FX zu Ankündigungszeiten | **nein** | `AWAIT_DATA` |
| A12/P12 Barardehi et al. | **ja, mit Korrektur** | Querschnitt: Intraday- vs. Overnight-Signale | **nein** — kein Querschnitt bei 4 Basiswerten | Bars vorhanden | Zeitreihen-Analogon getestet → `FAIL` |
| A13/P13 Guo, Earnings Extrapolation | **nein** | Prognosekraft variiert im Berichtsquartal | braucht Berichtskalender | **nein** | `AWAIT_DATA` |
| A14/P14 Papagelis & Dotsis | **nein** | VRP getrennt nach Handels-/Nichthandelszeit | braucht Optionsdaten | **nein** | `AWAIT_DATA` |
| A15/P15 Bali et al., Options-ML | **nein** | Optionsmerkmale als Prädiktoren | braucht Optionspaneldaten | **nein** | `AWAIT_DATA` |
| A16/P16 Goyal & Saretto, IPCA | **nein** | Faktorbereinigung echten Alphas | **hoch als Methode** | n/a | `OFFEN` |
| A17/P17 Jensen et al., Implementable Frontier | **nein** | Modellauswahl nach Netto-Ökonomie | **hoch als Methode** | teilweise | `OFFEN` |

Zusätzlich aus früheren Sitzungen **bereits gemessen und dauerhaft
dokumentiert** (nicht Teil dieser 17): analytische First-Passage-Modelle und
Brownian-Bridge-Korrekturen (`docs/ko_reference_2026-10-25.md`), Adaptive
Multilevel Splitting und Importance Sampling (verworfen, MC-Fehler ist nur
0,23–0,31× des Modellfehlers), Tail Calibration (W10, `NO PROMOTION`),
Anytime-valide Tests und E-Prozesse (`docs/mathematical_validation_2026.md`),
MNAR-Analyse fehlender Labels und Stale-Quote-Analysen
(`docs/scientific_upgrade_2026-10-25.md`).

---

## B. Tatsächlich ausgeführte Experimente

### Experiment 1 — Tag/Nacht-Momentum-Familie · **FAIL**

| Feld | Wert |
|---|---|
| Hypothese | `RO-DAY-NIGHT-MOMENTUM` (neu im Katalog) |
| Märkte | DAX, NDX, EURUSD, XAU (in-universe) + SX5E als Kontext |
| Zeitraum | 1990-01-02 bis 2026-10-23 (DAX/NDX), 2003/2000 ff. für EURUSD/XAU |
| Stichprobe | 8.814 / 8.770 / 5.440 / 6.062 OOS-Beobachtungen je Zelle |
| **Effektive** Stichprobe | **= nominell** — Ein-Tages-Ziele überlappen nicht |
| Nullmodell | zirkuläre Blockverschiebung (erhält Drift *und* Autokorrelation) + Always-Long + NO_TRADE |
| Mehrfachtest | Benjamini-Hochberg über **alle 24 Zellen als eine Familie** |
| Prognosevorteil | 2 von 24 Zellen überleben BH — **beide dieselbe Shared-Price-Zelle** |
| Wirtschaftlicher Vorteil | **keiner.** `best_break_even_cost` = `nan` |
| Urteil | `LOOKAHEAD_SUSPECT` → **FAIL** |

**Was gefunden wurde und warum es kein Alpha ist.** DAX NIGHT_DAY ergab
8,2 bp/Tag (p = 0,0005), SX5E 23,2 bp/Tag. Beide sind das einzige Paar, bei
dem Signal und Ziel den **Eröffnungskurs teilen**. `night` und `day`
enthalten O_t mit entgegengesetztem Vorzeichen; ein *stale* Open — gestellt,
bevor alle Bestandteile gehandelt haben, also zum Vortagesschluss hin
verzerrt — hinterlässt einen Teil der Übernachtlücke auch im Sessionverlauf,
mit gleichem Vorzeichen. `corr(night, day)` beträgt **+0,264** bei SX5E und
+0,028 beim DAX. Eine Viertelkorrelation zwischen Übernachtlücke und
darauffolgender Session ist keine Markteigenschaft.

Nimmt man den geteilten Preis heraus, bricht es zusammen:

| | geteilt | sauber (nächste Session) | Faktor |
|---|---:|---:|---:|
| DAX | 0,000771 | 0,000150 | **5×** |
| SX5E | 0,002555 | 0,000100 | **25×** |

Zwei saubere Teilhypothesen wurden daraufhin **ergänzt** — nie ersetzt, die
Korrektur konnte also nur strenger werden — und keine überlebt: DAX
p = 0,0770, NDX p = 0,0820 gegen eine BH-Rang-1-Schwelle von 0,00417, beide
Intervalle durch die Null.

Unabhängig von der Datenqualität gilt ein schlichter
Ausführbarkeitseinwand: **niemand kann den Eröffnungskurs lesen und zu ihm
handeln.**

**Zwei Nebenbefunde.** EURUSD hat eine Übernacht-Standardabweichung von
**1,709×** der Session — für einen fast 24 Stunden gehandelten Markt ist das
der Schnitt des Datenanbieters, keine Markteigenschaft; die Zerlegung ist
dort wirtschaftlich nicht interpretierbar. Die Zellen bleiben trotzdem im
BH-Nenner, denn Zellen nach Sichtung zu entfernen macht Überleben leichter.
Und: das Signal ist ein **Ein-Tages**-Signal, während
`configs/forecast.yaml` 3/5/7/10/14 Tage fährt. Einen Ein-Tages-Horizont
gibt es im System nicht — selbst ein funktionierender Vorteil wäre ohne
Produktionsänderung nicht handelbar.

### Experiment 2 — FOMC-Reversal EUR/USD · **AWAIT_DATA**

Nicht ausgeführt, aus gemessenem Grund: der Effekt lebt in den Minuten bis
Stunden nach 20:00 CET, und verfügbare Intraday-Historie reicht 60 Tage
(30 min) bzw. 730 Tage (1 h). Für ein Event-Study-Design mit ~8
FOMC-Terminen pro Jahr wären das etwa 16 verwertbare Ereignisse bei
Stundenauflösung. Das ist keine Stichprobe, auf der ein Reversal von
Zufall zu trennen wäre. Eine Forward-Erfassung wäre der Weg, kein Backtest.

### Experiment 3 — Emittenten- und Hebelpreisverzerrungen · **teilweise, früher**

Der Quote-Alter-Teil wurde am 2026-10-25 gemessen und ist der **einzige
positive Befund dieser Arbeit**: über 125.296 archivierte Snapshots beträgt
die gruppeninterne Rangkorrelation zwischen Günstigkeit und Quote-Alter
**+0,1852, 95 %-KI [+0,1234, +0,2450]** über 184 unabhängige Marktmomente
(`docs/scientific_upgrade_2026-10-25.md`). Er zeigt in die **Gegenrichtung**
eines Alphas: ein Teil des sichtbaren Preisvorteils ist Veraltung.

Der Hebel-/Π-CAPM-Teil (A02) ist **nicht** ausgeführt. Er ist der
aussichtsreichste offene Kandidat — siehe E.

---

## C. Rangfolge nach gemessenem Alpha-Potenzial

**1. Potenziell wirtschaftlich relevant, Confirmation nötig:** *keine.*

**2. Wissenschaftlich interessant, bisher ohne Netto-Vorteil:**
- Quote-Alter-Assoziation (+0,1852 [+0,1234, +0,2450]) — ein gemessener
  Mechanismus, aber er erklärt einen Scheinvorteil statt einen zu liefern.
- σ-Angleichung der Pfadsimulation — halbiert den P(KO)-Fehler auf
  Aktienindizes, lockert aber ein Gate und scheitert auf FX.

**3. Unzureichende Daten:** A01, A03, A04, A07, A08, A11, A13, A14, A15 und
A06 — neun von zehn wegen fehlender Options-, Flow-, Konsens- oder
Intraday-Daten, A06 gemessen.

**4. Methodisch nicht übertragbar:** A12 (kein Querschnitt bei vier
Basiswerten), A06 (Granularität), A09 (Crypto-Mechanik; als *Methode* aber
schon umgesetzt).

**5. Widerlegt beziehungsweise gescheitert:** die Tag/Nacht-Familie, 24
Zellen, `LOOKAHEAD_SUSPECT`. Zusätzlich die sieben bereits dormanten
Hypothesen in `state/registry/failed_hypotheses.json`, von denen keine
berührt wurde.

---

## D. Implementierte Änderungen

| Änderung | Tests | Vergleich zur Baseline | Messbare Verbesserung | Commit | CI | Merge |
|---|---|---|---|---|---|---|
| `backtest/day_night_eval.py` — Zerlegung, Shift-Null, BH-Familie, **Pflicht-Look-Ahead-Audit** | 31 | es gab keinen Tag/Nacht-Test | erkennt einen Stale-Open-Artefakt, der sonst als 8–23 bp/Tag Alpha gelesen würde | `e5076b8` | grün | PR #8 |
| `research/harness.py` — `day_night` registriert | — | 1 Harness für 2 von 14 Fragen | **2 Harnesses für 3 von 15** | `e5076b8` | grün | PR #8 |
| `meta/catalog.py` — `RO-DAY-NIGHT-MOMENTUM` | 7 angepasst | 14 Hypothesen | 15; erste ohne Label-Abhängigkeit | `e5076b8` | grün | PR #8 |
| `research/controller.py` — Rangfolge statt nur Spitze | 4 | eine blockierte Spitzenfrage hungerte den Loop aus | Loop erreicht lauffähige Fragen | `051fb69` | grün | PR #9 |
| `research/harness.py` + `step.py` — **gemessene Bereitschaft** | 5 | `START_EXPLORATORY` war für *jede* Frage unerreichbar | Loop kann erstmals ein Experiment starten | — | grün | PR #10 |

**Die beiden Controller-Befunde sind das Wichtigste an diesem Auftrag**,
und beide wurden gefunden, indem der Loop ausgeführt statt gelesen wurde:

1. `decide_next_action` nahm `ranked[0]` und gab bei fehlenden Daten
   `AWAIT_DATA` zurück, **ohne `ranked[1:]` anzusehen** (Lauf 40235422444:
   Rang 1 wartet auf Labels, Rang 4 wäre lauffähig gewesen).
2. Danach zeigte Lauf 40236515594, dass **alle 14** Kandidaten
   `data_ready=False, effective_sample=None` melden. `effective_sample`
   verlangt ein MEASURED `estimated_sample_size`, das der Katalog
   absichtlich nur deklariert; `data_ready` verlangt die Erreichbarkeit der
   **Familien**-Quelle, für Microstructure also `bnp_paribas` — einen
   Produktfeed, den der Tag/Nacht-Screen nie liest. Zusammen war
   `START_EXPLORATORY` seit dem Bau des Loops unerreichbar, was
   `executed_screens_this_quarter = 0` erklärt.

Keine Gate-, Limit-, Schwellen- oder Budgetänderung. Die drei Wächtertests,
die Registry- und Katalogzahlen festnagelten, wurden nachgezogen, **nicht
gelockert** — ihre schützenden Zusicherungen (keine konfirmatorische
Harness, keine promotionsschreibende Harness) sind unberührt.

---

## Autonomie-Status

| Funktion | Status | Nachweis |
|---|---|---|
| Hypothesen dauerhaft gespeichert | **ja** | `meta/catalog.py`, 15 Einträge; `state/registry/failed_hypotheses.json`, 7 |
| Alle 17 Arbeiten berücksichtigt | **teilweise** | 6 verifiziert, 11 ungeprüft; alle in Abschnitt A bewertet |
| Weitere math. Methoden gespeichert | **ja** | 4 `docs/`-Berichte, dauerhaft |
| Forschungshistorie persistent | **ja** | Screen `2026Q4_SCREEN_230da101ff` im nächsten Lauf wiedergefunden |
| Neue Daten automatisch erfasst | **ja** | `pipeline.yml`, 5×/Werktag |
| Forward-Labels erzeugt | **nein, korrekt** | 3.670 offen, 0 fällig; erste Reife 2026-10-26 |
| Learning Loop verarbeitet Labels | **unbewiesen** | keine Labels zum Verarbeiten |
| Research Queue aktualisiert sich | **ja** | Kandidaten 13 → 12 nach dem Screen |
| **Neue Experimente starten automatisch** | **ja, jetzt** | Lauf 40237604797: `START_EXPLORATORY` ohne Chat-Eingabe |
| Ergebnisse dauerhaft gespeichert | **ja** | `screen_id`, `data_version 6a2c39b6612980b3`, `state_packed` |
| Negative Ergebnisse bleiben | **ja** | `NO_IMPROVEMENT` gespeichert, nicht wiederholt |
| Forward Shadow | **unbewiesen** | nie erreicht — es gab nie einen Kandidaten |
| Alpha-Validierung | **teilweise** | Stufe 1 läuft; Stufen 2–4 nie ausgelöst |
| Alpha-Degradation erkannt | **unbewiesen** | kein bestätigtes Alpha, das verfallen könnte |
| E-Mail-Benachrichtigung | **unbewiesen für Alpha** | Infrastruktur steht; nie für einen Kandidaten ausgelöst |
| Täglicher/wöchentlicher Scheduler | **ja** | `pipeline.yml` Crons, `research step` Zeile 588 |
| State Backup und Restore | **ja** | `state_unpacked` → `state_packed` in jedem Lauf |
| Nach `main` gemergt | **ja** | PR #8, #9, #10 |

### Nachweis autonomer Arbeitsweise

1. Hypothese selbstständig gewählt — **40237604797**, `START_EXPLORATORY`
   auf `RO-DAY-NIGHT-MOMENTUM`, effektive Stichprobe 3.500, „3 höherrangige
   Fragen mangels Daten übersprungen".
2. Experiment tatsächlich ausgeführt — `harness: day_night@1`, 1,666 s.
3. Ergebnis gespeichert — `screen_id 2026Q4_SCREEN_230da101ff`,
   `verdict NO_IMPROVEMENT`, `supported False`.
4. Späterer Lauf hat es wiederverwendet — **40237731541**,
   `executed_screens_this_quarter: 1`, **kein** erneuter Screen.
5. Queue korrekt aktualisiert — 12 statt 13 screenbare Kandidaten.
6. Nächster Versuch sinnvoll zurückgestellt — `AWAIT_DATA` mit allen 12
   namentlich.

---

## Antworten auf die zehn Abschlussfragen

**1. Ist TurboEdge-DE jetzt ein autonom arbeitendes Forschungssystem?**
Für den explorativen Teil ja, und erst seit heute: vor PR #10 konnte der Loop
überhaupt kein Experiment starten. Für Confirmation und Promotion nein, und
das ist Absicht.

**2. Kann es ohne Chat-Eingaben neue zulässige Experimente durchführen?**
Ja, belegt durch 40237604797. Grenze: zwei registrierte Harnesses; die
übrigen zwölf Fragen melden korrekt `NO_APPROVED_HARNESS`.

**3. Lernt es nachvollziehbar aus echten Forward-Daten?** Noch nicht, weil
es keine gibt — 3.670 offene Einträge, 0 aufgelöst, erste Reife morgen. Die
Kette ist gebaut und ungetestet an echten Labels.

**4. Vergisst es frühere Erkenntnisse?** Nein. Sieben dormante Hypothesen im
Register, vier Messberichte in `docs/`, und das Screen-Ergebnis wurde im
Folgelauf wiedergefunden statt wiederholt.

**5. Erkennt es selbstständig, wenn eine Hypothese testbar wird?** Jetzt ja —
die Harness zählt ihre eigenen Daten. Vorher nein, und zwar strukturell.

**6. Kann es Alpha-Kandidaten gegen Zufall absichern?** Ja, und heute
nachgewiesen: BH über die ganze Familie, ein Null-Modell, das Drift *und*
Autokorrelation erhält, und ein Look-Ahead-Audit, das den einzigen
„Treffer" als Artefakt entlarvte.

**7. Kann es bestätigtes Alpha zurückstufen?** Code dafür existiert
(`DriftEvent`, Registry-Status), ist aber nie gelaufen, weil nie etwas
bestätigt war. Unbewiesen.

**8. Werde ich bei Alpha-Entdeckungen automatisch informiert?** Die
Infrastruktur steht; für einen Alpha-Kandidaten wurde sie nie ausgelöst, weil
es keinen gab. Eine dedizierte Research-Discovery-Mail existiert **nicht** —
offen, siehe E.

**9. Was ist real nachgewiesen, was nur durch Tests abgesichert?** Real
nachgewiesen: Datenerfassung, State-Kontinuität, Queue-Scoring,
Experimentauswahl, Harness-Ausführung, Ergebnispersistenz, Wiederverwendung.
Nur durch Tests: Labeling-Reife, Learning-Loop auf echten Labels, Forward
Shadow, Alpha-Bestätigung und -Rückstufung, Alpha-Benachrichtigung.

**10. Ist wirtschaftliches Alpha nachgewiesen?** **Nein.** Kein Verfahren in
dieser oder einer früheren Sitzung hat einen positiven Netto-Erwartungswert
nach Kosten gezeigt. Der stärkste Befund — das Quote-Alter — erklärt, warum
ein Teil des bisher für Vorteil Gehaltenen keiner ist. `NO TRADE` bleibt das
richtige Ergebnis.

---

## E. Offene Research Queue, priorisiert

1. **A02 Π-CAPM auf Turbos** (`READY`, höchste Priorität). Die
   aussichtsreichste unausgeführte Hypothese: Turbos *sind*
   Lotterie-Payoffs, das Snapshot-Archiv hat 125.296 Beobachtungen und 218
   emittentenübergreifende Vergleichsgruppen. Falsifizierbar: steigt die
   Emittentenmarge systematisch mit dem Hebel? Braucht eine Harness.
2. **Label-Reife abwarten** (`AWAIT_MATURITY`, ab 2026-10-26). Erst dann
   werden `RO-MAE-PREDICTION`/`RO-MFE-PREDICTION` lauffähig und der Learning
   Loop erstmals an echten Daten prüfbar.
3. **Research-Discovery-Benachrichtigung** (offen). Der Auftrag verlangt sie
   verbindlich; sie existiert nicht. Klein und unabhängig umsetzbar.
4. **A05 Skewness als Regime-Challenger** (`READY`). Bars genügen, Harness
   fehlt. Ausdrücklich **kein** Richtungssignal.
5. **A16/A17 als Methoden** (offen). Faktorbereinigung und kostenbewusste
   Modellauswahl werden erst relevant, wenn es einen Kandidaten gibt.
6. **Zu schließen:** A06 und A12 in der vom Auftrag gemeinten Form — A06 an
   der Datentiefe, A12 am fehlenden Querschnitt. Nicht wieder öffnen ohne
   Intraday-Daten beziehungsweise ein Aktienuniversum.

### Was menschliche Mitwirkung zwingend braucht

- **Confirmatory-Trials.** `research approve` verlangt einen benannten
  Menschen; der Loop darf das nicht. Unverändert.
- **Promotion-Gates.** `PROMOTION_LOCKED`, sieben ungesetzte Kriterien.
- **Optionsdaten.** Neun der 17 Hypothesen scheitern daran. Eine Beschaffung
  ist eine Kosten- und Lizenzentscheidung, keine Forschungsaufgabe.
- **Ein-Tages-Horizont**, falls ein Intraday-Signal je verfolgt werden soll.

---

*Messläufe: [40235422444](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40235422444),
[40236515594](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40236515594),
[40237604797](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40237604797),
[40237731541](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40237731541).
Seeds 20261025 (Screen), 23 (Bootstrap).*
