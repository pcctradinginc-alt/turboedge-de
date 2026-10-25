# Wissenschaftliches Upgrade 2026 — Abschlussbericht

**Stand 2026-10-25.** Alle Messungen reproduzierbar mit festen Seeds.
Nichtinvasiv: kein Produktionsmodell, kein Risikolimit, kein
ACTIONABLE-Kriterium, kein Promotion-Gate, kein historischer Messwert und
kein Label geändert. `PROMOTION_LOCKED` steht.

## Vorbemerkung zur Literatur

Mein Wissensstand endet im Mai 2026. Die im Auftrag genannten Arbeiten habe
ich, soweit öffentlich erreichbar, **auf Existenz und Inhalt geprüft** und
sage bei jeder, was tatsächlich übertragbar ist. Ich behaupte nirgends, eine
Volltextfassung gelesen zu haben, wo ich nur das Abstract gesehen habe.

| Publikation | Status | Übertragbar? |
|---|---|---|
| Wei, Qu, Miao — *OPE for Missingness-Aware Policies … MNAR* | existiert, [arXiv 2606.20206](https://arxiv.org/abs/2606.20206), 18.06.2026 | **Teilweise** — das Identifikationsmittel (Shadow Variable, Bridge Function) ist übertragbar, der MDP-/FQE-Apparat nicht |
| *How Much Screening Buys a Second?* | existiert, [arXiv 2610.21510](https://arxiv.org/abs/2610.21510) | **Begrifflich** — Krypto-AMM-Pools, nicht Turbos; die Kerneinsicht überträgt sich invertiert |
| Potiron — *First passage time … local martingales* | im Auftrag mit DOI genannt | implementiert wurde die **klassische** First-Passage-Theorie, nicht Potirons spezifischer Beitrag |
| Gozzo — *Adaptive Multilevel Splitting* | existiert, [arXiv 2510.23461](https://arxiv.org/abs/2510.23461), CRAN `amsSim` | **Nein, gemessen** — siehe Phase 4 |
| Baule & Jensen — *Fading Attention … German Structured Products* | forthcoming *Journal of Futures Markets*; Vorversion EFMA 2022 | **Schwach** — Discount-Zertifikate, nicht Turbos |
| Wessel et al. — *Enforcing tail calibration* | im Auftrag mit DOI genannt | bereits in W10 adressiert, dort `NO PROMOTION` |
| Koning & van Meer — *Anytime validity is free* | im Auftrag mit DOI genannt | gemessen, siehe `mathematical_validation_2026.md` |

Die restlichen vier (PC-Audit, Wasserstein, Shaer et al., Zhang et al.) habe
ich **nicht** geprüft und auch nicht umgesetzt — siehe Tabelle 3.

---

## Tabelle 1 — Forschungsbewertung

| Forschungsgebiet | Test durchgeführt? | Ergebnis | Nutzen für TurboEdge | Empfehlung |
|---|---|---|---|---|
| Fehlende Labels und Quotes | **Ja**, auf echten Daten | 3.670 Einträge, **0 aufgelöst, 0 überfällig** | Instrument steht; Verzerrung heute nicht messbar | `INSUFFICIENT_DATA` |
| Executable Quotes | **Ja**, 125.296 Snapshots | Günstigkeit ↔ Quote-Alter: **+0,1852 [+0,1234, +0,2450]** | **Ja** — scheinbare Preisvorteile sind teilweise Veraltung | `WEITER VALIDIEREN` |
| Analytische KO-Modelle | **Ja**, 128 Zellen, echte Hoch/Tiefs | Simulation besser als analytisch (MAE 0,0140 vs 0,0175; CI 67 % vs 36 %) | **Diagnostisch ja**, als Marktmodell nein | `NUR RESEARCH` |
| Rare-Event-Simulation | **Ja**, rechnerisch | MC-Fehler ist nur **0,23–0,31×** des Modellfehlers | **Nein** | `VERWERFEN` |
| Tail Calibration | Ja (W10, 150.000 OOS) | isotonic ΔBrier −0,0002; Cox-Slope 0,840 → 0,584 | **Nein** | `VERWERFEN` |
| Modellselektions-Robustheit | Nein | — | — | `OFFEN` |
| Sequentielle Statistik | Ja (Vorsitzung) | FPR 0,0143 i.i.d. vs **0,4923** überlappend | **Nein** — gefährlicher, nicht sicherer | `WEITER TESTEN` |
| Robuste Optimierung (Wasserstein) | Nein | — | — | `OFFEN` |
| Marktregime / Schockansteckung | Nein | — | — | `OFFEN` |
| Emittentenrisiko | **Ja**, auf echten Daten | **BNP 3.652 von 3.670 Einträgen (99,5 %)** | **Nein** — Vergleich im Ledger unmöglich | `INSUFFICIENT_DATA` |
| Forschungsbudget-Optimierung | Nein | vorhandenes VoI-System nicht ersetzt | — | `OFFEN` |
| Komplexe Volatilitätsmodelle | Nein | — | — | `OFFEN` |

---

## Die fünf Befunde mit Messwerten

### 1. Scheinbare Günstigkeit korreliert mit Quote-Alter

Über 125.296 archivierte Snapshots, gruppiert nach Basiswert, Richtung,
Abrufminute und Barriereabstand, nur Gruppen mit mindestens zwei
Emittenten: die gruppeninterne Rangkorrelation zwischen Quote-Alter und
Günstigkeit gegenüber dem Gruppenmedian beträgt

> **+0,1852, 95 %-KI [+0,1234, +0,2450]**, über **184 Gruppen** mit 10.018
> Produkten (Bootstrap über Gruppen, Lauf 40232072811).

Das Intervall schließt die Null aus. Innerhalb vergleichbarer Produkte
verschiedener Emittenten, zur selben Minute gesehen, bei gleichem
Barriereabstand, sind die **günstiger aussehenden systematisch die mit dem
älteren Quote**. Das ist das Adverse-Selection-Muster, auf das Phase 2
zielt: wer das billigste Produkt wählt, wählt teilweise das am längsten
nicht aktualisierte.

Nüchterne Einordnung, dreifach:

- Es ist eine **Assoziation** auf Beobachtungsdaten, keine Ursache.
- Sie sagt **nicht**, wie viel des Vorteils bei Ausführung verschwindet.
  Dafür bräuchte es echte Ausführungsdaten; eine modellierte
  Fill-Wahrscheinlichkeit wäre eine Vermutung im Gewand einer Messung.
- Die gepoolte Korrelation beträgt +0,2952, also war etwa ein Drittel
  Zwischen-Gruppen-Struktur. Ein Test zeigt die Falle direkt: 60
  gruppenintern unkorrelierte Gruppen, deren Mittelwerte zusammenlaufen,
  ergeben gepoolt +0,9 und gruppenintern unter 0,2.

**Zwei eigene Fehler auf dem Weg dorthin**, beide hätten die Messung
lautlos zerstört:

- Gruppierung über den Hebel-Bucket aus `product_pairs` konditioniert auf
  den Ask, weil `pricing/intrinsic.leverage` daraus berechnet wird.
  Gemessen: 40 vergleichbare Produkte in 7 Gruppen zersplittert, eine
  gepflanzte Korrelation von ~0,9 auf 0,14 zusammengedrückt.
- `observation_time − quote_timestamp` ist **identisch null**, weil
  `adapters/issuer_feeds.py:1082` `observation_time` auf `quote_timestamp`
  setzt. Ich hatte ein Feld von sich selbst abgezogen und das Ergebnis
  zunächst dem Archiv zugeschrieben. Richtig ist `retrieved_at`, laut
  Schema „purely when this process fetched it".

### 2. Die analytische Referenz ist nicht besser, sondern symmetrisch

Volles Gitter: 4 Basiswerte × 4 Horizonte × 4 Barriereabstände × 2
Richtungen = **128 Zellen**, je ~772 Fenster, echte Tageshochs/-tiefs seit
2010.

| Verfahren | MAE | im realisierten 95 %-KI | überschätzt |
|---|---:|---:|---:|
| Produktionssimulation | **0,01404** | **86/128 (67 %)** | 113/128 |
| analytisch `2Φ(−k)` | 0,01749 | 46/128 (36 %) | 38/128 |

**Das korrigiert meinen eigenen Befund vom selben Tag.** Aus den zwei
dokumentierten DAX-Zellen hatte ich geschlossen, die analytische Referenz
sei der Realität um Faktor ~2,6 näher. Über das Gitter ist das falsch, und
selbst auf k=2 beschränkt, aber über alle vier Basiswerte, führt die
Simulation. Der frühere Schluss hing an zwei Zellen eines Basiswerts.

Die Messung validiert sich zuerst selbst: bei DAX h=10 k=2 ergibt sie
realisiert 0,0751 long und 0,0324 short gegen die dokumentierten 0,072 und
0,036.

Der Grund ist strukturell. Die realisierte Long/Short-Trefferquote steigt
von 1,19 (k=1,5) auf 2,01 (k=3) — und sie tritt genau dort auf, wo die
Theorie sie erwartet: **S&P 500 2,75, DAX 2,10, EUR/USD 1,09, Gold 0,78.**
Aktienindizes zeigen den Leverage-Effekt, FX ist symmetrisch, Gold leicht
invertiert. Eine driftfreie Brown'sche Bewegung hat für Schiefe **keinen
Parameter**. Ein geschlossener Ausdruck ist das richtige Werkzeug, um
Numerik zu prüfen, und das falsche, um einen Index zu prognostizieren.

### 3. Die Überschätzung kommt aus einer Volatilitäts-Inkonsistenz

Verhältnis der realisierten Horizont-σ der simulierten Pfade zur
Volatilität, mit der die Barriere platziert wird:

| Basiswert | Standard | mit `target_sigma` |
|---|---:|---:|
| DAX | 1,358 | 1,089 |
| S&P 500 | 1,223 | 1,022 |
| Gold | 1,332 | 1,044 |
| EUR/USD | 1,707 | 1,610 |
| **Mittel** | **1,405** | 1,191 |

Der Bootstrap zieht aus 400 Tagen, die Barriere benutzt eine EWMA-Schätzung
mit ~16 Tagen Halbwertszeit, und Volatilität ist rechtsschief. Auf
Aktienindizes (64 Zellen) **halbiert** die σ-Angleichung den Fehler
(MAE 0,01224 → 0,00615) und hebt die Intervallabdeckung von 69 % auf 91 %
bei Nominalniveau 95 %.

Der Parameter `target_sigma` **existiert bereits** in `simulate_paths`; die
produktive Pipeline übergibt ihn nicht.

**Falsifikation und Grenze:** auf EUR/USD verschlechtert die Angleichung das
Ergebnis drastisch (0,0198 → 0,1553), das Ziel bleibt unerreichbar
(Verhältnis 1,610). Ob das am Modell liegt oder an den synthetischen
Tages-OHLC, die yfinance für FX liefert, ist **nicht** entschieden.

**Und die Richtung verlangt eine Betreiberentscheidung:** eine zu hohe
P(KO) macht das System zu **konservativ**. Sie zu korrigieren würde
ACTIONABLE leichter erreichbar machen. Das ist eine Lockerung mit
wirtschaftlicher Wirkung und durch diesen Auftrag nicht freigegeben.

### 4. Rare-Event-Verfahren lösen ein Problem, das hier nicht besteht

Monte-Carlo-Standardfehler bei 2.000 Pfaden gegen den gemessenen
**Modellfehler** derselben Zelle:

| k | p | MC-SE | Modell-MAE | MC/Modell |
|---|---:|---:|---:|---:|
| 1,5 | 0,1336 | 0,00761 | 0,02802 | **0,27** |
| 2,0 | 0,0455 | 0,00466 | 0,01496 | **0,31** |
| 2,5 | 0,0124 | 0,00248 | 0,00813 | **0,30** |
| 3,0 | 0,0027 | 0,00116 | 0,00504 | **0,23** |

Über das gesamte Betriebsgitter ist das Monte-Carlo-Rauschen nur 23–31 %
des systematischen Modellfehlers. Adaptive Multilevel Splitting ist
unverzerrt und bis 200× effizienter — aber es beschleunigt die **kleinere**
der beiden Fehlerquellen. Der Bereich, in dem es trägt (relativer SE > 1
erst ab k≈4, p≈6,3·10⁻⁵), entspricht Barrieren 15–30 % entfernt, also
Produkten mit Hebel 3–8, bei denen das KO-Risiko ohnehin nicht die bindende
Größe ist. Hochgehebelte Turbos sitzen bei **kleinem** k, wo der KO gar
nicht selten ist. `VERWERFEN`, mit Zahl statt Meinung.

### 5. Emittentenabdeckung macht die Emittentenfrage unbeantwortbar

Im Forward Ledger: **BNP Paribas 3.652, Goldman Sachs 7, UniCredit 6,
HSBC 5** — 99,5 % ein Emittent. Phase 10 ist auf dem Ledger nicht
beantwortbar, und zwar wegen Abdeckung, nicht wegen Historienlänge. Das
bestätigt `measured_results.md` §4.

Bemerkenswert: im **Snapshot-Archiv** existieren 218 emittentenübergreifende
Vergleichsgruppen. Die Vergleichbarkeit ist also vorhanden, sie landet nur
nicht in der Auswahl. Dass die Studie von Baule & Jensen Discount-
Zertifikate untersucht — mehrjährige, bondartige Ansprüche mit großer
Emittentenexposition —, macht die Übertragung auf Turbos, die Tage gehalten
werden und deren Wert klein gegen das Exposure ist, ohnehin schwach.

### 6. Fehlende Labels: das Instrument steht, die Verzerrung ist heute nicht messbar

Gemessen (Lauf 40232072811): **3.670 Forward-Beobachtungen, alle
`not_yet_due`. 0 aufgelöst, 0 überfällig, 0 mit angenommenem Wert.** Es geht
also nichts verloren; die Daten sind nur noch nicht reif. Früheste
Fälligkeit 2026-10-26.

`learning/labeler.py` imputiert nichts — ein fehlender Exit-Quote wird zum
letzten bekannten Bid abzüglich Spread, markiert als `ambiguous_path` mit
`NO_EXIT_QUOTE_CONSERVATIVE`, und ohne Rückfall bleibt der P&L `None`.

**Der statistische Kern, den der Auftrag meint:** dieser Rückfall ist
konservativ *relativ zum letzten Quote*. Hört ein Emittent auf zu
quotieren, **weil** der KO naht, ist der letzte bekannte Bid der Wert von
**vor** dem Verlust — der Rückfall ist dann optimistisch in genau den
Fällen, die über eine Strategie entscheiden. Das ist MNAR, und MNAR ist aus
den beobachteten Daten allein nicht identifizierbar.

Was diese Produktklasse rettet, ist ein struktureller Zufall: **der
Exitwert eines Turbos ist überwiegend Funktion von etwas, das nie fehlt.**
Der Basiswert ist beobachtbar, ob der Emittent quotiert oder nicht, und der
innere Wert ist `(spot − barrier) / ratio`. Statt einer scheinpräzisen
korrigierten Rendite klammert `exit_value_bounds` den Exit aus
beobachtbaren Größen unter ausgesprochenen Annahmen ein; die **Breite des
Intervalls ist die Antwort** auf „wie stark könnten fehlende Labels das
verzerren". Was an beiden Enden hält, kann Fehlen nicht umdrehen. Ein Test
sichert die Eigenschaft, die ein Intervall erst zu einer Schranke macht:
eine gelockerte Annahme kann es nie verengen.

### 7. Ein Nebenbefund zur Zeitstempel-Semantik

Auf dem Ledger ist das Quote-Alter zur Entscheidung bei **3.123 von 3.670
Einträgen negativ**, Median **−19,1 s**, Maximum +5.587 s. Der auf dem
Eintrag vermerkte Quote-Zeitstempel liegt also meist **nach** der
protokollierten `prediction_time`. Das ist kein Look-ahead im
Feature-Sinn — Features werden getrennt gegen `available_at` geprüft —,
aber `prediction_time` ist damit keine obere Schranke für die Information
im Eintrittsquote. Für jede Ausführbarkeitsauswertung auf Ledger-Basis muss
das zuerst geklärt werden; die Archiv-Auswertung in Befund 1 umgeht es, weil
sie `retrieved_at` benutzt.

---

## Tabelle 2 — Tatsächlich implementierte Verbesserungen

| Änderung | Neue Tests | Vergleich zur Baseline | Messbare Verbesserung | Neue Komplexität | Commit | CI | Merge |
|---|---|---|---|---|---|---|---|
| `research/ko_reference.py` — analytische Referenz, σ-Diagnose, Platzierungsbias, Block-Bootstrap | 76 | es gab keine unabhängige Vergleichsgröße auf dem P(KO)-Pfad | 3 Befunde, einer korrigiert einen früheren Fehlschluss | 1 Modul, kein Produktionsimport | `ca1ce51` | grün | `main` (PR #3) |
| `research/data_audit.py` — 8-Zustands-Zensus, Exitwert-Schranken, Quote-Alter, Staleness-Assoziation | 57 | es gab keine Missingness- oder Ausführbarkeitsdiagnostik | Phase-2-Befund mit KI, das die Null ausschließt | 1 Modul, nur CLI-Import | `7e5f9d9` | grün | `main` (PR #3) |
| `research data-audit` + `data-audit.yml` | — | — | macht die Messung auf echten Daten überhaupt möglich | 1 Workflow, `workflow_dispatch`, packt nie State | `7e5f9d9` | grün | `main` |
| Entartungsdiagnose (`degenerate_side`) | 2 | ein NaN galt als interpretierbar | verhindert, dass ein konstanter Input als Null-Ergebnis liest | — | `2edd916` | grün | `main` (PR #4) |
| Quote-Alter aus `retrieved_at` | — | Messgröße war identisch null | machte Befund 1 erst möglich | — | — | grün | `main` (PR #5) |
| Gruppeninterne Assoziation + Cluster-Bootstrap | 7 | gepoolt +0,2952 | +0,1852 [+0,1234,+0,2450], ein Drittel war Artefakt | 1 Funktion | — | grün | `main` (PR #6) |
| `first_passage`-Importwächter auf Produktionsmodule eingegrenzt | — | verbot auch Research-Wiederverwendung | erlaubt eine Formel an einer Stelle | — | `ca1ce51` | grün | `main` |

## Tabelle 3 — Nicht umgesetzte Ansätze

| Ansatz | Warum nicht |
|---|---|
| Challenger D (tail-gewichtete Kalibrierung) | W10 hat die einfachere Frage über 150.000 OOS-Beobachtungen beantwortet; liegt die Ursache stromaufwärts (Befund 3), kalibriert man ein Symptom |
| Adaptive Multilevel Splitting / Importance Sampling | **gemessen nutzlos** hier: MC-Fehler 0,23–0,31× des Modellfehlers |
| Wasserstein-robuster EV (Phase 6) | nicht begonnen. Die Vorfrage wäre, ob der bestehende LCB schon robust genug wirkt; das ist eine eigene Messung |
| PC-Audit / Modellselektions-Robustheit (Phase 7) | `walkforward_results` enthält zu wenig, um Rangstabilität über Regime zu prüfen; wäre heute eine Rechnung über fast leere Mengen |
| Conformal Test Martingales, Hawkes-Sprünge (Phase 9) | nicht begonnen. Priorität laut Auftrag MITTEL; die 40-%-Blöcke zuerst |
| Forschungsbudget-Optimierung (Phase 11) | der Auftrag sagt selbst: kein zweites Meta-System, wenn das vorhandene VoI-System reicht. Dass es nicht reicht, ist nicht gezeigt |
| Rough Volatility, Heston, Double-Exponential-Jumps (Phase 12) | nicht begonnen. Vor jeder Kalibrierung stünde die Identifizierbarkeitsfrage, und Befund 3 zeigt, dass schon die **Konsistenz** der vorhandenen Volatilität ungelöst ist |
| Produktive σ-Angleichung | Modellwechsel, durch diesen Auftrag nicht freigegeben, und die Wirkungsrichtung lockert ein Gate |

---

## Die zehn Abschlussfragen

**1. Welche neuen Erkenntnisse haben echten Mehrwert erbracht?**
Zwei. Das Adverse-Selection-Muster auf Quote-Alter (+0,1852 [+0,1234,
+0,2450]) ist ein messbarer, bisher unbekannter Mechanismus hinter
scheinbaren Preisvorteilen. Und die Einsicht, dass der Exitwert eines
Turbos aus dem Basiswert eingeklammert werden kann, macht die
MNAR-Frage ohne unprüfbare Annahmen behandelbar.

**2. Welche bisherigen Annahmen sind fragwürdig?**
Drei. Dass die Pfadsimulation die Volatilität trägt, mit der die Barriere
platziert wird (Faktor 1,405). Dass ein Prozentabstand ein Log-Abstand ist
(1,39× bei k=2, 10,9× bei k=4). Dass `prediction_time` eine obere Schranke
für die Information im Eintrittsquote ist (85 % negativ).

**3. Gibt es systematische Fehler in der KO-Risikobewertung?**
Ja, einen, und er ist jetzt lokalisiert: die Simulation überstreut um
~40 % gegenüber der Barriere-Volatilität und liest in 113 von 128 Zellen zu
hoch. Die Richtung ist konservativ, nicht gefährlich. Nicht bestätigt hat
sich, dass die Diskretisierung schuld ist (falsches Vorzeichen) oder die
Brownian-Idealisierung (Befund 2).

**4. Können fehlende Labels die Ergebnisse verzerren?**
Heute nicht, weil es keine gibt: 0 von 3.670 aufgelöst, 0 überfällig. Sobald
Labels reifen, ist der gefährliche Kanal benannt und das Instrument steht.

**5. Existieren echte, ausführbare Preisunterschiede zwischen Emittenten?**
Nicht nachgewiesen. Gezeigt ist das Gegenteil der naiven Hoffnung: ein Teil
des sichtbaren Unterschieds ist Quote-Alter. Ausführbarkeit selbst ist ohne
Ausführungsdaten nicht messbar, und im Ledger macht die 99,5-prozentige
BNP-Dominanz den Vergleich ohnehin unmöglich.

**6. Sind die derzeitigen Modellentscheidungen statistisch robust?**
Nicht geprüft (Phase 7 offen). Die bestehende Entscheidung — `NO TRADE` ohne
bestätigtes Alpha — ist durch diese Untersuchung unangetastet und bleibt die
richtige.

**7. Welcher Ansatz verbessert die wirtschaftliche Bewertung am deutlichsten?**
Die σ-Angleichung: halbierter Fehler und 91 % statt 69 % Intervallabdeckung
auf Aktienindizes, mit einem Parameter, der schon existiert. Sie ist
trotzdem nicht freigegeben, weil sie auf FX scheitert und ein Gate lockert.

**8. Welche Erweiterungen wären unnötige Komplexität?**
Rare-Event-Verfahren (gemessen: lösen die kleinere Fehlerquelle).
Tail-gewichtete Kalibratoren vor der Ursachenklärung. Sequentielle Tests auf
überlappenden Beobachtungen. Ein zweites Meta-System für die
Forschungspriorisierung.

**9. Was sollte als Nächstes experimentell geprüft werden?**
In dieser Reihenfolge: (a) die σ-Konsistenz auf FX klären — Modell oder
yfinance-OHLC; (b) die `prediction_time`-Semantik korrigieren, sonst ist
Ausführbarkeit auf Ledger-Basis dauerhaft unmessbar; (c) ab 2026-10-26 den
Zensus wöchentlich laufen lassen und die Kovariatenkontraste beobachten,
**bevor** Forward-Ergebnisse interpretiert werden.

**10. Ist durch irgendeine neue Methode wirtschaftliches Alpha nachgewiesen?**
**Nein.** Kein einziges Verfahren in dieser Untersuchung hat einen positiven
Netto-Erwartungswert nach Kosten gezeigt. Der stärkste Befund — das
Quote-Alter — zeigt in die entgegengesetzte Richtung: er erklärt, warum ein
Teil des bisher für Vorteil gehaltenen Unterschieds keiner ist. `NO TRADE`
bleibt richtig.

---

*Messläufe: [40229493270](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40229493270),
[40230382857](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40230382857),
[40231198366](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40231198366),
[40232072811](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40232072811).
Gitter-Seeds 20260125 (Simulation), 11/23 (Bootstrap).*
