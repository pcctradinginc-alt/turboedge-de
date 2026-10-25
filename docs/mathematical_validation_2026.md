# Mathematische Validierung 2026 — Messergebnisse

**Branch `research/mathematical-validation-2026`. Nichtinvasiv: kein
Produktionsmodell geändert, kein Risikolimit, kein ACTIONABLE-Kriterium,
kein Promotion-Gate. Keine der hier eingeführten Module wird von
`src/turboedge` außerhalb von `research/` importiert; zwei Tests erzwingen
das.**

## Vorbemerkung zur Literatur

Der Auftrag nennt Arbeiten von 2026 mit DOI. Mein Wissensstand endet im
Mai 2026, ich habe diese Texte nicht gelesen und behaupte nicht, ihre
spezifischen Beiträge geprüft zu haben. Was ich implementiert und gemessen
habe, ist die **klassische Mathematik, auf der sie aufbauen** —
Reflexionsprinzip, First-Passage-Verteilungen mit Drift, Brownian Bridge,
Test-Martingale und Villes Ungleichung. Diese Resultate sind exakt
nachrechenbar, und genau das ist unten geschehen.

---

## Ergebnistabelle

| Methode | Mathematisch korrekt? | Mit echten Daten getestet? | Verbesserung gegenüber Baseline? | Empfehlung |
|---|---|---|---|---|
| Analytische First-Passage-Modelle | **Ja**, exakt gegen Reflexionsprinzip | Indirekt, gegen dokumentierte W5-Messwerte | **Ja** — näher an der realisierten Quote als die Simulation | **WEITER TESTEN** (volle W5-Neumessung) |
| Brownian-Bridge-Korrektur | **Ja**, Restbias 0,001 statt 0,045 | Nein | **Nein** — verschiebt in die falsche Richtung | **VERWERFEN für Produktion**, als Benchmark behalten |
| Tail Calibration | n/a | **Ja**, W10 mit 150.000 OOS-Beobachtungen | **Nein** (isotonic ΔBrier −0,0002, Cox-Slope 0,840→0,584) | **VERWERFEN**, bis die einfache Frage beantwortet ist |
| Anytime-valid Tests | **Ja**, Fehlerrate 1,4 % ≤ 5 % bei Unabhängigkeit | Nein (0 verwertbare Forward-Labels) | **Nein** — 49 % Fehlerrate bei überlappenden Fenstern | **WEITER TESTEN** (~5 Jahre Daten nötig) |
| Conformal Drift Detection | nicht geprüft | Nein | nicht geprüft | **OFFEN** — siehe „Nicht untersucht" |
| Prognoseintervall-Backtesting | nicht geprüft | Nein | nicht geprüft | **OFFEN** |
| Jump-/Volatility-Modelle | nicht geprüft | Nein | nicht geprüft | **OFFEN** (niedrigste Priorität, so vorgesehen) |

---

## 1. Analytische First-Passage-Referenz

**Publikation / Annahmen.** Klassische First-Passage-Theorie für
Brown'sche Bewegung mit Drift. Für `X_t = m t + σ W_t` ab 0 mit Barriere
bei `−a`, `a > 0`:

    P(min X ≤ −a) = Φ((−a − mT)/(σ√T)) + exp(−2ma/σ²) · Φ((−a + mT)/(σ√T))

Für geometrische Brown'sche Bewegung dasselbe auf `ln(S_t/S_0)` mit
`m = μ − σ²/2` und `a = ln(S_0/B)`.

**Implementierung.** `src/turboedge/research/first_passage.py`. Neu, weil
das Repository auf dem P(KO)-Pfad **keine** analytische First-Passage-Formel
hatte: `simulation/barrier.brownian_bridge_hit_probability` existiert, wird
aber ausschließlich von `positions/reevaluate.py` für das Restrisiko
offener Positionen benutzt, nie von der Simulation.

**Testergebnisse.** Bei Drift 0 reduziert sich die Formel exakt auf das
Reflexionsprinzip `2Φ(−a/(σ√T))` — Abweichung ≤ 7·10⁻¹⁸ über die geprüften
Fälle, als Gleichheit getestet, nicht mit Toleranz. Invarianzprobe: bei
Barriere `k·σ·√h` ergibt die Formel konstant `2Φ(−k)` über σ ∈ {0,008,
0,02, 0,04} und h ∈ {3, 10, 30} — neunmal exakt 0,045500 für k=2.
Konvergenz: eine Simulation mit 4.000 Stützstellen trifft den analytischen
Wert innerhalb von 4 Standardfehlern.

**Unsicherheit.** Die Formel ist exakt; der einzige Fehler ist
Gleitkomma-Rundung. Die zweite Komponente wird im Logarithmus addiert, weil
`exp(−2ma/σ²)` und `Φ(…)` getrennt berechnet überlaufen, bevor sie sich
kürzen.

**Rechenaufwand.** Zwei Normalverteilungsaufrufe, ~2 µs pro Auswertung.

**Einschränkung.** Ein analytisches Brownian-Resultat belegt numerische
Korrektheit unter Brownian-Annahmen. Es belegt **nicht**, dass echte Kurse
sich so verhalten. `simulation/bootstrap.py` zieht jeden simulierten Tag
als `(gap, intraday, hi, lo)` aus **echten historischen Tagen** — die
Intraday-Spanne ist empirisch, nicht Brownian. Eine Abweichung zwischen
beiden ist daher kein Fehler in einem von beiden.

**Wirtschaftliche Relevanz.** Siehe Abschnitt 3: am handelsrelevanten
Barriereabstand liegt die analytische Referenz **näher an der realisierten
Trefferquote** als die Simulation.

---

## 2. Brownian-Bridge-Korrektur und der Diskretisierungsbias

**Hypothese (vorab festgelegt).** Eine Brownian-Bridge-Korrektur reduziert
den numerischen Fehler der geschätzten Barrierewahrscheinlichkeit gegenüber
der diskreten Simulation.

**Falsifikationskriterium (vorab).** Die Hypothese gilt als gescheitert,
wenn der Restfehler nach Korrektur nicht mindestens um den Faktor 10 kleiner
ist als vorher, oder wenn die Korrektur die empirische Kalibrierung
verschlechtert.

**Messung** (16 Zellen: Abstand ∈ {2, 5, 10, 20 %}, σ ∈ {0,01, 0,02}, h ∈
{3, 14} Tage, 400.000 Pfade, Seed 7, driftfrei):

| | mittlerer Bias | max. Bias |
|---|---:|---:|
| nur Tagesschluss beobachtet | 0,04510 | 0,18419 |
| mit Bridge-Korrektur | **0,00009** | **0,00098** |

Die Korrektur erfüllt ihr numerisches Kriterium um mehr als zwei
Größenordnungen.

**Was das bestehende Design schon leistet.** `first_hit_index` prüft nicht
nur den Schluss, sondern Eröffnung **und** Tagesextremum. Gemessen über 12
Zellen mit 20 Teilschritten pro Tag:

| | mittlerer Bias | max. Bias |
|---|---:|---:|
| nur Tagesschluss | 0,06066 | — |
| Eröffnung + Tagestief (= heutiges Design) | **0,01510** | 0,04978 |

Das bestehende Design entfernt also bereits **rund 75 %** des
Diskretisierungsbias. Der Rest beträgt im Mittel 1,5 Prozentpunkte, am
größten (5 pp) bei 2 % Abstand und hoher Volatilität.

**Entscheidung: VERWERFEN für die Produktion.** Nicht weil die Mathematik
falsch wäre, sondern wegen des **Vorzeichens**. Diskrete Beobachtung kann
Durchbrüche nur übersehen, nie erfinden — sie **unterschätzt** also. Das
Repository misst aber eine **Überschätzung** von 40–135 % relativ
(`docs/measured_results.md` §3). Beide Fehler zeigen in entgegengesetzte
Richtungen: die Bridge-Korrektur würde die simulierte P(KO) erhöhen und die
empirische Fehlkalibrierung damit **verschlimmern**.

Daraus folgt zugleich eine Ursachenausschließung: die beobachtete
Überschätzung kann **nicht** von der Diskretisierung kommen.

---

## 3. Tail Calibration und die Ursache der Überschätzung

**Baseline A vs. Challenger C war bereits durchgeführt.** Workstream W10
(`docs/measured_results.md` §3.1) verglich walk-forward und
out-of-sample über **150.000 Beobachtungen**:

| Methode | Brier | Cox-Intercept | Cox-Slope | ECE | abs. Kalibrierungsfehler |
|---|---:|---:|---:|---:|---:|
| raw | 0,1053 | −0,299 | 0,840 | 0,0160 | 0,2347 |
| isotonic | 0,1051 | −0,472 | **0,584** | 0,0130 | 0,2004 |
| platt | 0,1062 | +0,384 | 1,204 | 0,0349 | 0,2247 |

Ergebnis damals: **NO PROMOTION**. Isotonic verbessert den Brier Score um
0,0002 — numerisch belanglos — und verschlechtert den Cox-Slope von 0,840
auf 0,584, also deutlich **weiter** von 1 entfernt. Platt ist in Brier und
ECE schlechter.

**Baseline B, neu geliefert.** Mit der analytischen Referenz aus
Abschnitt 1 am dokumentierten W5-Messpunkt (DAX, h=10, k=2σ; analytisch
`2Φ(−2) = 0,0455`, invariant in σ und h):

| Seite | simuliert | realisiert | analytisch | \|sim−real\| | \|analyt−real\| |
|---|---:|---:|---:|---:|---:|
| long | 0,117 | 0,072 | 0,0455 | 0,045 | **0,026** |
| short | 0,084 | 0,036 | 0,0455 | 0,048 | **0,010** |

**Die naive analytische Brownian-Referenz ist auf beiden Seiten näher an
der realisierten Trefferquote als die Produktionssimulation** — mittlerer
absoluter Fehler 0,018 gegen 0,047, also etwa Faktor 2,6.

Das lokalisiert die Überschätzung in der **Pfaderzeugung** (gebootstrappte,
skalierte historische Tagesspannen), nicht in der Brownian-Idealisierung.
Eine driftfreie Brown'sche Bewegung mit derselben Volatilität berührt die
Barriere seltener als die Bootstrap-Simulation — und näher an der
Wirklichkeit.

**Einschränkung, ausdrücklich.** Das sind **zwei Zellen**, aus
dokumentierten Werten übernommen, keine eigene Neumessung über das volle
W5-Gitter. Es ist ein starker Hinweis, der eine vollständige Neumessung
rechtfertigt — **keine Grundlage, ein Modell zu wechseln**. Eine
Neumessung bräuchte yfinance-Zugriff und mehrere Stunden Rechenzeit.

**Challenger D (tail-gewichtete Kalibrierung): nicht implementiert.** Der
Auftrag selbst sagt, komplexere Modelle nur zu ergänzen, wenn die
einfacheren nachvollziehbar an ihre Grenzen stoßen. Hier ist die einfachere
Frage nicht erschöpft, sondern unbeantwortet: wenn die Ursache in der
Pfaderzeugung liegt, würde eine Tail-Gewichtung einen Kalibrator auf einen
Fehler trainieren, dessen Quelle unverändert bleibt. Zuerst die Ursache,
dann die Kalibrierung.

---

## 4. Anytime-valide sequentielle Tests

**Publikation / Annahmen.** Test-Martingale und Villes Ungleichung: für
einen nichtnegativen Prozess mit `E[E_t] ≤ 1` unter der Nullhypothese gilt
`P(sup_t E_t ≥ 1/α) ≤ α` — bei **jeder** Stoppregel, auch einer
datenabhängigen. Der benutzte Wett-E-Prozess ist
`E_t = ∏(1 + λ_i X_i)` mit prädiktierbarem `λ_i ∈ [0, c]`.

Die Voraussetzung ist `E[X_i | F_{i−1}] ≤ 0` — eine **bedingte**
Mittelwertbedingung in der Filtration, in der die Beobachtungen eintreffen.

**Implementierung.** `src/turboedge/research/sequential.py`. Der Einsatz
`λ_i` wird ausschließlich aus Beobachtungen **vor** `i` berechnet; ein Test
prüft verhaltensseitig, dass eine Störung an Stelle `k` keinen E-Wert vor
`k` verändert.

**Testergebnisse** (α = 0,05, n = 300, 4.000 Wiederholungen):

| Datenmodell | Falsch-Positiv-Rate | Urteil |
|---|---:|---|
| i.i.d. (Voraussetzung erfüllt) | **0,0143** ± 0,0019 | hält |
| überlappende 14-Tage-Fenster | **0,4923** ± 0,0079 | **verletzt, 10-fach** |
| AR(1), φ = 0,9 | **0,5250** ± 0,0079 | **verletzt, 10-fach** |

Power-Kontrolle auf unabhängigen Daten (Effekt +0,10, σ = 0,25): n=22 →
0,00; n=50 → 0,08; n=100 → 0,79; n=300 → 1,00. Die Implementierung
funktioniert also; das Versagen oben ist eine Eigenschaft des Datenmodells,
nicht des Codes.

**Reparatur und ihr Preis.** Nicht überlappende Blöcke (jede 14.
Beobachtung) stellen die Gültigkeit her — Fehlerrate 0,0000 — reduzieren
die Stichprobe aber von 300 auf **22**, und bei 22 Beobachtungen ist die
Power **0,00**.

**Entscheidung: WEITER TESTEN, nicht einsetzen.** Die Variante, deren
Voraussetzungen gelten, hat bei realistischen Stichproben keine Power; die
Variante mit Power ist bei 49 % Fehlerrate ungültig. Für brauchbare Power
braucht es ~100 **unabhängige** Beobachtungen, bei 14-Tage-Fenstern also
etwa **fünf Jahre** Forward-Daten. Das bestehende feste,
vorregistrierte Design bleibt damit die richtige Wahl — und der Auftrag
verlangt ausdrücklich, Gültigkeit bei unbekannter serieller Abhängigkeit
nicht zu behaupten.

`docs/measured_results.md` nennt unabhängig davon, dass bei 14 Tagen neun
von zehn Forward-Fenstern überlappen. Das ist genau das Datenmodell, unter
dem die Messung oben scheitert.

---

## Nicht untersucht, mit Begründung

**Conformal Drift Detection (Phase 5), Prognoseintervall-Backtesting
(Phase 6), Jump-/Volatility-Modelle (Phase 7).** Der Auftrag legt die
Reihenfolge fest — KO-Benchmark, dann Extremwertkalibrierung, dann
sequentielle Statistik — und weist Phase 7 ausdrücklich niedrigere
Priorität zu. Die drei vorrangigen Phasen sind mit Messwerten
abgeschlossen; diese drei habe ich **nicht** angefasst, statt sie flach zu
behandeln.

Für Phase 6 kommt ein sachlicher Grund hinzu: eine Abdeckungsprüfung
braucht Prognosen **und** eingetretene Ergebnisse. Der Forward-Ledger hält
3.670 offene Einträge und **null verwertbare Labels** (gemessen,
Produktionslauf 40186945583); der früheste Fälligkeitstermin ist der
26.10.2026. Eine Abdeckungsmessung wäre heute nicht fair durchführbar.

---

## Abschließende Bewertung

**A. Ist die bisherige Knock-out-Simulation mathematisch korrekt?**
Weitgehend. Der Barrierescan prüft Eröffnung und Tagesextremum und entfernt
damit rund 75 % des Diskretisierungsbias; es bleibt eine
**Unterschätzung** von im Mittel 1,5 Prozentpunkten (bis 5 pp bei 2 %
Abstand). Die Bridge-Formel im Repo ist korrekt implementiert.

**B. Wo bestehen systematische Modellabweichungen?** In der Pfaderzeugung.
Die Simulation überschätzt P(KO) um 40–135 % relativ, während eine
driftfreie Brown'sche Bewegung mit derselben Volatilität der realisierten
Quote um Faktor ~2,6 näher kommt. Die Abweichung stammt nicht aus der
Diskretisierung (falsches Vorzeichen) und nicht aus der
Brownian-Idealisierung (die ist besser), sondern aus dem Bootstrap der
historischen Tagesspannen.

**C. Welche neuen Methoden verbessern die Risikobewertung tatsächlich?**
Keine der geprüften, in der Produktion. Die analytische Referenz verbessert
die **Diagnose** erheblich, weil sie erstmals eine unabhängige
Vergleichsgröße liefert.

**D. Können extreme Knock-out-Risiken besser kalibriert werden?** Nicht
mit den geprüften Kalibratoren — W10 zeigt das über 150.000 Beobachtungen.
Solange die Ursache in der Pfaderzeugung liegt, kalibriert man ein Symptom.

**E. Ist das autonome Research-System statistisch ausreichend gegen
Zufallstreffer geschützt?** Ja, durch das bestehende Design. Ein
sequentieller Test wäre hier **gefährlicher**, nicht sicherer: auf dem
tatsächlichen Datenmodell liegt seine Fehlerrate bei 49 % statt 5 %.

**F. Welche Verbesserungen besitzen einen nachweisbaren wirtschaftlichen
Nutzen?** Keine, heute. Der einzige Kandidat — die Pfaderzeugung zu
korrigieren — ist durch zwei Zellen begründet und braucht eine vollständige
Neumessung, bevor er angefasst wird.

**G. Welche Methoden sollten bewusst nicht integriert werden?** Die
Bridge-Korrektur in die Produktions-P(KO) (richtiges Vorzeichen des
Fehlers, falsche Richtung der Wirkung). Tail-gewichtete Kalibrierung, vor
der Ursachenklärung. Sequentielle Tests auf überlappenden Beobachtungen.

---

*Erstellt 2026-10-25. Alle Zahlen reproduzierbar mit festen Seeds; die
Testsuite prüft Richtung und Größenordnung, nicht die dritte Dezimale.*
