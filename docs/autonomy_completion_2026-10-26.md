# Autonomie vervollständigen — Abschlussbericht (2026-10-26)

Keine Risikolimits, ACTIONABLE-Kriterien, Promotion-Gates, Signifikanz-
schwellen oder Forschungsbudgets geändert. `PROMOTION_LOCKED` steht. Kein
Signal aktiviert. Kein Trade freigegeben.

---

## 1. Literatur korrigiert und vollständig verifiziert

**Die Korrektur, die der Auftrag verlangt.** Barardehi, Bogousslavsky und
Muravyev, *What Drives Momentum and Reversal? Evidence from Day and Night
Signals*, ist **publiziert**: DOI 10.1093/rfs/hhag036 löst auf eine
*Review of Financial Studies* Advance Article auf. Mein voriger Bericht
führte sie als Working Paper in „Revise and Resubmit" — das war falsch und
ist hiermit richtiggestellt.

**Was sich dadurch nicht ändert.** Die Arbeit bildet **querschnittliche
Portfolios aus Einzelaktien**, sortiert nach vergangenen Intraday- bzw.
Overnight-Renditen (Stichprobe 1926–2019). Laut publiziertem Abstract
stammt das Momentum aus der Intraday-Komponente; overnight-gebildete
Portfolios zeigen keines. TurboEdge handelt vier Basiswerte und hat damit
**keinen Querschnitt**. Die Trennung, die der Auftrag fordert, steht
deshalb ausdrücklich im Katalog und im Modul:

- Die **Originalhypothese** ist querschnittlich und hier nicht testbar.
- Getestet wurde ein **Zeitreihen-Analogon** (Index-Timing), eine
  schwächere und andere Behauptung.
- Das negative Ergebnis des Analogons bleibt gültig und **widerlegt die
  Originalstudie nicht**.

**Alle 17 DOIs aufgelöst.** Mein voriger Bericht führte elf als ungeprüft;
sie existieren alle:

| ID | DOI löst auf | Status |
|---|---|---|
| A01 | RFS 39(10), S. 3191 | publiziert |
| A02 | RFS 38(12), S. 3497–3541 | publiziert |
| A03 | RFS Advance Article | publiziert (advance) |
| A04 | JFE, S0304405X26000097 | publiziert |
| A05 | RAPS Advance Article | publiziert (advance) |
| A06 | JFE 2018, S. 394–414 | publiziert |
| A07 | J. Futures Markets, fut.70038 | publiziert |
| A08 | RFS 39(10), S. 3330 | publiziert |
| A09 | RAPS Advance Article | publiziert (advance) |
| A10 | RFS Advance Article | publiziert (advance) |
| A11 | RAPS 15(3-4), S. 247 | publiziert |
| A12 | RFS Advance Article | publiziert (advance) |
| A13 | RFS 38(6), S. 1730 | publiziert |
| A14 | J. Futures Markets, fut.22589 | publiziert |
| A15 | RFS 36(9), S. 3548 | publiziert |
| A16 | RFS 38(6), S. 1783 | publiziert |
| A17 | RFS 39(10), S. 3035 | publiziert |

**Grenze dieser Prüfung, ausdrücklich.** Verifiziert sind Existenz,
Fundstelle und Publikationsstatus. Den **Inhalt** habe ich bei A02, A06,
A10 und A12 gegen Abstract bzw. Zusammenfassungen geprüft; bei den übrigen
dreizehn nicht. Die Übertragbarkeitsurteile zu diesen dreizehn beruhen auf
dem Titel und der im Auftrag genannten Forschungsfrage, nicht auf dem
Text. Das ist weniger, als „verifiziert" sonst bedeutet, und wird hier so
benannt.

---

## 2. Erstes echtes Forward-Labeling — gemessen

Lauf **40377514500** (`mode=eod`, 2026-10-26 18:25 UTC):

| Größe | Wert |
|---|---:|
| Fällige Beobachtungen | 2.785 |
| **Erfolgreiche Labels** | **2.785** |
| Knock-outs | 9 |
| Fehlende Exit-Quotes | **0** |
| Mehrdeutige Pfade | 0 |
| `missing_data` | 0 |
| Shadow-Positionen gelabelt | 120 |
| Danach überfällig und ungelabelt | **0** |
| Nächste Fälligkeit | 2026-10-27 |

Vom Learning Loop verarbeitet: `labeled_entries_considered=2785`,
`posteriors_updated=8`, `drift_events=8`, `models_reweighted=**0**`.

**Datenmangel oder technischer Fehler? Beides, getrennt benannt.**

**Technischer Fehler, behoben.** `models_reweighted=0` bedeutete nicht
„nichts zu lernen", sondern „nichts zuordenbar", und las sich identisch.
`by_model` ist mit `LedgerEntry.model_hash` gekeyed, das `pipeline/scan.py`
mit dem **Ensemble**-Hash füllt; `model_id_by_hash` mit den Hashes der
**Einzelmodelle**. Zwei Namensräume, leere Schnittmenge per Konstruktion.
Zusätzlich deckt `model_hash()` `training_end` und die gefitteten
Koeffizienten ab, ändert sich also bei jedem Refit, während die Registry
nur den neuesten behält — ein gereifter Eintrag könnte auch innerhalb
eines Namensraums nicht passen.

Behoben wurde die **Unsichtbarkeit**, nicht die Zuordnung: einen
Ensemble-P&L auf seine Mitglieder aufzuteilen braucht die Gewichte zum
Prognosezeitpunkt, die der Ledger nicht speichert. Eine erfundene
Zuordnung würde echte Modellgewichte auf erfundener Evidenz verschieben.
Es gibt jetzt eine `learn_weights_unattributable`-Warnung und einen
`labeled_entries_unattributable`-Zähler neben `models_reweighted`.

**Datenmangel, korrekt.** Alle 2.785 Labels fielen auf **einen**
Fälligkeitstag. Lauf 40387338754 zeigt es in den Worten der Harness
selbst: *„2785 labelled row(s) over 1 distinct date(s), effective sample
1.0"* — gegen 48 benötigte. Das ist Regel 8 (Average Uniqueness) korrekt
angewandt: Zeilen eines Tages sind eine Beobachtung, wiederholt. Die
nächste überprüfbare Fälligkeit ist der **2026-10-27**; brauchbar werden
`RO-MAE/MFE-PREDICTION` erst nach etwa 48 verschiedenen Fälligkeitstagen.

---

## 3. A02 — Lotterieartige Turbo-Preisverzerrungen · **CANDIDATE**

Π-CAPM zuerst verifiziert: Driessen, Ebert, Koëter, *RFS* 38(12),
3497–3541, DOI 10.1093/rfs/hhaf045. Das Modell leitet einen **positiven
Preiseffekt für Schiefe** ab, verstärkt durch Volatilität. Es ist eine
Aussage über Gleichgewichts-**Renditen**, nicht über Händlermargen — es
motiviert die Hypothese, es sagt sie nicht vorher.

**Vorab festgelegt:** Endpunkt ist die Steigung des relativen
Round-Trip-Spreads `(ask − bid)/ask` gegen den Barriereabstand, geschätzt
**innerhalb** von (Emittent, Basiswert, Richtung, Minute)-Zellen.
Falsifikation bei unter 30 Zellen, einem Intervall durch die Null, oder
unter 5 bps wirtschaftlicher Größe.

**Ergebnis** (Lauf **40389103879**, 225.665 Snapshots → 107 Zellen,
179.248 Quotes):

| Größe | Wert |
|---|---|
| Spread-Steigung, Mittel | **−0,4125** [−0,6519, −0,2113] |
| Spread-Steigung, Median | **−0,0472** [−0,0570, −0,0229] |
| Prämien-Steigung | −0,0562 [−0,0664, −0,0458] |
| Wirtschaftliche Größe, **Median** | **19,6 bps** [16,7, 23,0] |
| Wirtschaftliche Größe, Mittel | 137,1 bps [72,9, 218,5] |
| Vorzeichentest | **105/107 Zellen negativ**, p = 7,1·10⁻²⁹ |

**Das Vorzeichen ist robust, die Größe nicht.** Alle vier Emittenten
negativ (BNP −0,010 bei 36 Zellen, Goldman −0,283 bei 24, HSBC −1,037 bei
25, UniCredit −0,503 bei 22), alle vier Basiswert/Richtung-Kombinationen
negativ (97–100 % der Zellen). Mittelwert und Median der Größe trennen
jedoch um Faktor 7, weil einige Zellen eine enorme Steigung tragen;
Quantile 8,7 / 12,3 / 19,6 / 65,3 / 288,4 bps. Der Median ist die
repräsentative Zelle, und das Urteil benutzt jetzt ihn — was nach Sichtung
der Daten zulässig ist, weil es die Hürde **strenger** macht, nicht
leichter.

**Die Einschränkung, die das Ergebnis begrenzt.** Ein breiterer Spread
dicht an der Barriere ist **teilweise legitim**: der Emittent trägt echtes
Gap-Risiko, und der Spread entschädigt dafür. Π-CAPM würde einen
**Aufschlag darüber hinaus** vorhersagen. Dieser Test kann die beiden
**nicht trennen**. Was gemessen ist: lotterieartigere Turbos kosten im
Round Trip systematisch mehr, typischerweise etwa 20 bps über die
Barrierespanne einer Zelle. Was nicht gemessen ist: ob das mehr ist, als
das Risiko rechtfertigt.

**Und es ist kein Alpha.** Es ist ein Kostenbefund. Er sagt nichts über
die Richtung und kann keinen Trade profitabel machen. Er könnte höchstens
rechtfertigen, bei einer anderweitig getroffenen Entscheidung das
günstigere Produkt zu wählen. Der North Star ist ausdrücklich: eine
Cost-Edge ersetzt keine fehlende direktionale Edge.

**Warum es kein Loop-Schritt ist.** Die Familie `product_selection` fehlt
bewusst in `controller.PERMITTED_EXPLORATORY_FAMILIES`. Lauf 40387338754
zeigt die Folge: die Frage erscheint in der Rangliste und nie in der
Skip-Liste, weil sie für autonome Exploration gar nicht zulässig ist.
Diese Menge zu erweitern ist eine Governance-Entscheidung und nicht meine.
Die Harness bleibt registriert; der Screen läuft als `research
lottery-pricing`, wenn ein Mensch ihn anstößt.

---

## 4. Alpha-Benachrichtigung

Drei Klassen in `notifications/alpha.py`, nach dem bestehenden Muster
(`*Context` + Template + `render_*` + `notification_hash`), kein
Parallelsystem:

1. `ALPHA_CANDIDATE` — Betreff „Neuer Alpha-Kandidat"; der Verdict-Block
   sagt in der ersten Zeile „Noch NICHT bestätigt".
2. `ALPHA_CONFIRMED` — trägt die 15 verlangten Punkte, einschließlich der
   verbleibenden Einschränkungen.
3. `ALPHA_DEGRADED` — was sich änderte, seit wann, auf welchen Belegen, und
   ob automatisch zurückgestuft.

**Was das Modul verweigert**, statt es nur zu konventionieren: eine
`CONFIRMED`-Mail ohne Wirkung, Intervall, effektive Stichprobe **und**
benannte Kostenbasis wird nicht gerendert, sondern wirft. Eine Mail mit
„wissenschaftlich bestätigt", deren Zahlen optional wären, wäre das
irreführendste Artefakt dieses Systems.

`mark_sent` läuft **nur** nach bestätigtem Versand. Eine auf allen
Versuchen gescheiterte Zustellung bleibt damit für den nächsten Lauf
offen, statt als zugestellt zu gelten — der eine Fehlermodus, der den
Betreiber glauben ließe, informiert worden zu sein. Backoff 2 s, 4 s bei
drei Versuchen.

Der Dedup-Hash deckt Art, Hypothese und gerundete Kennzahlen ab, **nicht**
den Prosatext (eine Template-Änderung darf keine gesendete Meldung
wiederbeleben) und schließt `is_test` ein (ein Zustelltest darf keine
echte Meldung unterdrücken).

`notify alpha-test` belegt den Pfad ohne eine Entdeckung vorzutäuschen:
synthetische Felder, Präfix `RO-DELIVERY-TEST`, `[TEST]` im Betreff, und
ein Banner als erste Zeile. 29 Tests, die meisten davon Verweigerungen.

**Nicht nachgewiesen:** eine echte Zustellung. Ohne konfigurierte
Gmail-Secrets läuft der Befehl als Trockenlauf; dann wird korrekt nichts
vermerkt und Pass 2 nicht unterdrückt. Der Versand selbst und die
Unterdrückung sind durch Unit-Tests abgedeckt, nicht durch eine Mail im
Postfach.

---

## 5. Unbeaufsichtigter Wochenbetrieb

**Nachgewiesen, dass der Zeitplan selbst feuert.** Samstag, 2026-10-24:

| Lauf | Event | Geplant | Gefeuert | Verzögerung | Job |
|---|---|---|---|---|---|
| [40084459900](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40084459900) | `schedule` | 06:00 UTC | 10:04 UTC | 4 h 04 min | weekly ✓ |
| [40095851280](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40095851280) | `schedule` | 08:40 UTC | 12:54 UTC | 4 h 14 min | **research** ✓ |

Der Research-Controller startet also ohne manuelle Eingabe. Die
GitHub-Verzögerung von gut vier Stunden ist gemessen und für diesen
Zeitplan unschädlich: die Reihenfolge weekly → research bleibt erhalten,
weil beide ähnlich verzögert werden und `concurrency: turboedge-state`
Läufe ohnehin serialisiert.

**Was dieser Lauf damals tat: nichts.** Er meldete `AWAIT_DATA` mit der
Vor-Fix-Formulierung („ranks highest but its data are not ready") und
führte 0 Screens aus — genau die Aushungerung, die inzwischen behoben ist.

**Was ich nicht beweisen kann.** Einen künftigen Lauf. Der nächste
planmäßige Research-Lauf ist Samstag, **2026-10-31, 08:40 UTC** (nach
gemessener Verzögerung real gegen 12:50 UTC). Dass derselbe Job nach den
Fixes tatsächlich ein Experiment ausführt, ist an einem manuell
ausgelösten Lauf desselben Jobs belegt (40237604797: Auswahl, Ausführung,
`screen_id`), nicht an einem zeitgesteuerten.

---

## 6. Forschungsabdeckung

**16 Katalogeinträge, 3 Harnesses, 4 beantwortbare Fragen.**

| Katalogeintrag | Familie | Harness | Loop darf? |
|---|---|---|---|
| RO-MAE-PREDICTION | path_statistics | `excursion` | ja |
| RO-MFE-PREDICTION | path_statistics | `excursion` | ja |
| RO-DAY-NIGHT-MOMENTUM | microstructure | `day_night` | ja |
| RO-LOTTERY-PRICING | product_selection | `lottery_pricing` | **nein** (Familie nicht zugelassen) |
| RO-CBOE-VOL-STATE | volatility_surface | — | ja, sobald Harness |
| RO-CFTC-POSITIONING | positioning | — | ja, sobald Harness |
| RO-EUREX-POSITIONING | positioning | — | ja, sobald Harness |
| RO-EUWAX-SENTIMENT | sentiment | — | ja, sobald Harness |
| RO-RATES-CREDIT | rates_credit | — | ja, sobald Harness |
| RO-MARKET-BREADTH | breadth_dispersion | — | ja, sobald Harness |
| RO-DISPERSION | breadth_dispersion | — | ja, sobald Harness |
| RO-EVENT-RISK | event_risk | — | ja, sobald Harness |
| RO-CROSS-ASSET-RESIDUAL-MOMENTUM | cross_asset | — | ja, sobald Harness |
| RO-ISSUER-SPREAD-BEHAVIOUR | product_selection | — | **nein** |
| RO-FINANCING-BEHAVIOUR | product_selection | — | **nein** |
| RO-PRODUCT-SELECTION-EDGE | product_selection | — | **nein** |

**Die 17 Publikationen gegen diese Liste:**

- **Als ausführbare Hypothese registriert (2):** A02 → `RO-LOTTERY-PRICING`
  mit Harness; A12 → `RO-DAY-NIGHT-MOMENTUM` mit Harness, ausdrücklich
  als Zeitreihen-Analogon und nicht als die Originalhypothese.
- **Als Methodik umgesetzt (2):** A09 (Pfadabhängigkeit, KO-Zeitpunkt,
  Totalverlust, Kosten) war bereits im Payoff-Pfad; A10 (Look-Ahead) ist
  jetzt der Pflicht-Audit in `day_night_eval.shares_a_price`, der genau
  diesen Experiment-Befund gekippt hat.
- **Nur als Dokumentation (13):** A01, A03, A04, A05, A06, A07, A08, A11,
  A13, A14, A15, A16, A17. Neun davon scheitern an fehlenden Options-,
  Flow-, Konsens- oder Intraday-Daten; A06 an der Datentiefe (30 min = 60
  Tage); A05 wäre mit Bars machbar und hat nur keine Harness; A16/A17 sind
  Methoden, die erst mit einem Kandidaten relevant werden.

Keine Harness wurde ohne Daten, konkrete Hypothese und Nutzen ergänzt.

---

## Antworten

**Werden echte neue Forward-Labels verarbeitet?** Ja. 2.785 gelabelt,
9 Knock-outs, 0 fehlende Exit-Quotes, 0 überfällig danach.

**Funktioniert der Learning Loop mit diesen Labels?** Teilweise, und die
Trennung ist jetzt sichtbar. Posteriors auf Familienebene: ja, 8
aktualisiert. Modellgewichte: nein, strukturell 0 — die Zuordnung ist
unmöglich, weil der Ledger den Ensemble-Hash speichert und nicht die
Gewichte zum Prognosezeitpunkt.

**Wurde A02 gemessen und mit welchem Ergebnis?** Ja. `CANDIDATE`: 107
Zellen, 105 davon negativ (p = 7,1·10⁻²⁹), Median-Größe 19,6 bps
[16,7, 23,0]. Robust im Vorzeichen über alle vier Emittenten und alle
Basiswert/Richtung-Kombinationen. Kein Alpha, sondern ein Kostenbefund,
und er trennt nicht zwischen Präferenzaufschlag und fairer
Gap-Risiko-Entschädigung.

**Sind Literatur und Forschungsgedächtnis korrekt?** Jetzt ja für
Existenz und Status: alle 17 DOIs aufgelöst, die Barardehi-Angabe
korrigiert. Für den **Inhalt** nur bei vier Arbeiten; bei dreizehn beruht
das Übertragbarkeitsurteil auf Titel und Forschungsfrage.

**Ist die Alpha-Benachrichtigung funktionsfähig?** Durch Tests ja, durch
eine echte Mail nein — ohne Gmail-Secrets ist jeder Lauf ein Trockenlauf.

**Startet und arbeitet der Research Controller ohne manuelle Eingabe?**
Der Start ja, zeitgesteuert belegt am 2026-10-24. Die Ausführung eines
Screens ist bislang nur an manuell ausgelösten Läufen desselben Jobs
belegt; der erste zeitgesteuerte Lauf nach den Fixes ist der 2026-10-31.

**Wie viele der registrierten Hypothesen sind ausführbar?** 4 von 16 haben
eine Harness. 3 davon darf der Loop selbst starten; die vierte
(`RO-LOTTERY-PRICING`) nicht, weil ihre Familie nicht zugelassen ist. Von
den 3 sind heute 0 datenbereit: `RO-DAY-NIGHT-MOMENTUM` ist gelaufen und
abgeschlossen, `RO-MAE/MFE-PREDICTION` haben effektive Stichprobe 1,0 von
48.

**Gibt es nachgewiesenes wirtschaftliches Alpha?** **Nein.** A02 ist ein
Kostenbefund, kein Richtungssignal. Kein Verfahren in dieser oder einer
früheren Sitzung hat einen positiven Netto-Erwartungswert nach Kosten
gezeigt. `NO TRADE` bleibt richtig.

---

## Was eine Betreiberentscheidung braucht

1. **`product_selection` für autonome Exploration zulassen** — oder nicht.
   Drei Katalogfragen, darunter der einzige CANDIDATE dieser Arbeit, sind
   sonst dauerhaft nur manuell startbar. Eine Zeile in
   `controller.PERMITTED_EXPLORATORY_FAMILIES`.
2. **Gmail-Secrets**, falls die Alpha-Mails real zustellen sollen.
3. **Confirmatory-Trial für A02** — `research approve` verlangt einen
   benannten Menschen, und der Loop darf das nicht. Vorher wäre zu klären,
   wie Präferenzaufschlag und Gap-Risiko-Entschädigung getrennt werden
   sollen; ohne diese Trennung bleibt der Befund interpretierbar, aber
   nicht handlungsleitend.
4. **Per-Modell-Gewichte auf neuen Ledger-Einträgen** mitschreiben, falls
   Modellgewichte je aus echten Ergebnissen lernen sollen. Additiv,
   betrifft nur künftige Einträge; die 2.785 vorhandenen bleiben
   unzuordenbar.

---

*Läufe: 40377514500 (eod/Labeling), 40385586039 und 40387338754
(Research-Loop), 40389103879 (A02), 40084459900 und 40095851280
(zeitgesteuerter Samstag). Seeds 20261026 (A02), 20261025 (Tag/Nacht).*
