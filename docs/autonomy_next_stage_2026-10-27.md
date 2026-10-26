# Nächste Autonomiestufe — Abschlussbericht (2026-10-27)

Keine Risikolimits, ACTIONABLE-Kriterien, Promotion-Gates, Signifikanz-
schwellen oder Forschungsbudgets gelockert. `PROMOTION_LOCKED` steht.
Kein Signal aktiviert, kein Trade freigegeben.

---

## 1. Autonome Produktselektionsforschung — freigegeben und nachgewiesen

`product_selection` ist in `PERMITTED_EXPLORATORY_FAMILIES` aufgenommen,
mit Ihren Bedingungen im Code neben der Zeile. Sechs Tests sichern, was
die Freigabe **nicht** gewährt: ein gestützter Screen hält weiterhin an
`AWAIT_CONFIRMATION_APPROVAL`, die Quartalsobergrenze bindet, Promotion
bleibt gesperrt, und eine bereits gescreente Frage wird auf denselben
Daten nicht wiederholt.

**Nachgewiesen am laufenden System**, Lauf
[40397622319](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40397622319):
der Controller wählte `RO-LOTTERY-PRICING` selbst, führte
`lottery_pricing@1` aus, erhielt `CANDIDATE` und speicherte
`2026Q4_SCREEN_d092329f73`. Lauf
[40399207936](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40399207936)
screente **nicht erneut**, sondern meldete
`AWAIT_CONFIRMATION_APPROVAL` — beide Bedingungen greifen also.

**A02 ist `COST_EDGE`**, maschinenlesbar und nicht nur in Prosa.
`ResearchKind` ist eine neue Achse neben `InformationFamily`: die Familie
sagt, woher Information kommt, die Art sagt, was ein positives Ergebnis
überhaupt belegen würde. Alle 16 Katalogeinträge deklarieren eine — 10
`ALPHA_HYPOTHESIS`, 4 `COST_EDGE`, 2 `RISK_IMPROVEMENT` — und eine
festgenagelte Zuordnungstabelle macht ein Verschieben zu einem bewussten
Akt.

---

## 2. Point-in-Time-Audit — geklärt, und ein zweiter Befund

**Kein Datenleck.** Die Ursache ist eine Zeile: `prediction_time =
clock()` ist die **erste Anweisung** von `pipeline/scan.run_scan`, vor
jedem Produktabruf. Die Quote-Zeitstempel der Emittenten liegen danach,
und zwar zwangsläufig.

Gemessen über 7.297 Entscheidungen (Lauf
[40404282304](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40404282304)):

| | |
|---|---:|
| Quote **vor** der Entscheidung | 1.099 |
| Quote danach, innerhalb des 900-s-Scanfensters | 6.198 |
| Quote danach, **jenseits** des Fensters | **0** |
| extremster Wert danach | **−74,7 s** |
| `leak_suspected` | **False** |

Die „nach der Entscheidung"-Seite reicht also maximal 75 Sekunden — klar
innerhalb eines Scans, und kein einzelner Fall ist durch die Reihenfolge
unerklärbar. Das ist eine **zulässige zeitliche Reihenfolge**: ein Leck
hieße Information aus der Zukunft *der Entscheidung*, hier ist der Quote
der Preis, auf dem die Entscheidung tatsächlich getroffen wurde. Die
Feature-Prüfung `available_at <= prediction_time` ist sogar
konservativ, weil sie Features auf einen früheren Moment festlegt als den,
an dem die Auswahl wirklich geschah.

**Was es bedeutet:** Quote-Alter *zur Entscheidung* ist aus dem Ledger
**nicht** rekonstruierbar, und mein eigenes `quote_age_seconds` war
benannt, als wäre es das. Die Zeitzonenkonversion wurde geprüft und ist
korrekt (`ZoneInfo("Europe/Berlin")` → UTC), ein systematischer Offset ist
ausgeschlossen.

**Einfluss auf A02: keiner.** A02 benutzt `retrieved_at −
quote_timestamp` auf der Archivseite, also das Maß, das die
Frischefrage beantwortet.

**Der zweite Befund, auf der anderen Seite.** 1.099 Entscheidungen
benutzten Quotes von **vor** der Entscheidung, und die Flanke ist
extrem: p99 bei +5.400 s (1,5 Stunden), Maximum **+2.854.484 s — 33
Tage**. Nach Emittent: BNP hat beide Sorten (1.061 vorher, 6.198 im
Fenster); Goldman Sachs, UniCredit und HSBC haben **ausschließlich**
Quotes von vorher (14/11/13). Das ist keine Zukunftsinformation, sondern
Veraltung — und bei den drei kleinen Emittenten systematisch. Bei 38
Einträgen insgesamt ist es klein, aber es ist real und bisher ungeprüft.

---

## 3. Modellattribution — die Daten waren schon da

Die Komposition und die Gewichte zum Prognosezeitpunkt werden bereits
unveränderlich gespeichert, und ein zweiter Speicher wäre das
Parallelsystem gewesen, das Ihr Auftrag verbietet. `pipeline/scan.py`
schreibt pro `(run_id, underlying_id, horizon_days, model_id)` eine
`forecasts`-Zeile für jedes Komponentenmodell plus eine
`ensemble`-Zeile, deren `component_weights` die Gewichtskarte hält, **wie
sie zum Prognosezeitpunkt stand**. Jede Zeile trägt Modell-ID und -Hash,
eigene Prognose (mean/sigma/quantiles/p_up), `n_train`, `n_effective`,
`prediction_time`, `frozen_at`, `config_hash` und `git_commit`. Inserts
sind `ON CONFLICT DO NOTHING` — einmal geschrieben, nie überschrieben.

Gefehlt hat die **Verknüpfung**. `LedgerEntry` trägt `run_id`,
`underlying` und `horizon_days` — genau der Schlüssel, unter dem die
Zeilen liegen.

**Und das korrigiert eine meiner Aussagen zu Ihren Gunsten.** Lauf
[40406641630](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40406641630):
**2.785 von 2.785 gelabelten Einträgen attributierbar, 0 unvollständig,
kein gewichtetes Modell ohne Prognosezeile.** Die Prognosezeilen
existieren auch für die alten Einträge. Unattributierbar war der
`model_hash`-Join des Learn-Loops, **nicht** das Archiv. Einzelmodelle
können also *jetzt* bewertet werden, nicht erst künftig.

| Modell | n | Fälligkeitstage | MSE | Richtungstreffer | Gewicht | lesbar |
|---|---:|---:|---:|---:|---:|---|
| tsmom_forecast_v1 | 2785 | **1** | 1,89·10⁻⁵ | 0,000 | 1,0 | **nein** |
| null_v1 | 2785 | **1** | 4,16·10⁻⁵ | 0,000 | 1,0 | **nein** |
| logistic_direction_v1 | 2785 | **1** | 1,08·10⁻⁴ | 0,000 | 1,0 | **nein** |
| ensemble | 2785 | **1** | 4,76·10⁻⁵ | 0,515 | — | **nein** |

**Derselbe Lauf hat einen Mangel in dem Code gezeigt, den ich gerade
geschrieben hatte.** Alle drei Modelle meldeten eine Richtungstrefferquote
von **exakt 0,000** und `interpretable=True`. Als Zeilenzahl gelesen ist
das ein vernichtendes Urteil über drei Modelle. Es ist **ein**
Fälligkeitstag: alle 2.785 Labels reiften am selben Tag, also ist die
Zahl ein einzelner Münzwurf, 2.785-mal notiert, und alle drei zeigten an
diesem einen Tag in die falsche Richtung. `ModelScore` zählt jetzt
`distinct_maturity_dates`, und `interpretable` hängt daran
(`MIN_MATURITY_DATES = 20`), nicht an `n`. Das Datum kommt vom Label,
nicht von der Prognose: zwei an verschiedenen Tagen prognostizierte
Einträge, die gemeinsam fällig werden, teilen **ein** Ergebnis.

**Was bewusst nicht passiert:** der Ensemble-P&L wird nicht nach
Gewichten auf die Mitglieder verteilt. Diese Zahl gehört zu *einer*
Entscheidung der Kombination; sie aufzuteilen erfände ein
Kontrafaktisches, in dem jedes Modell allein gehandelt hätte. Bewertet
wird stattdessen die **eigene Prognose** jedes Modells gegen die
realisierte Rendite des Basiswerts — nicht gegen den P&L des Produkts,
denn eine Prognose ist eine Aussage über den Markt, und sie an einem
gehebelten Auszahlungsprofil zu messen würde die Produktauswahl
mitbewerten. Und es wird **kein Gewicht aktualisiert**: ein Test greppt
den Quelltext nach `update_weights`, `upsert_model_registry_entry` und
`ModelRegistry` und schlägt an, falls eines auftaucht.

Nebenbefund: die Registry gibt allen drei Modellen Gewicht **1,0**. Das
Ensemble ist also ein gleichgewichteter Mittelwert, kein gelernter — was
daraus folgt, dass `models_reweighted` seit dem Bau des Loops
strukturell 0 war.

---

## 4. A02 wissenschaftlich weiterentwickelt — und zurückgestuft

Sie fragten, ob die 107 Vergleichszellen hinreichend unabhängig sind.
**Sie sind es nicht**, und die Antwort stuft meinen eigenen positiven
Befund zurück.

Die 107 Zellen verteilen sich auf **3 Tage und 16 Minuten** — 27 Zellen
am 22. Oktober, 16 am 23., 64 am 26. Ein Bootstrap über Zellen hat die
Präzision damit deutlich überschätzt.

| | vorher | nachher |
|---|---|---|
| Resampling-Einheit | 107 Zellen | **12 (Tag, Emittent)-Cluster** |
| mittlere Steigung | −0,4125 [−0,6519, −0,2113] | **−0,4467 [−0,8168, −0,2795]** |
| Tageskriterium | keines | **`MIN_INDEPENDENT_DAYS = 10`**, vor dem Intervall geprüft |
| **Urteil** | `CANDIDATE` | **`INSUFFICIENT_SAMPLE`** |

Erneut gemessen (Lauf 40404282304): 3 Tage, 18 Minuten, 12 Cluster, Median
+20,16 bps, Mittel +145,83 bps. Das Vorzeichen bleibt konsistent — 105
von 107 Zellen negativ, Vorzeichentest p = 7,1·10⁻²⁹, alle vier
Emittenten und alle vier Basiswert/Richtung-Kombinationen negativ. Das
**Intervall** bleibt nicht.

**Strukturelle Grenze, nicht Laufgrenze.** `state/retention.py` dünnt
`product_snapshots` älter als fünf Tage auf eine Zeile pro ISIN und Tag
aus, und eine Zeile pro Tag kann keine Vergleichsgruppe derselben Minute
bilden. Ein Rückblick über 30 Tage liefert daher etwa **fünf** Tage
nutzbarer Minutenstruktur, wie oft der Screen auch läuft. Unabhängige
Tage zu sammeln heißt: Ergebnisse über Läufe hinweg kombinieren, oder für
eine Stichprobe von Produkten die volle Tiefe behalten. Das ist eine
Retention-Entscheidung, keine statistische.

### Die Identifikationsfrage, die offen bleibt

Ihre eigentliche Frage — *teurer, als das eingegangene Risiko
rechtfertigt?* — ist mit dem jetzigen Design **nicht** beantwortbar, und
der Grund ist nicht Stichprobengröße.

Innerhalb einer Zelle (ein Emittent, ein Basiswert, eine Minute) **ist
der Barriereabstand das Risiko.** Die analytische KO-Wahrscheinlichkeit
ist eine deterministische Funktion davon
(`research/first_passage.py`), also perfekt kollinear mit dem
Regressor. „Für Risiko bepreist" und „für Lotteriepräferenz bepreist"
sind in dieser Spezifikation dieselbe Variable.

Trennen ließe sich das nur über eine Größe, die das Risiko **bei
festem** Barriereabstand verschiebt — Volatilität. Faire
Gap-Risiko-Entschädigung sollte mit σ skalieren, ein
Präferenzaufschlag nicht zwingend. Der Test wäre also die
**Interaktion** `Barriereabstand × σ` über Zellen hinweg, mit σ aus
`features/volatility.causal_ewma_sigma`. Das braucht Variation in σ
zwischen Zellen bei Variation im Barriereabstand innerhalb — mit 3
unabhängigen Tagen ist das heute nicht zu haben.

Der primäre Endpunkt ist eingefroren, einschließlich medianbasierter
wirtschaftlicher Effektgröße und Clusterung, bevor eine neue Evaluation
läuft. Eine bestätigende Freigabe ist **nicht** gerechtfertigt.

---

## 5. Alpha-Benachrichtigung — automatisch, und eine Fehlmeldung auf dem Weg

`alert_for_screen` macht aus einem abgeschlossenen Screen eine Nachricht
oder nichts, und `run_research_step` ruft es auf. `None` ist die häufige
und richtige Antwort: nur ein gestütztes Urteil meldet, also kann
`LOOKAHEAD_SUSPECT` nicht verkündet werden. Jede Screen-Meldung ist ein
**Kandidat**, unabhängig von der Effektgröße — ein Screen läuft auf
DEVELOPMENT-Daten, und das ist eine Eigenschaft des Ortes, nicht der Zahl.

**Die Lücke, die das Ganze wirkungslos gemacht hätte:** der
`research`-Job gab **keine** Gmail-Variablen weiter. `scan`, `eod`,
`weekly` und `monthly` taten es, `research` nicht. Der automatische Weg
existierte im Code und hätte nie zugestellt. Behoben.

**Und eine Fehlmeldung, die tatsächlich hinausging.** In Lauf
40397622319 lief die Kette erstmals vollständig durch und meldete `sent:
true` — die Secrets sind konfiguriert, eine echte Mail erreichte das
Postfach. Sie war **falsch betitelt**: das Log zeigt
`research_kind: "ALPHA_HYPOTHESIS"` für eine Frage, die der Katalog als
`COST_EDGE` führt, der Betreff lautete also „Neuer Alpha-Kandidat".
Genau das, was Ihre Bedingung verhindern sollte.

Die Ursache war meine: ich persistierte `research_kind` als nullable
Spalte und ließ ein NULL aus der Zeit vor der Migration auf den
Modell-Default fallen — in der Annahme, `research queue` sät neu und
heilt das. Das tut es nicht: `seed_from_catalog` überspringt eine
vorhandene ID absichtlich, damit Re-Seeding keine menschliche Arbeit
zurücksetzt. Behoben doppelt: `catalog.research_kind_for` macht den
Katalog autoritativ (Quelltext kann nicht veralten), und das Seeding
frischt diese **eine** Spalte auf, weil sie aussagt, *was* eine Frage
ist, und nicht, wie jemand sie beurteilt. Ein Test sichert, dass
`status`, `approved_by` und `status_note` unberührt bleiben.

**Formulierung nach Art.** Ein `COST_EDGE` heißt im Betreff
„Kostenvorteil-Kandidat", nie „Alpha-Kandidat", und der Rumpf beginnt mit
„DIES IST KEIN ALPHA" **vor** der Statuszeile. Tests prüfen die
Reihenfolge, nicht nur das Vorkommen.

**Gekennzeichneter echter Zustelltest**, Lauf
[40402691806](https://github.com/pcctradinginc-alt/turboedge-de/actions/runs/40402691806):
**4 gesendet, 4 unterdrückt, 0 fehlgeschlagen.** Vier Klassen — je eine
pro Typ plus ein COST_EDGE-Kandidat, also genau der Pfad, der in der
Produktion schiefging. Pass 2 wurde vollständig als Duplikat
unterdrückt. Die Secrets sind gesetzt;
`docs/alpha_notification_setup.md` beschreibt sie für den Fall eines
Neuaufbaus.

---

## 6. Lernprozess über mehrere Fälligkeitstage

Basislinie gemessen, Instrument steht, **zweiter Tag noch nicht
vorhanden**:

| Größe | 2026-10-26 |
|---|---:|
| Ledger-Einträge | 7.297 |
| gelabelt (Horizont / KO) | 2.776 / 9 |
| noch nicht fällig | 4.512 |
| fehlende Exit-Quotes | 0 |
| **unabhängige Fälligkeitstage** | **1** |
| Posterior-Updates | 8 |
| Modellattribution | 2.785 / 2.785 |
| Modellgewicht-Updates | 0 (strukturell, siehe §3) |
| MAE/MFE-Datenbereitschaft | effektive Stichprobe 1,0 von 48 |

Der letzte Lauf (40413657202, 2026-10-26 22:35 UTC) war ein Scan; ein
eod seit meinem manuellen um 18:24 UTC hat nicht stattgefunden. Es gibt
also noch keinen zweiten Fälligkeitstag, und mehr Tage kann ich nicht
herstellen.

Beobachtbar ist jetzt pro Lauf: `distinct_maturity_dates` je Modell,
`entries_attributable`, `labeled_entries_unattributable`,
`point_in_time_ordering` und die Trennung
`not_yet_due`/`overdue_unlabeled`. Die Mindestanforderungen an
unabhängige Beobachtungen wurden **nicht** gesenkt, sondern an zwei
Stellen angehoben (`MIN_MATURITY_DATES`, `MIN_INDEPENDENT_DAYS`).

---

## 7. Forschungsabdeckung

**16 Katalogfragen, 4 mit Harness.** Was mit vorhandenen Daten sinnvoll
dazukommen könnte:

| Frage | Daten vorhanden? | Einschätzung |
|---|---|---|
| RO-ISSUER-SPREAD-BEHAVIOUR | **ja**, dasselbe Snapshot-Archiv wie A02 | **aussichtsreichste Ergänzung**; Familie jetzt zugelassen; dieselbe Tages-Unabhängigkeitsgrenze |
| RO-FINANCING-BEHAVIOUR | **ja**, `financing_level_history` | klar falsifizierbar: steigt die implizite Finanzierung mit dem Hebel? |
| RO-PRODUCT-SELECTION-EDGE | **ja** | braucht erst die Pair-Zählung aus `research product-pairs` |
| RO-RATES-CREDIT | ja, ECB/ESTR | machbar, aber Horizontfrage offen |
| RO-CROSS-ASSET-RESIDUAL-MOMENTUM | ja (Bars) | `cross_asset_leadlag` ist **dormant**; ohne neuen Mechanismus nicht wieder öffnen |
| RO-CBOE-VOL-STATE, RO-CFTC-POSITIONING | ja | bereits gemessen und gescheitert (0/20 Zellen) |
| RO-EUREX-POSITIONING | nein (effektive Stichprobe ~357) | blockiert |
| RO-EUWAX-SENTIMENT | nein (HTTP 403) | blockiert |
| RO-MARKET-BREADTH, RO-DISPERSION | nein | brauchen einen Querschnitt; vier Basiswerte haben keinen |
| RO-EVENT-RISK | teilweise | braucht einen zeitpunktgetreuen Makrokalender |
| RO-MAE/MFE-PREDICTION | ja, aber unreif | warten auf Fälligkeitstage |

Keine Harness ohne Daten, konkrete Hypothese und Nutzen ergänzt. Und:
die inhaltliche Prüfung der Originalarbeiten vor einer neuen
Hypothesenimplementierung steht aus — verifiziert sind alle 17 DOIs nach
Existenz und Fundstelle, inhaltlich geprüft sind vier.

---

## Antworten

**1. Ist `product_selection` freigegeben?** Ja, mit Ihren Bedingungen im
Code und sechs Tests darauf. Nachgewiesen in Lauf 40397622319.

**2. Ist die Quote-Zeitstempel-Auffälligkeit geklärt?** Ja: zulässige
Reihenfolge, kein Leck. 0 von 7.297 Fällen jenseits des Scanfensters,
extremster Wert −74,7 s. A02 unberührt. Dabei ist ein **neuer** Befund
entstanden: 1.099 Entscheidungen auf *veralteten* Quotes, Maximum 33
Tage, bei drei Emittenten systematisch.

**3. Werden Komposition und Gewichte unveränderlich gespeichert?** Ja —
und sie wurden es schon, in der `forecasts`-Tabelle. Neu ist die
Verknüpfung, nicht der Speicher.

**4. Kann der Learning Loop Einzelmodelle bewerten?** Ja, und zwar auch
rückwirkend: 2.785 von 2.785 attributierbar. Was heute fehlt, sind
unabhängige Fälligkeitstage — 1 von 20 — nicht die Attribution.

**5. Ist A02 nach Risikobereinigung noch auffällig?** Im **Vorzeichen**
ja (105/107 Zellen, p = 7,1·10⁻²⁹, alle Emittenten). Im **Urteil** nein:
`INSUFFICIENT_SAMPLE` bei 3 unabhängigen Tagen. Und
risiko*bereinigt* ist es nicht, weil Barriereabstand und KO-Risiko in
dieser Spezifikation dieselbe Variable sind — die Trennung bräuchte eine
σ-Interaktion.

**6. Funktioniert der automatische Benachrichtigungsweg?** Ja, live
belegt: eine echte Meldung aus dem Loop (40397622319) und ein
gekennzeichneter Zustelltest mit 4 gesendet / 4 unterdrückt
(40402691806). Die erste echte Meldung war falsch betitelt; die Ursache
ist behoben und doppelt abgesichert.

**7. Werden neue Forward-Labels zuverlässig verarbeitet?** Ja: 2.785
gelabelt, 9 KO, 0 fehlende Exit-Quotes, 0 danach überfällig, 8
Posteriors. Modellgewichte bleiben bei 0, und das ist jetzt sichtbar
statt stumm.

**8. Welche Experimente wurden tatsächlich ausgeführt?** A02
(`lottery_pricing@1`) zweimal — einmal vom Controller selbst gewählt
(40397622319), einmal als menschlich gestarteter Screen (40389103879,
40404282304). Zuvor die Tag/Nacht-Familie. Keine Wiederholung ohne neue
Daten.

**9. Sind alle sicheren Änderungen gemergt?** Ja: PRs #16–#21, CI grün,
keine offenen PRs.

**10. Gibt es nachgewiesenes wirtschaftliches Alpha?** **Nein.** A02 ist
ein Kostenbefund, derzeit mit unzureichender Stichprobe, und nicht
risikobereinigt. `NO TRADE` bleibt richtig.

---

## Was Ihnen gehört

1. **Retention für A02.** Ohne mehr als ~5 Tage Minutenstruktur erreicht
   A02 die 10 unabhängigen Tage nie in einem Lauf. Entweder Ergebnisse
   über Läufe hinweg kombinieren oder für eine Produktstichprobe die
   volle Tiefe behalten.
2. **Veraltete Quotes bei Goldman, UniCredit und HSBC** — 38 Einträge,
   ausschließlich Quotes von vor der Entscheidung, Maximum 33 Tage. Soll
   eine Frischegrenze im Scan greifen?
3. **Confirmatory-Trial für A02** — erst sinnvoll, wenn die σ-Interaktion
   die Risikobereinigung leistet.
4. **Inhaltliche Literaturprüfung** der 13 noch nicht inhaltlich
   geprüften Arbeiten, vor einer weiteren Hypothesenimplementierung.

---

*Läufe: 40377514500 (Labeling), 40397622319 (autonomer A02-Screen mit
Meldung), 40399207936 (keine Wiederholung), 40402691806 (Zustelltest),
40404282304 (Point-in-Time + A02 geclustert), 40406641630
(Modellattribution). Seeds 20261026 (A02), 20261025 (Tag/Nacht).*
