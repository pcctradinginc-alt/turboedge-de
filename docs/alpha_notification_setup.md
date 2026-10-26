# Alpha-Benachrichtigung: die einmalige Einrichtung

Der automatische Weg ist verdrahtet — ein abgeschlossener Research-Screen
bestimmt selbst seinen Meldungstyp, baut die Nachricht, versucht die
Zustellung, vermerkt den Erfolg dauerhaft und lässt einen Fehlschlag für
den nächsten Lauf offen. Was fehlt, ist ausschließlich die
Zugangsberechtigung. **Ohne die drei Secrets unten läuft jede Meldung als
Trockenlauf: es wird nichts zugestellt und — richtigerweise — auch nichts
als zugestellt vermerkt.**

Ich lege keine Zugangsdaten an und sehe keine. Die folgenden Schritte
müssen Sie einmal selbst ausführen.

## 1. Google-App-Passwort erzeugen

Ein App-Passwort ist nötig, weil SMTP mit dem normalen Kontopasswort bei
aktivierter Zwei-Faktor-Authentifizierung nicht funktioniert.

1. Zwei-Faktor-Authentifizierung für das Google-Konto aktivieren, falls
   noch nicht aktiv (ohne sie gibt es keine App-Passwörter).
2. <https://myaccount.google.com/apppasswords> öffnen.
3. Ein App-Passwort erzeugen, Name frei wählbar (zum Beispiel
   „TurboEdge-DE").
4. Die 16 Zeichen notieren. Google zeigt sie **genau einmal**.

## 2. Die drei GitHub-Secrets setzen

Im Repository unter **Settings → Secrets and variables → Actions → New
repository secret**, jeweils exakt mit diesen Namen:

| Name | Wert |
|---|---|
| `GMAIL_USER` | die sendende Gmail-Adresse, zum Beispiel `pcctradinginc@gmail.com` |
| `GMAIL_APP_PASSWORD` | die 16 Zeichen aus Schritt 1, **ohne** Leerzeichen |
| `TURBOEDGE_EMAIL_TO` | Empfänger; mehrere mit Komma trennen |

Das sind dieselben drei Secrets, die der Scan für Trade-Proposals
benutzt. Sind sie bereits gesetzt, ist hier nichts zu tun.

**Warum das vorher wirkungslos gewesen wäre:** bis zum 2026-10-26 gab der
`research`-Job diese Variablen nicht an den Schritt weiter —
`scan`, `eod`, `weekly` und `monthly` taten es, `research` nicht. Der
automatische Benachrichtigungsweg hätte also auch mit korrekt gesetzten
Secrets nie zugestellt. Das ist behoben.

## 3. Zustellung prüfen

```
gh workflow run pipeline.yml -f mode=research
```

Oder gezielt der gekennzeichnete Zustelltest, der drei Nachrichten sendet
— eine pro Klasse — und dabei keine Entdeckung vortäuscht:

```
gh workflow run pipeline.yml -f mode=eod
```

Lokal, falls die Secrets in der Umgebung liegen:

```
turboedge notify alpha-test
```

Die Testnachrichten tragen `[TEST]` im Betreff, Hypothesen-IDs mit dem
Präfix `RO-DELIVERY-TEST` und als erste Zeile ein Banner, das sagt, dass
kein Ergebnis beschrieben wird. Sie schreiben nichts in die Alpha
Registry.

## Was Sie erwarten dürfen — und was nicht

- Eine Meldung entsteht **nur** bei einem gestützten Urteil. Ein Screen,
  der nichts gefunden hat, ist keine Nachricht.
- Ein Screen kann **niemals** eine Bestätigung auslösen. Er läuft auf
  DEVELOPMENT-Daten, die ausdrücklich nicht als Nachweis gelten; jede
  Screen-Meldung ist ein Kandidat, unabhängig davon, wie groß die Zahl
  ist.
- Ein `COST_EDGE` wird als **Kostenvorteil**-Kandidat gemeldet, nicht als
  Alpha, und die Nachricht beginnt mit „DIES IST KEIN ALPHA". Das ist die
  Bedingung, unter der die autonome Produktselektionsforschung am
  2026-10-26 freigegeben wurde.
- Gleicher Status meldet sich nicht erneut: der Dedup-Hash deckt Art,
  Klassifikation, Hypothese und die gerundeten Kennzahlen ab. Ändert sich
  die Wirkung, gibt es eine neue Meldung — die bewegte Zahl *ist* die
  Nachricht.
- Eine fehlgeschlagene Zustellung wird protokolliert
  (`alpha_alert_undelivered`) und **nicht** als gesendet vermerkt, bleibt
  also für den nächsten Lauf offen. Drei Versuche mit 2 s und 4 s Pause.
