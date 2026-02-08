# MARley

MARley ist ein ChatBot zur Studienberatung fuer den Studiengang `M.Sc. Computer Science` an der Philipps-Universitaet Marburg.

Dieses Repository enthaelt aktuell produktionsreife Implementierungen fuer:
1. `pdf_extractor`
2. `chunker.pdf_chunker`
3. `chunker.faq_chunker`

`retrieval` und `generator` sind im Projekt angelegt, aber in dieser README noch nicht als vollstaendige Laufanleitung beschrieben.

## 1. Voraussetzungen

1. Python `3.10+`
2. `pip`
3. Windows PowerShell (die Beispiele unten sind fuer PowerShell geschrieben)

## 2. Installation

1. In das Projektverzeichnis wechseln:

```powershell
cd c:\Users\joshu\OneDrive\Informatikstudium\Bachelor\MARley
```

2. Virtuelle Umgebung erstellen:

```powershell
python -m venv .venv
```

3. Virtuelle Umgebung aktivieren:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Abhaengigkeiten installieren:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Optionaler Schnellcheck:

```powershell
python -c "import fitz, pdfplumber, tiktoken, syntok; print('ok')"
```

Wichtig: Starte alle Services mit dem Python aus `.venv`. Sonst fehlen dir ggf. Pakete (z. B. `syntok`).

## 3. Erwartete Datenstruktur

Standardpfade (ohne eigene Config) fuer `document_id = msc-computer-science`:

```text
data/
  msc-computer-science/
    raw/
      msc-computer-science.pdf
    knowledgebases/
      ...pdf-extractor....json
      ...faq....json
    chunks/
      ...pdf-chunker....json
      ...faq-chunker....json
```

## 4. Schnellstart: Pipeline bis Komponente 2

### 4.1 PDF Extractor starten

```powershell
python -m uvicorn pdf_extractor.app:app --host 127.0.0.1 --port 8001 --reload
```

Endpoints:
1. `GET http://127.0.0.1:8001/health`
2. `GET http://127.0.0.1:8001/ready`
3. `POST http://127.0.0.1:8001/extractions`
4. Swagger UI: `http://127.0.0.1:8001/docs`

Extractor ausfuehren (Standard-`document_id`, Standardpfade):

```powershell
$body = @{
  document_id = "msc-computer-science"
  persist_result = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8001/extractions" `
  -ContentType "application/json" `
  -Body $body
```

Ergebnis: JSON in `data/<document_id>/knowledgebases/*-pdf-extractor-*.json`.

### 4.2 PDF Chunker starten

```powershell
python -m uvicorn chunker.pdf_chunker.app:app --host 127.0.0.1 --port 8002 --reload
```

Endpoints:
1. `GET http://127.0.0.1:8002/health`
2. `GET http://127.0.0.1:8002/ready`
3. `POST http://127.0.0.1:8002/chunks`
4. Swagger UI: `http://127.0.0.1:8002/docs`

Chunking ausfuehren:

```powershell
$body = @{
  document_id = "msc-computer-science"
  persist_result = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8002/chunks" `
  -ContentType "application/json" `
  -Body $body
```

Ergebnis: JSON in `data/<document_id>/chunks/*-pdf-chunker-*.json`.

Aktuelle Standardparameter:
1. `min_chunk_tokens = 256`
2. `max_chunk_tokens = 512`
3. `overlap_tokens = 64`
4. Tabellen werden als eigene Chunks aufgenommen.

### 4.3 FAQ Chunker starten

```powershell
python -m uvicorn chunker.faq_chunker.app:app --host 127.0.0.1 --port 8003 --reload
```

Endpoints:
1. `GET http://127.0.0.1:8003/health`
2. `GET http://127.0.0.1:8003/ready`
3. `POST http://127.0.0.1:8003/chunks`
4. Swagger UI: `http://127.0.0.1:8003/docs`

Studienordnungs-FAQ (`faq_source = so`) chunken:

```powershell
$body = @{
  document_id = "msc-computer-science"
  faq_source = "so"
  persist_result = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8003/chunks" `
  -ContentType "application/json" `
  -Body $body
```

Ergebnis: JSON in `data/<document_id>/chunks/*-faq-chunker-so-*.json`.

Hinweis zu `sb`:
1. `faq_source = sb` ist implementiert.
2. Falls noch keine SB-Datei vorhanden ist, liefert der Service absichtlich `404`.
3. Sobald die Datei da ist, sollte ihr Dateiname einen SB-Marker enthalten, z. B. `faq-sb`.
4. Alternativ kannst du `input_file` im Request explizit setzen.

## 5. Typische Fehler

1. `syntok` fehlt beim `pdf_chunker`:
   Ursache ist fast immer ein falscher Python-Interpreter.
   Loesung: `.venv` aktivieren und Service neu starten.
2. `404` beim `faq_source = sb`:
   Es gibt noch keine passende SB-Inputdatei oder kein passendes Dateinamen-Muster.
3. Keine PDF gefunden:
   Lege die Datei unter `data/<document_id>/raw/` ab oder setze `source_file` im Extractor-Request.

## 6. Aktueller Stand

Bis jetzt kannst du die komplette Pipeline bis inklusive Chunking produktiv ausfuehren:
1. PDF -> `pdf_extractor` JSON
2. Extractor JSON -> `pdf_chunker` Chunks
3. FAQ JSON -> `faq_chunker` Chunks
