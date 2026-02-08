# MARley

MARley ist ein ChatBot zur Studienberatung fuer den Studiengang `M.Sc. Computer Science` an der Philipps-Universitaet Marburg.

Aktueller Stand (produktionsreif):
1. `pdf_extractor`
2. `chunker.pdf_chunker`
3. `chunker.faq_chunker`
4. `retrieval.sparse_retrieval` (BM25)
5. `retrieval.vector_retrieval` (ChromaDB + Ollama Embeddings)
6. `retrieval.hybrid_retrieval` (RRF ueber Sparse + Vector)

`generator` ist im Projekt angelegt, aber in dieser README noch nicht beschrieben.

## 1. Voraussetzungen

1. Python `3.10+`
2. `pip`
3. Windows PowerShell
4. Laufender Ollama-Server mit Embedding-Modell `nomic-embed-text:latest`

## 2. Installation

1. In das Projekt wechseln:

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

5. Schnellcheck:

```powershell
python -c "import fitz,pdfplumber,tiktoken,syntok,chromadb,rank_bm25; print('ok')"
```

Wichtig: Starte alle Services mit dem Python aus `.venv`.

## 3. Erwartete Datenstruktur

Standardpfade fuer `document_id = msc-computer-science`:

```text
data/
  msc-computer-science/
    raw/
      msc-computer-science.pdf
    knowledgebases/
      ...-pdf-extractor-....json
      ...faq....json
    chunks/
      ...-pdf-chunker-....json
      ...-faq-chunker-so-....json
      ...-faq-chunker-sb-....json
```

## 4. Komponente 1: PDF Extractor

Start:

```powershell
python -m uvicorn pdf_extractor.app:app --host 127.0.0.1 --port 8001 --reload
```

Ausfuehren:

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

Output: `data/<document_id>/knowledgebases/*-pdf-extractor-*.json`

## 5. Komponente 2: Chunker

### 5.1 PDF Chunker

Start:

```powershell
python -m uvicorn chunker.pdf_chunker.app:app --host 127.0.0.1 --port 8002 --reload
```

Ausfuehren:

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

Standardeinstellungen:
1. `min_chunk_tokens = 256`
2. `max_chunk_tokens = 512`
3. `overlap_tokens = 64`
4. Tabellen werden als eigene Chunks erzeugt.

### 5.2 FAQ Chunker

Start:

```powershell
python -m uvicorn chunker.faq_chunker.app:app --host 127.0.0.1 --port 8003 --reload
```

Ausfuehren (`faq_source = so`):

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

Output: `data/<document_id>/chunks/*-faq-chunker-so-*.json`

Hinweis zu `faq_source = sb`:
1. Ist bereits implementiert.
2. Fehlt eine SB-Datei, ist das Verhalten absichtlich `404`.
3. Sobald SB-Datei vorhanden ist, laeuft dieselbe Pipeline.

## 6. Komponente 3: Retrieval

### 6.1 Services starten

1. ChromaDB (HTTP Backend):

```powershell
python -m uvicorn chromadb.app:app --host 127.0.0.1 --port 8000
```

2. Sparse Retrieval (BM25):

```powershell
python -m uvicorn retrieval.sparse_retrieval.app:app --host 127.0.0.1 --port 8004
```

3. Vector Retrieval (Chroma + Ollama):

```powershell
python -m uvicorn retrieval.vector_retrieval.app:app --host 127.0.0.1 --port 8005
```

4. Hybrid Retrieval (RRF):

```powershell
python -m uvicorn retrieval.hybrid_retrieval.app:app --host 127.0.0.1 --port 8006
```

Alternative Entry-Point (zeigt auf Hybrid):

```powershell
python -m uvicorn retrieval.app:app --host 127.0.0.1 --port 8006
```

### 6.2 Index aufbauen

```powershell
$body = @{ document_id = "msc-computer-science" } | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8004/index/rebuild" `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8005/index/rebuild" `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8006/index/rebuild" `
  -ContentType "application/json" `
  -Body $body
```

### 6.3 Suchen (Hybrid)

```powershell
$body = @{
  query = "Welche Zulassungsvoraussetzungen gibt es?"
  document_id = "msc-computer-science"
  top_k = 10
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8006/search" `
  -ContentType "application/json" `
  -Body $body
```

Nur PDF-Quelle:

```powershell
$body = @{
  query = "Masterarbeit Bearbeitungszeit"
  document_id = "msc-computer-science"
  top_k = 10
  source_types = @("pdf")
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8006/search" `
  -ContentType "application/json" `
  -Body $body
```

## 7. Endpoints

Alle Retrieval-Services haben dieselben Kernendpoints:
1. `GET /health`
2. `GET /ready`
3. `POST /index/rebuild`
4. `POST /search`

Swagger:
1. Sparse: `http://127.0.0.1:8004/docs`
2. Vector: `http://127.0.0.1:8005/docs`
3. Hybrid: `http://127.0.0.1:8006/docs`

## 8. Typische Fehler

1. `ModuleNotFoundError` bei Chroma/Vector:
   Ursache: falscher Interpreter (global statt `.venv`).
2. Vector meldet Chroma nicht erreichbar:
   Ursache: Chroma auf Port `8000` nicht gestartet.
3. `SOURCE_FILE_MISSING` Flag:
   Wenn `faq_sb` fehlt, ist das erwartetes Verhalten. Pipeline laeuft weiter.
4. Umlaute im Request-Body:
   Bei Bedarf JSON explizit UTF-8 senden (z. B. per Datei + `curl --data-binary`).

## 9. Aktueller Pipeline-Stand

Du kannst aktuell End-to-End bis Retrieval produktiv ausfuehren:
1. PDF -> `pdf_extractor`
2. Extractor-JSON -> `pdf_chunker`
3. FAQ-JSON -> `faq_chunker`
4. Chunks -> `sparse` / `vector`
5. `sparse + vector` -> `hybrid (RRF)`
