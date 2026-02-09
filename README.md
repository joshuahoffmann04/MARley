# MARley

MARley ist ein ChatBot zur Studienberatung fuer den Studiengang `M.Sc. Computer Science` an der Philipps-Universitaet Marburg.

Aktueller Stand (produktionsreif):
1. `pdf_extractor`
2. `chunker.pdf_chunker`
3. `chunker.faq_chunker`
4. `retrieval.sparse_retrieval` (BM25)
5. `retrieval.vector_retrieval` (ChromaDB + Ollama Embeddings)
6. `retrieval.hybrid_retrieval` (RRF ueber Sparse + Vector)
7. `generator` (Antwortgenerierung mit Abstention)

## 1. Voraussetzungen

1. Python `3.10+`
2. `pip`
3. Windows PowerShell
4. Laufender Ollama-Server mit
   `nomic-embed-text:latest` (Retrieval) und
   `llama3.1:latest` (Generator)

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

Wichtig: Starte alle Services mit dem Python aus `.venv`.

## 3. Single-App Schnellstart (empfohlen)

Wenn du die komplette Pipeline inkl. Frontend als **eine einzige App** starten willst:

1. Ollama starten (machst du bereits selbst):

```powershell
ollama serve
```

2. Sicherstellen, dass beide Modelle vorhanden sind:

```powershell
ollama pull nomic-embed-text:latest
ollama pull llama3.1:latest
```

3. MARley Single-App starten:

```powershell
python -m uvicorn MARley.app:app --host 127.0.0.1 --port 8010 --reload
```

Alternative direkt mit Uvicorn (mit stabilen Reload-Regeln):

```powershell
python -m uvicorn MARley.app:app --host 127.0.0.1 --port 8010 --reload `
  --reload-dir MARley `
  --reload-dir generator `
  --reload-dir retrieval `
  --reload-dir chunker `
  --reload-dir pdf_extractor `
  --reload-exclude ".venv/*" `
  --reload-exclude ".venv/**" `
  --reload-exclude "data/**/databases/**"
```

4. Frontend öffnen:
1. Chat: `http://127.0.0.1:8010/`
2. Debug-Subpage (JSON): `http://127.0.0.1:8010/debug`

Was die App intern macht:
1. Erkennt automatisch verfügbare Studienordnungen unter `data/*`.
2. Nutzt lokale Retrieval-Services im selben Prozess (`sparse`, `vector`, `hybrid`).
3. Nutzt für Vector standardmäßig lokalen Chroma-Client im Dokumentpfad `data/<document_id>/databases`.
4. Ruft nur Ollama extern auf (Embeddings + Generator).

## 4. Erwartete Datenstruktur

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
    databases/
      ...
```

## 5. Komponente 1: PDF Extractor

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

## 6. Komponente 2: Chunker

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

## 7. Komponente 3: Retrieval (Einzelservices / Legacy-Modus)

Hinweis: Fuer die neue Single-App musst du diese Services **nicht** separat starten.

### 7.1 Services starten

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

### 7.2 Index aufbauen

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

### 7.3 Suchen (Hybrid)

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

## 8. Komponente 4: Generator (Einzelservice / Legacy-Modus)

### 8.1 Service starten

```powershell
python -m uvicorn generator.app:app --host 127.0.0.1 --port 8007 --reload
```

Wichtig: Der Generator erwartet ein laufendes Hybrid-Retrieval auf `http://127.0.0.1:8006` und einen laufenden Ollama-Server.

### 8.2 Antwort erzeugen

```powershell
$body = @{
  query = "Welche Zulassungsvoraussetzungen gibt es?"
  document_id = "msc-computer-science"
  top_k = 10
  model = "llama3.1:latest"
  total_budget_tokens = 2048
  max_answer_tokens = 384
  include_used_chunks = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8007/generate" `
  -ContentType "application/json" `
  -Body $body
```

Response-Eigenschaften:
1. `answer`: finale Antwort (Deutsch)
2. `abstained`: `true/false`
3. `abstention_reason`: Grund fuer Abstention (falls aktiv)
4. `used_chunks`: verwendete Chunks inkl. `chunk_id` und kompletter `metadata`
5. `retrieval_quality_flags` und `generator_quality_flags`

### 8.3 Abstention feinjustieren (optional pro Request)

```powershell
$body = @{
  query = "Muss ich ein Nebenfach waehlen?"
  abstention = @{
    min_hits = 2
    min_best_rrf_score = 0.02
    min_dual_backend_hits = 1
    abstain_on_retrieval_errors = $true
    abstain_on_backend_degradation = $true
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8007/generate" `
  -ContentType "application/json" `
  -Body $body
```

## 9. Endpoints

Single-App Endpoints:
1. `GET /`
2. `GET /debug`
3. `GET /api/options`
4. `GET /api/ready`
5. `POST /api/chat`

Alle Retrieval-Services haben dieselben Kernendpoints:
1. `GET /health`
2. `GET /ready`
3. `POST /index/rebuild`
4. `POST /search`

Swagger:
1. Sparse: `http://127.0.0.1:8004/docs`
2. Vector: `http://127.0.0.1:8005/docs`
3. Hybrid: `http://127.0.0.1:8006/docs`
4. Generator: `http://127.0.0.1:8007/docs`
5. MARley Single-App: `http://127.0.0.1:8010/docs`

Generator-Kernendpoints:
1. `GET /health`
2. `GET /ready`
3. `POST /generate`

## 10. Typische Fehler

1. `ModuleNotFoundError` bei Chroma/Vector:
   Ursache: falscher Interpreter (global statt `.venv`).
2. Vector meldet Chroma nicht erreichbar:
   Ursache: Chroma auf Port `8000` nicht gestartet.
3. `SOURCE_FILE_MISSING` Flag:
   Wenn `faq_sb` fehlt, ist das erwartetes Verhalten. Pipeline laeuft weiter.
4. Umlaute im Request-Body:
   Bei Bedarf JSON explizit UTF-8 senden (z. B. per Datei + `curl --data-binary`).
5. Generator liefert `503`:
   Ursache: Hybrid-Retrieval auf Port `8006` oder Ollama nicht erreichbar.
6. Generator abstaint zu haeufig:
   Loesung: `abstention`-Schwellen im Request oder per `GENERATOR_*` Config anpassen.
7. Vector in Single-App langsam beim ersten Lauf:
   Ursache: Embeddings/Index werden initial aufgebaut. Danach sind Folgeanfragen schneller.

## 11. Aktueller Pipeline-Stand

Du kannst aktuell End-to-End bis Generator produktiv ausfuehren:
1. PDF -> `pdf_extractor`
2. Extractor-JSON -> `pdf_chunker`
3. FAQ-JSON -> `faq_chunker`
4. Chunks -> `sparse` / `vector`
5. `sparse + vector` -> `hybrid (RRF)`
6. `hybrid hits + query` -> `generator`
