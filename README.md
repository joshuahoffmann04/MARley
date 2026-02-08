# MARley


1. Titel & kurze Beschreibung

Projektname: MARley: Ein ChatBot zur Studienberatung

Beschreibung: MARley ist ein ChatBot zur Studienberatung. Konkret wird dieser ChatBot für den "M.Sc. Computer Science" an der Philipps-Universität Marburg verwendet. Der ChatBot besteht dabei aus insgesamt 4 Komponenten. 
Die erste Komponente ist der PDF-Extractor. Die bereitgestellte PDF wird mittel PyMuPDF und pdfplumber (für Tabellen) als JSON eingelesen. Ein möglichst vollständiges Datenmodell und eine korrekte Erkennung der Abschnitte, Paragraphen und Anhänge ermöglichen erhalten die Struktur der Studienordnung und bilden später eine von drei Wissensbasen für den ChatBot
Die anderen beiden Wissensbasen sind FAQs. Eine davon wird aus der Studienordnung erstellt und hat das Ziel den gesamten Inhalt der Studienordnung abzubilden. Das zweite FAQ kommt aus der Studienberatung und enthält typische Fragen von realen Studierenden und die passenden Antworten. Diese 3 Wissensbasen bilden das Fundament für die zweite Komponente.
Die zweite Komponente ist der Chunker. Die 3 Wissensbasen werden als JSON eingelesen und in kleinere Chunks aufgeteilt. Für das Chunking werden zwei verschiedene Chunker verwendet. Zum einen gibt es den pdf_chunker. Hier wird der eingelesene Text aus der PDF mithilfe von tiktoken und einem sentencesplitter in möglichst gleichgroße Chunks aufgeteilt. Die min. und max. größe der Chunks ist hier variabel einstellbar. Der faq_chunker ist der Zweite. Hier wird aus jeder Frage ein eigenständiger Chunk gebaut und die Größe des Chunks wird mittels tiktoken zwar ermittelt, aber hat keine auswirkungen auf die Pipeline. Die erstellten Chunks dienen als Grundlage für die dritte Komponente.
Die dritte Komponente ist der Retrieval, der aus drei verschiedenen Möglichkeiten besteht. Die erste ist ein Sparse-Retrieval (BM25). Die zweite ist ein Vektor-Retrieval (ChromaDB). Die dritte ist ein Hybrid-Retrieval (RRF). Die TOP-Treffer werden hier mit allen Metadaten an die vierte Komponente weitergegeben.
Die vierte Komponente ist der Generator. Aus der Nutzeranfrage und den passenden Chunks wird dann eine korrekte Antwort ermittelt. Die Anzahl der verwendeten Token ist hier variabel und die verwendeten Chunks werden dementsprechend abgeschnitten.

2. Badges (optional)

Build-Status

Version

Lizenz

Downloads

3. Inhaltsverzeichnis 

Hilfreich bei längeren README-Dateien.

4. Installation / Start

Voraussetzungen

Installationsschritte

Start

5. Configuration

Umgebungsvariablen

Konfigurationsdateien