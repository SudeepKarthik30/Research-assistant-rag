# ResearchRAG — Hybrid RAG Research Assistant

A research paper Q&A tool that combines **dense vector search** and **sparse BM25 retrieval** with **Reciprocal Rank Fusion** to deliver citation-grounded answers from uploaded PDFs.

Live Demo: [https://karthik0055-research-assistant-rag.hf.space](https://karthik0055-research-assistant-rag.hf.space)

---

## Features

- **Hybrid retrieval** — semantic (ChromaDB + sentence-transformers) + BM25 fused via RRF
- **Section-aware chunking** — detects Abstract, Methods, Results, etc. as metadata
- **Citation-grounded answers** — every claim includes `[Source N]` with page numbers
- **Paper management** — upload, list, and remove indexed papers from the UI
- **Markdown rendering** — answers display with proper formatting, bold, and bullet points
- **Text summarizer** — paste any text for a concise 2–3 sentence summary
- **General Q&A** — ask anything outside the uploaded papers

---

## User Interface

[ResearchRAG User Interface](UI.png)

The Streamlit app provides two primary modes of interaction:

### 1. Research Assistant Mode
Ask complex questions about your literature corpus in plain English. The system performs a **Hybrid Search**—fusing semantic meaning via ChromaDB with keyword matching via BM25—to retrieve specific evidence and deliver answers grounded in the text.

### 2. Knowledge Management
Use the sidebar tools to maintain your research library:
- **Upload PDFs** — Automatically parse documents into section-aware chunks (Abstract, Methods, Results) using PyMuPDF.
- **Manage Index** — View a list of currently indexed papers or remove specific documents to update your persistent vector store.
- **Summarize Text** — Access a standalone summarizer to get concise 2–3 sentence insights from abstracts or external text snippets.

## Architecture

```
PDF Upload
    │
    ▼
PyMuPDF (text extraction)
    │
    ▼
Section Detection (Abstract / Methods / Results / ...)
    │
    ▼
LangChain RecursiveCharacterTextSplitter (semantic boundaries)
    │
    ├──► ChromaDB (dense vectors via sentence-transformers/all-MiniLM-L6-v2)
    │
    └──► BM25Okapi (sparse, in-memory)

Query
    │
    ├──► Semantic search (cosine similarity)
    │
    ├──► BM25 search (keyword matching)
    │
    ▼
Reciprocal Rank Fusion  score = Σ 1/(rank + k)
    │
    ▼
Top-K chunks → Groq Llama 3.3 70B
    │
    ▼
Citation-grounded answer [Source N]
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/research-rag
cd research-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and paste your Groq API key (free at console.groq.com)

# 5. Run
python main.py
# Open http://localhost:5000
```

---

## Retrieval Evaluation

A small evaluation framework measures retrieval quality against hand-labeled QA pairs.

```bash
# 1. Copy the example and fill in your own questions from a paper you've indexed
cp eval/test_set.example.json eval/test_set.json

# 2. Run the eval
python eval/run_eval.py
```

Sample output:
```
  [01] ✓ | pages=[4, 3, 5] | kw=100% | What dataset was used for training?
  [02] ✓ | pages=[1, 2, 1] | kw= 67% | What is the main contribution?
  ...
──────────────────────────────────────────────────
  Retrieval P@3  : 4/5 = 80%
  Keyword hit rate: 78%
──────────────────────────────────────────────────
Results saved to eval/results.json
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB (persistent) |
| Sparse retrieval | BM25Okapi (rank-bm25) |
| PDF parsing | PyMuPDF (fitz) |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Frontend | Vanilla JS, marked.js |

---

## Project Structure

```
research-rag/
├── main.py              # Flask routes
├── rag_pipeline.py      # Hybrid RAG pipeline
├── requirements.txt
├── .env.example
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── eval/
│   ├── run_eval.py          # Evaluation script
│   ├── test_set.example.json
│   └── results.json         # Generated after running eval
└── data/
    └── vector_store/        # ChromaDB persistent storage (auto-created)
```
