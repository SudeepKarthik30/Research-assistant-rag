"""
rag_pipeline.py — Hybrid RAG pipeline for research paper analysis.

Architecture:
  PDF -> PyMuPDF -> LangChain RecursiveCharacterTextSplitter -> ChromaDB + BM25
  Query -> RRF fusion of both retrieval lists -> Groq Llama 3.3 70B
  -> Citation-grounded answer [Source N] with metadata
"""

import os
import re
import uuid
from typing import Dict, List, Optional

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_PDF_MB = 10   # reject PDFs larger than this
MAX_CHUNKS = 600  # safety cap — prevents embedding timeout on free tier

ACADEMIC_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "literature review", "methodology", "methods", "experimental setup",
    "experiments", "results", "evaluation", "discussion", "limitations",
    "conclusion", "conclusions", "future work", "references",
    "acknowledgements", "appendix",
]

# ── Section detection ─────────────────────────────────────────────────────────

def detect_section(line: str) -> Optional[str]:
    stripped = line.strip().lower()
    cleaned  = re.sub(r"^\d+(\.\d+)*\.?\s*", "", stripped)
    for section in ACADEMIC_SECTIONS:
        if cleaned.startswith(section):
            return section.title()
    return None

# ── PDF loading ───────────────────────────────────────────────────────────────

def load_pdf_chunks(pdf_path: str, filename: str, chunk_size: int = 600, overlap: int = 100) -> List[Dict]:    
    """
    Extract and chunk text from a PDF using PyMuPDF + LangChain splitter.
    Respects semantic boundaries (paragraphs → sentences → words).
    Returns list of {"text": str, "metadata": {...}} dicts.
    Raises ValueError if the file exceeds MAX_PDF_MB.
    """
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise ValueError(
            f"PDF is {size_mb:.1f} MB — exceeds the {MAX_PDF_MB} MB limit. "
            "Please upload a smaller file."
        )

    doc = fitz.open(pdf_path)
    chunks: List[Dict] = []
    current_section = "Body"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        lines    = [l.strip() for l in raw_text.split("\n") if l.strip()]

        page_parts = []
        for line in lines:
            sec = detect_section(line)
            if sec:
                current_section = sec
            page_parts.append(line)

        page_text   = " ".join(page_parts)
        page_chunks = text_splitter.split_text(page_text)

        for chunk_text in page_chunks:
            if len(chunk_text) > 50:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source":      filename,
                        "page":        page_num,
                        "section":     current_section,
                        "chunk_index": len(chunks),
                    },
                })
                if len(chunks) >= MAX_CHUNKS:
                    print(f"[RAG] Chunk cap ({MAX_CHUNKS}) reached — truncating.")
                    doc.close()
                    return chunks

    doc.close()
    print(f"[RAG] Extracted {len(chunks)} chunks from '{filename}' ({size_mb:.1f} MB)")
    return chunks

# ── Main pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Hybrid RAG pipeline.

    Storage  : ChromaDB (persistent) + BM25Okapi (in-memory)
    Retrieval: RRF fusion — score(d) = Σ 1/(rank + k), k=60
    Generation: Groq Llama 3.3 70B, temperature=0.1, citation-enforced
    """

    def __init__(
        self,
        persist_dir: str     = "data/vector_store",
        collection_name: str = "research_papers",
    ):
        os.makedirs(persist_dir, exist_ok=True)

        print("[RAG] Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection    = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[Dict]   = []
        self._load_bm25_from_db()

        print(f"[RAG] Ready — {self.collection.count()} chunks in store.")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _load_bm25_from_db(self) -> None:
        """Load existing corpus from ChromaDB into memory on startup."""
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs   = result.get("documents", [])
        metas  = result.get("metadatas", [])
        ids    = result.get("ids", [])

        if docs:
            self._bm25_corpus = [
                {"id": i, "text": d, "metadata": m}
                for i, d, m in zip(ids, docs, metas)
            ]
            tokenized  = [d["text"].lower().split() for d in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25        = None
            self._bm25_corpus = []

    def _append_to_bm25(self, new_chunks: List[Dict]) -> None:
        """Extend the in-memory BM25 corpus without re-fetching from ChromaDB."""
        self._bm25_corpus.extend(new_chunks)
        tokenized  = [d["text"].lower().split() for d in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def get_indexed_papers(self) -> List[str]:
        """Return a sorted list of unique filenames currently indexed."""
        result = self.collection.get(include=["metadatas"])
        metas  = result.get("metadatas", [])
        return sorted({m["source"] for m in metas if m})

    def is_indexed(self, filename: str) -> bool:
        """Check if a paper is already in the store."""
        result = self.collection.get(where={"source": filename}, include=["metadatas"])
        return len(result.get("ids", [])) > 0

    def add_pdf(self, pdf_path: str, original_filename: str = None) -> int:
        """
        Ingest a PDF — size-checked, chunked, embedded, stored.
        Returns number of chunks added.
        Raises ValueError for oversized or already-indexed files.
        """
        filename = original_filename if original_filename else os.path.basename(pdf_path)

        if self.is_indexed(filename):
            raise ValueError(f"'{filename}' is already indexed. Remove it first to re-index.")

        chunks = load_pdf_chunks(pdf_path, filename)
        if not chunks:
            return 0

        texts     = [c["text"]     for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [f"chunk_{uuid.uuid4()}" for _ in chunks]

        print(f"[RAG] Embedding {len(texts)} chunks...")
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        new_docs = [{"id": i, "text": t, "metadata": m} for i, t, m in zip(ids, texts, metadatas)]
        self._append_to_bm25(new_docs)

        print(f"[RAG] Added {len(chunks)} chunks. Total: {self.collection.count()}")
        return len(chunks)

    def remove_paper(self, filename: str) -> int:
        """Delete all chunks for a given filename. Returns number of chunks removed."""
        result = self.collection.get(where={"source": filename}, include=["metadatas"])
        ids_to_delete = result.get("ids", [])
        if not ids_to_delete:
            return 0
        self.collection.delete(ids=ids_to_delete)
        # Rebuild BM25 from DB after deletion
        self._load_bm25_from_db()
        print(f"[RAG] Removed {len(ids_to_delete)} chunks for '{filename}'.")
        return len(ids_to_delete)

    @property
    def doc_count(self) -> int:
        return self.collection.count()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        semantic_docs: List[Dict],
        bm25_docs: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion — keyed on unique chunk ID to avoid collisions."""
        scores: Dict[str, Dict] = {}

        for rank, item in enumerate(semantic_docs):
            key = item["id"]
            if key not in scores:
                scores[key] = {"score": 0.0, "item": item}
            scores[key]["score"] += 1.0 / (rank + 1 + k)

        for rank, item in enumerate(bm25_docs):
            key = item["id"]
            if key not in scores:
                scores[key] = {"score": 0.0, "item": item}
            scores[key]["score"] += 1.0 / (rank + 1 + k)

        return [
            e["item"]
            for e in sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []

        n_candidates = min(top_k * 2, total)

        # Semantic search
        query_emb  = self.embed_model.encode([query])[0]
        sem_result = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=n_candidates,
            include=["documents", "metadatas"],
        )
        semantic_docs = []
        if sem_result["documents"] and sem_result["documents"][0]:
            for doc_id, text, meta in zip(
                sem_result["ids"][0],
                sem_result["documents"][0],
                sem_result["metadatas"][0],
            ):
                semantic_docs.append({"id": doc_id, "text": text, "metadata": meta})

        # BM25 search
        bm25_docs = []
        if self._bm25 and self._bm25_corpus:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            top_idx     = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )
            bm25_docs = [
                self._bm25_corpus[i]
                for i in top_idx[:n_candidates]
                if bm25_scores[i] > 0
            ]

        return self._rrf_fuse(semantic_docs, bm25_docs)[:top_k]

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: int = 5) -> Dict:
        chunks = self.retrieve(query, top_k)

        if not chunks:
            return {
                "answer":    "No relevant content found. Please upload a research paper first.",
                "citations": [],
            }

        context_parts = []
        citations     = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            context_parts.append(
                f"[Source {i}] — {meta['source']}, p.{meta['page']} ({meta.get('section', 'Body')})\n"
                f"{chunk['text']}"
            )
            citations.append({
                "index":   i,
                "source":  meta["source"],
                "page":    meta["page"],
                "section": meta.get("section", "Body"),
            })

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a rigorous academic research assistant. "
            "Answer strictly from the provided context — never use outside knowledge. "
            "Format your answer using markdown: use **bold** for key terms, bullet points for lists, "
            "and short paragraphs for clarity. "
            "Every factual claim MUST include an inline citation like [Source 1] or [Source 2]. "
            "If the context lacks enough information, say so explicitly. "
            "Never fabricate information."
        )

        user_prompt = (
            f"Context from uploaded research papers:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a thorough, well-structured answer with inline citations [Source N] for every claim."
        )

        response = self.groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return {
            "answer":    response.choices[0].message.content.strip(),
            "citations": citations,
        }
