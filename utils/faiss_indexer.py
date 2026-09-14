"""Vector store for schema + unstructured document retrieval (used by Agent 1).

Builds one FAISS ``IndexFlatL2`` over two kinds of text chunks:

* ``schema`` chunks - one per database table (name + columns), generated
  directly from the live SQLite schema so it can never drift from the real
  database.
* ``document`` chunks - paragraphs pulled from the unstructured documents in
  ``data/documents/`` (business glossary, KPI definitions, schema notes...).

This is the small, fast index the live Streamlit app rebuilds on every
session start (a few dozen chunks, sub-second). For a demonstration of
retrieval latency at production scale (50,000+ vectors) see
``scripts/benchmark_vector_search.py``, which builds a separate large-scale
FAISS index and reports real measured p50/p95 latency - mixing that into the
interactive app's startup path would make every page load slow for no
benefit.
"""

import os
import re
import glob
import sqlite3
import time

import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_CACHE = {}


def _get_model():
    """Loads the sentence-transformer model once per process and reuses it."""
    if _MODEL_NAME not in _MODEL_CACHE:
        _MODEL_CACHE[_MODEL_NAME] = SentenceTransformer(_MODEL_NAME)
    return _MODEL_CACHE[_MODEL_NAME]


def _chunk_document(path: str, max_chars: int = 500):
    """Splits a text/markdown file into paragraph-sized chunks (~max_chars)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    chunks = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > max_chars:
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


class VectorIndexer:
    """FAISS-backed semantic index over schema definitions and documents."""

    def __init__(self, db_path: str = "data/chinook.db", docs_dir: str = "data/documents"):
        self.model = _get_model()
        self.index = None
        self.docs = []  # list of dicts: {id, text, type, source}
        self.build_latency_ms = None
        self._build_index(db_path, docs_dir)

    def _add_schema_docs(self, db_path: str):
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

            for table in tables:
                cursor.execute(f"PRAGMA table_info('{table}');")
                cols = [col[1] for col in cursor.fetchall()]
                doc_text = f"Table: {table}, Columns: {', '.join(cols)}"
                self.docs.append({"id": table, "text": doc_text, "type": "schema", "source": db_path})
        finally:
            conn.close()

    def _add_document_chunks(self, docs_dir: str):
        if not os.path.isdir(docs_dir):
            return
        for path in sorted(glob.glob(os.path.join(docs_dir, "*"))):
            if not path.lower().endswith((".txt", ".md")):
                continue
            for i, chunk in enumerate(_chunk_document(path)):
                doc_id = f"{os.path.basename(path)}#{i}"
                self.docs.append({"id": doc_id, "text": chunk, "type": "document", "source": path})

    def _build_index(self, db_path: str, docs_dir: str):
        start = time.perf_counter()
        self._add_schema_docs(db_path)
        self._add_document_chunks(docs_dir)

        if not self.docs:
            # Nothing to index yet - keep an empty-but-valid index instead of
            # crashing the whole app (this is what previously surfaced as a
            # confusing "'NoneType' object is not iterable" error downstream).
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.build_latency_ms = (time.perf_counter() - start) * 1000
            return

        embeddings = self.model.encode([d["text"] for d in self.docs], show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.build_latency_ms = (time.perf_counter() - start) * 1000

    def search(self, query: str, top_k: int = 3, doc_type: str = None):
        """Returns up to ``top_k`` nearest chunks, optionally filtered by
        ``doc_type`` ('schema' or 'document'). Never raises on an empty
        index - callers get an empty list instead of a crash.
        """
        if self.index is None or self.index.ntotal == 0 or not self.docs:
            return []

        query_vector = self.model.encode([query])
        k = min(top_k * 3 if doc_type else top_k, self.index.ntotal)
        _, indices = self.index.search(query_vector, k)

        results = []
        for i in indices[0]:
            if i < 0 or i >= len(self.docs):
                continue
            doc = self.docs[i]
            if doc_type and doc["type"] != doc_type:
                continue
            results.append(doc)
            if len(results) >= top_k:
                break
        return results
