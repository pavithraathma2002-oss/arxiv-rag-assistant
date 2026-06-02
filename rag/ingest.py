"""
ingest.py — Load, chunk, embed and store PDF into ChromaDB.
Supports fixed and semantic chunking strategies.
"""

import re
import uuid
import tempfile
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from sentence_transformers import SentenceTransformer

# Local embedding model — free, no API key needed
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def extract_text_by_page(pdf_file) -> list[dict]:
    """Extract text from PDF, return list of {page, text} dicts."""
    pages = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    with pdfplumber.open(tmp_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})

    return pages


def fixed_chunk(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split pages into fixed-size token chunks with overlap."""
    chunks = []
    for page_data in pages:
        words = page_data["text"].split()
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_data["page"],
                "chunk_id": str(uuid.uuid4()),
            })
    return chunks


def semantic_chunk(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split on sentence boundaries, respecting chunk_size."""
    chunks = []
    sentence_end = re.compile(r'(?<=[.!?])\s+')

    for page_data in pages:
        sentences = sentence_end.split(page_data["text"])
        current, current_words = [], 0

        for sent in sentences:
            words = sent.split()
            if current_words + len(words) > chunk_size and current:
                chunks.append({
                    "text": " ".join(current),
                    "page": page_data["page"],
                    "chunk_id": str(uuid.uuid4()),
                })
                # keep overlap
                overlap_words = " ".join(current).split()[-overlap:] if overlap > 0 else []
                current = overlap_words + words
                current_words = len(current)
            else:
                current.extend(words)
                current_words += len(words)

        if current:
            chunks.append({
                "text": " ".join(current),
                "page": page_data["page"],
                "chunk_id": str(uuid.uuid4()),
            })

    return chunks


def ingest_pdf(
    pdf_file,
    chunk_size: int = 512,
    overlap: int = 50,
    strategy: Literal["fixed", "semantic"] = "fixed",
) -> str:
    """
    Full ingest pipeline:
    1. Extract text from PDF
    2. Chunk (fixed or semantic)
    3. Embed with SentenceTransformer
    4. Store in ChromaDB

    Returns: collection_name
    """
    collection_name = f"paper_{uuid.uuid4().hex[:8]}"

    # 1. Extract
    pages = extract_text_by_page(pdf_file)
    if not pages:
        raise ValueError("Could not extract text from PDF.")

    # 2. Chunk
    if strategy == "semantic":
        chunks = semantic_chunk(pages, chunk_size, overlap)
    else:
        chunks = fixed_chunk(pages, chunk_size, overlap)

    if not chunks:
        raise ValueError("No chunks generated.")

    # 3. Embed
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # 4. Store in ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in chunks],
    )

    return collection_name
