"""Semantic gap analysis: sub-queries vs. content chunks via embeddings.

Model choice: all-MiniLM-L6-v2 (default). It is ~5× faster than
all-mpnet-base-v2 and this endpoint embeds every sentence of an article
synchronously inside a request — latency dominates. On semantic textual
similarity benchmarks MiniLM trails mpnet by a few points, which matters
less here because we only need a coarse covered/not-covered decision against
a threshold, not a fine-grained ranking. Swap via EMBEDDING_MODEL env var.

Cosine similarity: embeddings are encoded with normalize_embeddings=True, so
every vector has unit length and the dot product IS the cosine similarity —
mathematically exact, not an approximation.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from app.services.nlp import get_nlp

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.72

MIN_CHUNK_WORDS = 4
MAX_CHUNKS = 512  # cap embedding work for very long articles


def get_similarity_threshold() -> float:
    return float(os.getenv("SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))


@lru_cache(maxsize=1)
def get_embedding_model():
    # Imported lazily: sentence-transformers pulls in torch, which we don't
    # want to pay for on /api/aeo/analyze requests or at app startup.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))


def chunk_content(text: str) -> list[str]:
    """Split content into sentence-level chunks via spaCy.

    Sentences shorter than MIN_CHUNK_WORDS words (heading fragments, list
    bullets like "Pricing") are dropped — they embed poorly and can produce
    spurious high similarity on keyword overlap alone.
    """
    chunks: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        doc = get_nlp()(block)
        for sent in doc.sents:
            sentence = sent.text.strip()
            if len(sentence.split()) >= MIN_CHUNK_WORDS:
                chunks.append(sentence)
    return chunks[:MAX_CHUNKS]


def max_similarities(queries: list[str], content: str) -> list[float]:
    """For each query, the max cosine similarity against all content chunks.

    Returns 0.0 for every query if the content yields no usable chunks.
    """
    chunks = chunk_content(content)
    if not chunks or not queries:
        return [0.0] * len(queries)

    model = get_embedding_model()
    chunk_vectors = model.encode(chunks, normalize_embeddings=True)
    query_vectors = model.encode(queries, normalize_embeddings=True)

    # Unit vectors -> dot product == cosine similarity.
    similarity_matrix = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T
    return [round(float(row.max()), 2) for row in similarity_matrix]
