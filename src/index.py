"""
In-memory embedding index for a single document's chunks.

Uses numpy cosine similarity rather than a vector database, appropriate
for the scale of individual contracts. The encoder is injectable
(`encode_fn`) so tests can substitute a deterministic fake.
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.models import Chunk

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

EncodeFn = Callable[[List[str]], np.ndarray]


def _load_default_encoder() -> EncodeFn:
    """Lazily import and load the local embedding model. Only touched if
    no custom encode_fn is supplied, so importing this module never
    requires sentence-transformers to be loaded (or network access)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DEFAULT_MODEL_NAME)

    def encode(texts: List[str]) -> np.ndarray:
        return np.array(model.encode(list(texts)))

    return encode


class EmbeddedIndex:
    """Embeds a document's chunks once; supports repeated `.search()`
    calls for multiple questions against the same document without
    re-embedding anything."""

    def __init__(self, chunks: List[Chunk], encode_fn: Optional[EncodeFn] = None):
        self.chunks = chunks
        self._encode_fn = encode_fn or _load_default_encoder()
        if chunks:
            self.vectors = self._encode_fn([c.text for c in chunks])
        else:
            self.vectors = np.zeros((0, 1))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (Chunk, cosine_similarity) pairs, sorted by
        similarity descending. Empty list if the index has no chunks."""
        if not self.chunks:
            return []

        query_vec = self._encode_fn([query])[0]
        sims = _cosine_similarity(self.vectors, query_vec)

        top_k = max(1, min(top_k, len(self.chunks)))
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_indices]


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of `matrix` and `vector`.
    Computed explicitly (not assuming pre-normalized embeddings) so this
    is correct regardless of what encoder is plugged in."""
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = matrix_norms * vector_norm
    denom[denom == 0] = 1e-10  # avoid divide-by-zero for empty/zero vectors
    return (matrix @ vector) / denom
