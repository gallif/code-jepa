from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np


def build_ip_index(embeddings: np.ndarray) -> faiss.Index:
    """Inner-product index on L2-normalized vectors (= cosine)."""
    x = embeddings.astype(np.float32)
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def ranks_of_targets(
    queries: np.ndarray,
    index: faiss.Index,
    target_ids: Sequence[int],
    batch_size: int = 256,
) -> np.ndarray:
    """Return 1-based rank of each target id for the corresponding query."""
    q = queries.astype(np.float32).copy()
    faiss.normalize_L2(q)
    n = q.shape[0]
    # search all if corpus small enough
    k = index.ntotal
    ranks = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores, idxs = index.search(q[start:end], k)
        for i, tid in enumerate(target_ids[start:end]):
            hits = idxs[i]
            pos = np.where(hits == tid)[0]
            ranks[start + i] = int(pos[0] + 1) if len(pos) else k + 1
    return ranks


def recall_at_k(ranks: np.ndarray, ks: Sequence[int]) -> dict[int, float]:
    return {k: float((ranks <= k).mean()) for k in ks}
