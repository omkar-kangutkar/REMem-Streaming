code = '''import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. Run: pip install faiss-cpu")


class HNSWSynonymyUpdater:
    """HNSW-based incremental synonymy edge updater."""

    def __init__(self, embedding_dim: int = 768, threshold: float = 0.8,
                 top_k: int = 10, hnsw_m: int = 32):
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.top_k = top_k
        self._idx_to_id: Dict[int, str] = {}
        self._count = 0
        self.search_times: List[float] = []
        self.insert_times: List[float] = []
        if FAISS_AVAILABLE:
            self._index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
            self._index.hnsw.efSearch = 32
            self._use_hnsw = True
            logger.info("HNSW index initialised (dim=%d, M=%d, threshold=%.2f)",
                        embedding_dim, hnsw_m, threshold)
        else:
            self._stored: List[Tuple] = []
            self._use_hnsw = False

    def update(self, new_gist_id: str, new_embedding: np.ndarray) -> List[Tuple[str, str]]:
        """Find synonymy pairs for a new gist and update the index."""
        emb = self._normalise(new_embedding)
        pairs = []
        if self._count > 0:
            t0 = time.perf_counter()
            similar_ids = self._search(emb)
            self.search_times.append(time.perf_counter() - t0)
            for existing_id in similar_ids:
                if existing_id != new_gist_id:
                    pairs.append((new_gist_id, existing_id))
        else:
            self.search_times.append(0.0)
        t0 = time.perf_counter()
        self._insert(new_gist_id, emb)
        self.insert_times.append(time.perf_counter() - t0)
        return pairs

    @property
    def num_gists(self) -> int:
        return self._count

    @property
    def avg_search_ms(self) -> Optional[float]:
        if not self.search_times:
            return None
        return (sum(self.search_times) / len(self.search_times)) * 1000

    @property
    def avg_insert_ms(self) -> Optional[float]:
        if not self.insert_times:
            return None
        return (sum(self.insert_times) / len(self.insert_times)) * 1000

    def complexity_report(self) -> dict:
        n = self._count
        return {
            "num_gists": n,
            "method": "HNSW" if self._use_hnsw else "brute-force",
            "approx_comparisons": int(np.log2(n + 1)) * self.top_k if n > 0 else 0,
            "avg_search_ms": self.avg_search_ms,
            "avg_insert_ms": self.avg_insert_ms,
        }

    def _normalise(self, embedding: np.ndarray) -> np.ndarray:
        emb = np.array(embedding, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return emb / norm

    def _search(self, emb: np.ndarray) -> List[str]:
        k = min(self.top_k, self._count)
        if self._use_hnsw:
            similarities, indices = self._index.search(emb, k)
            result = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0 and float(sim) >= self.threshold:
                    gist_id = self._idx_to_id.get(int(idx))
                    if gist_id:
                        result.append(gist_id)
            return result
        else:
            if not self._stored:
                return []
            stored_matrix = np.vstack([e for e, _ in self._stored])
            sims = np.dot(emb, stored_matrix.T)[0]
            return [self._stored[int(i)][1]
                    for i in np.where(sims >= self.threshold)[0]]

    def _insert(self, gist_id: str, emb: np.ndarray) -> None:
        self._idx_to_id[self._count] = gist_id
        self._count += 1
        if self._use_hnsw:
            self._index.add(emb)
        else:
            self._stored.append((emb, gist_id))


class BruteForceBaseline:
    """Brute-force O(n) synonymy baseline for ablation comparison."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._stored_embeddings: List[np.ndarray] = []
        self._stored_ids: List[str] = []
        self.search_times: List[float] = []

    def update(self, new_gist_id: str, new_embedding: np.ndarray) -> List[Tuple[str, str]]:
        emb = np.array(new_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        pairs = []
        if len(self._stored_embeddings) > 0:
            t0 = time.perf_counter()
            stored = np.vstack(self._stored_embeddings)
            sims = np.dot(stored, emb)
            self.search_times.append(time.perf_counter() - t0)
            for idx in np.where(sims >= self.threshold)[0]:
                pairs.append((new_gist_id, self._stored_ids[int(idx)]))
        else:
            self.search_times.append(0.0)
        self._stored_embeddings.append(emb)
        self._stored_ids.append(new_gist_id)
        return pairs

    @property
    def num_gists(self) -> int:
        return len(self._stored_ids)

    @property
    def avg_search_ms(self) -> Optional[float]:
        if not self.search_times:
            return None
        return (sum(self.search_times) / len(self.search_times)) * 1000
'''
open('hnsw_synonymy.py', 'w').write(code)
print("Done. Lines:", len(code.split('\n')))
