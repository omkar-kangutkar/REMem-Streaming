
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        print(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def encode(self, texts, batch_size=32, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, batch_size=batch_size,
                                        normalize_embeddings=True,
                                        show_progress_bar=False)
        return embeddings.tolist()

    def get_embeddings(self, texts):
        return self.encode(texts)
