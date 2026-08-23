from .base import BaseEmbeddingModel, EmbeddingConfig
import numpy as np

def _get_embedding_client(global_config, embedding_model_name: str = "nvidia/NV-Embed-v2", openai_style_server=True):

    if "sentence-transformers" in embedding_model_name or "MiniLM" in embedding_model_name:
        from sentence_transformers import SentenceTransformer

        class LocalSTModel(BaseEmbeddingModel):
            def __init__(self, global_config=None, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.st_model = SentenceTransformer(embedding_model_name)
                self.embedding_dim = self.st_model.get_sentence_embedding_dimension()
                self.embedding_model_name = embedding_model_name
                print(f"Loaded local embedding model: {embedding_model_name}, dim={self.embedding_dim}")

            def batch_encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                embs = self.st_model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                return np.array(embs)

        return LocalSTModel(global_config=global_config, embedding_model_name=embedding_model_name)

    elif "text-embedding" in embedding_model_name:
        from .openai_embedding_client import CacheOpenAIEmbeddingModel
        return CacheOpenAIEmbeddingModel(None, global_config, embedding_model_name, base_url="https://api.openai.com/v1/")

    elif "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel
        return GritLMEmbeddingModel(global_config, embedding_model_name)

    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        return NVEmbedV2EmbeddingModel(global_config, embedding_model_name)

    else:
        from .openai_embedding_client import CacheOpenAIEmbeddingModel
        return CacheOpenAIEmbeddingModel(None, global_config, embedding_model_name, base_url="http://localhost:8001/v1/")
