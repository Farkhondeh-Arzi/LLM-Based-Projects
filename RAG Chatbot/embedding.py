import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import Chunk


class EmbeddingModel:

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        print(f"Loading model {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,  # To calculate cosine similarity easier
            show_progress_bar=len(texts) > 20,
        )

    def embed_query(self, query: str) -> np.ndarray:
        prefixed = f"query: {query}"
        return self.model.encode([prefixed], normalize_embeddings=True)[0]


def embed_chunks(chunks: list[Chunk], model: EmbeddingModel) -> np.ndarray:
    texts = [c.text for c in chunks]
    return model.embed_passages(texts)
