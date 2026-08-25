from chromadb import Documents, EmbeddingFunction, Embeddings

import ollama

from config import EMBED_MODEL, OLLAMA_BASE_URL


class OllamaEmbedding(EmbeddingFunction):
    """使用本地 Ollama 嵌入模型将文本转为向量，保证完全离线。"""

    def __init__(self, model: str = EMBED_MODEL, host: str = OLLAMA_BASE_URL):
        self.model = model
        self.client = ollama.Client(host=host)

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for doc in input:
            resp = self.client.embeddings(model=self.model, prompt=doc)
            emb = resp.embedding if hasattr(resp, "embedding") else resp["embedding"]
            embeddings.append(emb)
        return embeddings
