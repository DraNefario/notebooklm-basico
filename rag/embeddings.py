from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Embedder:
    modelo_nome: str = DEFAULT_MODEL

    def __post_init__(self):
        print(f"Carregando modelo de embeddings '{self.modelo_nome}'...")
        self._model = SentenceTransformer(self.modelo_nome)

    def encode(self, chunks: list[str]) -> list[list[float]]:
        return self._model.encode(chunks).tolist()

    def encode_one(self, texto: str) -> list[float]:
        return self.encode([texto])[0]
