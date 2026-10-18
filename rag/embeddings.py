from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Embedder:
    modelo_nome: str = DEFAULT_MODEL

    def __post_init__(self):
        print(f"Carregando modelo de embeddings '{self.modelo_nome}'...")
        self._model = SentenceTransformer(self.modelo_nome)

    def encode(self, chunks: List[str]) -> List[List[float]]:
        return self._model.encode(chunks).tolist()

    def encode_one(self, texto: str) -> List[float]:
        return self.encode([texto])[0]
