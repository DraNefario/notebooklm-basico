from dataclasses import dataclass

import numpy as np


@dataclass
class MemoriaRAG:
    """Vetores + chunks em memória"""

    textos: list[str]
    vetores: list[list[float]]

    def recuperar(self, vetor_pergunta: list[float], top_k: int = 3) -> list[str]:
        if not self.textos or not self.vetores:
            return []

        vetor_pergunta = np.array(vetor_pergunta)
        vetores_memoria = np.array(self.vetores)

        pergunta_normalizada = np.linalg.norm(vetor_pergunta)
        memoria_normalizada = np.linalg.norm(vetores_memoria, axis=1)

        dot_products = np.dot(vetores_memoria, vetor_pergunta)
        similaridades = dot_products / (memoria_normalizada * pergunta_normalizada)

        top_k_indices = np.argsort(similaridades)[-top_k:][::-1]
        return [self.textos[idx] for idx in top_k_indices]
