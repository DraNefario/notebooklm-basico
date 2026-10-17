#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NotebookLM simples — RAG na mão.
Passos: 1) carregar texto, 2) chunkar, 3) embed, 4) armazenar, 5) recuperar, 6) perguntar ao LLM.
"""

import pathlib, sys, os
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_model_instance = None


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> List[str]:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i+size]))
        i += max(size - overlap, 1)
    return [chunk for chunk in out if chunk.strip()]


def carregar_arquivos(pasta: pathlib.Path) -> str:
    """Lê todos os .txt da pasta e retorna o texto concatenado."""
    texto_total = ""
    for arquivo in sorted(pasta.rglob("*.txt")):
        print(f"Lendo {arquivo.name}...")
        texto_total += arquivo.read_text(encoding="utf-8") + "\n"
    return texto_total


def get_embedding_model(modelo_nome: str = _DEFAULT_MODEL) -> SentenceTransformer:
    """
    Retorna o modelo de embeddings carregado.
    """
    global _model_instance
    if _model_instance is None:
        print(f"Carregando modelo de embeddings '{modelo_nome}'...")
        _model_instance = SentenceTransformer(modelo_nome)
    return _model_instance


def gerar_embeddings(chunks: List[str], modelo_nome: str = _DEFAULT_MODEL) -> List[List[float]]:
    """
    Retorna uma lista de vetores (uma por chunk) usando o modelo já carregado.
    """
    model = get_embedding_model(modelo_nome)
    embeddings = model.encode(chunks)
    return embeddings.tolist()


@dataclass
class MemoriaRAG:
    """Vetores + chunks em memória"""
    textos: List[str]        # os chunks originais
    vetores: List[List[float]]  # embeddings correspondentes

    def recuperar(self, pergunta: str, top_k: int = 3) -> List[str]:
        """
        Dada uma pergunta, retorna os top_k chunks mais relevantes.
        """
        if not self.textos or not self.vetores:
            return []

        # 1. Gera o embedding da pergunta usando gerar_embeddings
        vetor_pergunta = np.array(gerar_embeddings([pergunta])[0])
        vetores_memoria = np.array(self.vetores)

        # 2. Calcula similaridade (cosine similarity)
        # Fórmula: (A . B) / (||A|| * ||B||)
        pergunta_normalizada = np.linalg.norm(vetor_pergunta)
        memoria_normalizada = np.linalg.norm(vetores_memoria, axis=1)

        #produtos escalares
        dot_products = np.dot(vetores_memoria, vetor_pergunta)

        #similaridade do cosseno para cada chunk
        similaridades = dot_products / (memoria_normalizada * pergunta_normalizada)

        # 3. Pegar os índices dos top_k maiores valores
        # np.argsort ordena do menor para o maior, então pegamos os últimos top_k e invertemos [::-1]
        top_k_indices = np.argsort(similaridades)[-top_k:][::-1]

        # 4. Retornar os textos correspondentes
        return [self.textos[idx] for idx in top_k_indices]



def main():
    base_docs = pathlib.Path("docs")  # pasta onde você põe seus arquivos .txt
    if not base_docs.exists():
        print(f"Crie a pasta '{base_docs}' e coloque seus arquivos .txt nela.")
        return

    # 1) Carregar todos os textos
    texto_total = carregar_arquivos(base_docs)

    # 2) Quebrar em chunks
    chunks = chunk_text(texto_total, size=512, overlap=64)

    # 3) Gerar embeddings
    embeddings = gerar_embeddings(chunks)

    # 4) Guardar na memória
    memoria = MemoriaRAG(textos=chunks, vetores=embeddings)

    print("\n=== NotebookLM pronto ===")
    print("Digite suas perguntas (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break

        # 5) Recuperar trechos relevantes
        trechos_relevantes = memoria.recuperar(pergunta, top_k=3)

        # 6) Montar prompt para o LLM
        #    (aqui você monta o contexto + pergunta e chama Ollama)
        contexto = "\n".join(trechos_relevantes)
        prompt = f"""Use o contexto abaixo para responder a pergunta do usuário.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

Responda de forma direta e educativa, usando apenas o contexto acima. Se a resposta não estiver no contexto, diga que não tem informação suficiente."""
        
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        if provider == "ollama":
            from ollama import Client
            model_name = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5-coder:1.5b")
            client = Client(host=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            resposta = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            print(f"LLM: {resposta.message.content}")


if __name__ == "__main__":
    main()