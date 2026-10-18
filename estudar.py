#!/usr/bin/env python3
"""
NotebookLM simples — RAG na mão.
Passos: 1) carregar texto, 2) chunkar, 3) embed, 4) armazenar, 5) recuperar, 6) perguntar ao LLM.
"""

import pathlib

from rag import (
    Embedder,
    MemoriaRAG,
    carregar_arquivos,
    carregar_settings,
    chunk_text,
    criar_llm,
)

_PROMPT = """Use o contexto abaixo para responder a pergunta do usuário.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

Responda de forma direta e educativa, com citacoes [1],[2] das fontes, usando apenas o contexto acima. Se a resposta não estiver no contexto, diga que não tem informação suficiente."""


def main():
    base_docs = pathlib.Path("docs")
    if not base_docs.exists():
        print(f"Crie a pasta '{base_docs}' e coloque seus arquivos .txt nela.")
        return

    settings = carregar_settings()
    llm = criar_llm(settings)

    texto_total = carregar_arquivos(base_docs)
    chunks = chunk_text(texto_total, size=512, overlap=64)

    embedder = Embedder()
    embeddings = embedder.encode(chunks)
    memoria = MemoriaRAG(textos=chunks, vetores=embeddings)

    print("\n=== NotebookLM pronto ===")
    print("Digite suas perguntas (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break

        trechos = memoria.recuperar(embedder.encode_one(pergunta), top_k=3)
        prompt = _PROMPT.format(contexto="\n".join(trechos), pergunta=pergunta)
        print(f"LLM: {llm.responder(prompt)}")


if __name__ == "__main__":
    main()
