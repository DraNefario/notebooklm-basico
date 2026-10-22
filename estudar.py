#!/usr/bin/env python3
"""
NotebookLM simples — RAG na mão.
Passos: 1) carregar texto, 2) chunkar, 3) embed, 4) armazenar, 5) recuperar, 6) perguntar ao LLM.
"""

from rag import NotebookRAG


def main():
    try:
        rag = NotebookRAG()
    except FileNotFoundError as e:
        print(e)
        return

    print("\n=== NotebookLM pronto ===")
    print("Digite suas perguntas (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até mais!")
            break
        print(f"LLM: {rag.consultar(pergunta).resposta}")


if __name__ == "__main__":
    main()
