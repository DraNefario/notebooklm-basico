#!/usr/bin/env python3
"""
Avalia o pipeline RAG com métricas Ragas.

Uso:
    python -m evals.rodar
    python -m evals.rodar --top-k 5
    python -m evals.rodar --somente-recuperacao   # só métricas de contexto
"""

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

_DATASET = pathlib.Path(__file__).parent / "dataset.json"
_RESULTADOS = pathlib.Path(__file__).parent / "resultados.json"


def _criar_judge():
    """Cria o LLM juiz: Gemini > Groq > Ollama, nessa ordem de preferência."""
    from ragas.llms import llm_factory

    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    provider = os.environ.get("EVAL_JUDGE_PROVIDER", "").lower()
    modelo = os.environ.get("EVAL_JUDGE_MODEL", "")

    if provider == "gemini" or (not provider and gemini_key):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = modelo or "gemini-3.1-flash-lite"
        print(f"[juiz] Gemini: {model}")
        return llm_factory(model, provider="openai", client=client)

    if provider == "groq" or (not provider and groq_key):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = modelo or "llama-3.3-70b-versatile"
        print(f"[juiz] Groq: {model}")
        return llm_factory(model, provider="openai", client=client)

    host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = modelo or "llama3.1:8b"
    print(f"[juiz] Ollama ({host}): {model}")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="ollama", base_url=f"{host}/v1")
    return llm_factory(model, provider="openai", client=client)


def _criar_metricas(judge, embeddings, somente_recuperacao=False):
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextEntityRecall,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
        SemanticSimilarity,
    )

    metricas = {
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
        "context_entity_recall": ContextEntityRecall(llm=judge),
    }
    if not somente_recuperacao:
        metricas.update(
            {
                "faithfulness": Faithfulness(llm=judge),
                "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
                "answer_correctness": AnswerCorrectness(llm=judge, embeddings=embeddings),
                "noise_sensitivity": NoiseSensitivity(llm=judge),
                "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
            }
        )
    return metricas


# Cada métrica aceita só seus próprios kwargs.
_KWARGS_POR_METRICA = {
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "reference", "retrieved_contexts"),
    "context_entity_recall": ("reference", "retrieved_contexts"),
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "answer_correctness": ("user_input", "response", "reference"),
    "noise_sensitivity": ("user_input", "response", "reference", "retrieved_contexts"),
    "semantic_similarity": ("reference", "response"),
}


def _avaliar_metrica(nome, metrica, resultado, tentativas=3):
    """Roda um .score() só com os kwargs que a métrica aceita; retenta em 429 com backoff."""
    chaves = _KWARGS_POR_METRICA[nome]
    kwargs = {k: resultado[k] for k in chaves}
    for tentativa in range(tentativas):
        try:
            r = metrica.score(**kwargs)
            return float(r.value)
        except Exception as e:
            eh_rate_limit = "429" in str(e) or "RateLimit" in type(e).__name__
            if eh_rate_limit and tentativa < tentativas - 1:
                espera = 15 * (tentativa + 1)
                print(f"    [~] {nome}: rate limit, retry em {espera}s...")
                time.sleep(espera)
                continue
            print(f"    [!] {nome}: {type(e).__name__}: {str(e)[:200]}")
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Avalia o RAG com Ragas")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--somente-recuperacao", action="store_true")
    args = parser.parse_args()

    from rag import NotebookRAG
    from ragas.embeddings import HuggingFaceEmbeddings

    dataset = json.loads(_DATASET.read_text(encoding="utf-8"))
    print(f"Dataset: {len(dataset)} perguntas | top_k={args.top_k}\n")

    print("Carregando pipeline RAG...")
    t0 = time.time()
    rag = NotebookRAG(top_k=args.top_k)
    print(f"Pipeline pronto em {time.time() - t0:.1f}s\n")

    judge = _criar_judge()
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    metricas = _criar_metricas(judge, embeddings, args.somente_recuperacao)

    resultados_por_metrica: dict[str, list[float]] = {k: [] for k in metricas}
    detalhes = []

    for i, item in enumerate(dataset, 1):
        pergunta = item["pergunta"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(dataset)}] {pergunta}")

        resp = rag.consultar(pergunta)
        print(f"    resposta: {resp.resposta[:100]}...")
        print(f"    contextos: {len(resp.contextos)}")

        kwargs_base = {
            "user_input": pergunta,
            "response": resp.resposta,
            "retrieved_contexts": resp.contextos,
            "reference": ground_truth,
        }

        scores = {}
        for nome, metrica in metricas.items():
            t1 = time.time()
            score = _avaliar_metrica(nome, metrica, kwargs_base)
            dt = time.time() - t1
            scores[nome] = score
            if score is not None:
                resultados_por_metrica[nome].append(score)
                print(f"    {nome}: {score:.3f} ({dt:.1f}s)")
            else:
                print(f"    {nome}: falhou ({dt:.1f}s)")
            time.sleep(1)  # respeita rate limit do free tier

        detalhes.append(
            {
                "pergunta": pergunta,
                "resposta": resp.resposta,
                "contextos": resp.contextos,
                "scores": scores,
            }
        )
        print()

    # Relatório
    print("=" * 55)
    print(f"{'MÉTRICA':<28} {'MÉDIA':>8} {'N':>4}")
    print("-" * 55)
    medias = {}
    for nome, valores in resultados_por_metrica.items():
        if valores:
            media = sum(valores) / len(valores)
            medias[nome] = round(media, 4)
            print(f"{nome:<28} {media:>8.3f} {len(valores):>4}")
        else:
            medias[nome] = None
            print(f"{nome:<28} {'—':>8} {0:>4}")
    print("=" * 55)

    _RESULTADOS.write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "somente_recuperacao": args.somente_recuperacao,
                "n_perguntas": len(dataset),
                "medias": medias,
                "detalhes": detalhes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResultados salvos em {_RESULTADOS}")


if __name__ == "__main__":
    main()
