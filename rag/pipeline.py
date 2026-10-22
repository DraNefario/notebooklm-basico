import pathlib
from dataclasses import dataclass
from typing import List

from .config import Settings, carregar_settings
from .documents import carregar_arquivos, chunk_text
from .embeddings import Embedder
from .llm import ChatClient, criar_llm
from .store import MemoriaRAG

_PROMPT = """Use o contexto abaixo para responder a pergunta do usuário.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

Responda de forma direta e educativa, com citacoes [1],[2] das fontes, usando apenas o contexto acima. Se a resposta não estiver no contexto, diga que não tem informação suficiente."""


@dataclass
class Resposta:
    pergunta: str
    contextos: List[str]
    resposta: str


class NotebookRAG:
    def __init__(
        self,
        docs_dir: pathlib.Path = pathlib.Path("docs"),
        settings: Settings | None = None,
        top_k: int = 3,
    ):
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Pasta '{docs_dir}' não existe. Coloque seus arquivos .txt nela."
            )
        self.settings = settings or carregar_settings()
        self.top_k = top_k
        self.llm: ChatClient = criar_llm(self.settings)
        self.embedder = Embedder()

        texto_total = carregar_arquivos(docs_dir)
        chunks = chunk_text(texto_total, size=512, overlap=64)
        self.memoria = MemoriaRAG(
            textos=chunks, vetores=self.embedder.encode(chunks)
        )

    def recuperar(self, pergunta: str, top_k: int | None = None) -> List[str]:
        return self.memoria.recuperar(
            self.embedder.encode_one(pergunta), top_k=top_k or self.top_k
        )

    def consultar(self, pergunta: str, top_k: int | None = None) -> Resposta:
        contextos = self.recuperar(pergunta, top_k=top_k)
        prompt = _PROMPT.format(contexto="\n".join(contextos), pergunta=pergunta)
        return Resposta(
            pergunta=pergunta,
            contextos=contextos,
            resposta=self.llm.responder(prompt),
        )
