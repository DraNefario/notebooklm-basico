from typing import Protocol


class ChatClient(Protocol):
    def responder(self, prompt: str) -> str: ...


class OllamaClient:
    def __init__(self, host: str, model: str):
        from ollama import Client
        self._client = Client(host=host)
        self._model = model

    def responder(self, prompt: str) -> str:
        resposta = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.message.content


class MockClient:
    def responder(self, prompt: str) -> str:
        return f"[mock] Recebi o prompt com {len(prompt)} caracteres. Sem Ollama configurado."


def criar_llm(settings) -> ChatClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(host=settings.ollama_base_url, model=settings.ollama_chat_model)
    if settings.llm_provider == "mock":
        return MockClient()
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.llm_provider!r} (use 'ollama' ou 'mock')")
