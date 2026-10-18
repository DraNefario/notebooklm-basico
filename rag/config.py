import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    llm_provider: str
    ollama_base_url: str
    ollama_chat_model: str


def carregar_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5-coder:1.5b"),
    )
