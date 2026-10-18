import pathlib


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i : i + size]))
        i += max(size - overlap, 1)
    return [chunk for chunk in out if chunk.strip()]


def carregar_arquivos(pasta: pathlib.Path) -> str:
    """Lê todos os .txt da pasta e retorna o texto concatenado."""
    texto_total = ""
    for arquivo in sorted(pasta.rglob("*.txt")):
        print(f"Lendo {arquivo.name}...")
        texto_total += arquivo.read_text(encoding="utf-8") + "\n"
    return texto_total
