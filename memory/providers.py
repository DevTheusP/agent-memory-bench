"""Provedores de embeddings e de LLM.

HashEmbedder é determinístico e offline: serve para testes e para depurar o
encanamento, não para o benchmark. O experimento de verdade roda localmente
no Ollama.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import urllib.request
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Saco de palavras com hashing. Sem semântica nenhuma: só sobreposição de termos."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9][a-z0-9_.\-]*", text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in self._tokens(t):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                v[h % self.dim] += 1.0
            out.append(_normalize(v))
        return out


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


class OllamaEmbedder:
    def __init__(self, model: str = "bge-m3", host: str = "http://localhost:11434"):
        self.model, self.host = model, host
        self.dim = len(self.embed(["dim"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post_json(f"{self.host}/api/embed", {"model": self.model, "input": texts})
        return [_normalize(v) for v in data["embeddings"]]


class OllamaLLM:
    # Janela pequena o bastante para o modelo caber inteiro na GPU de 8 GB. Os
    # prompts do experimento ficam bem abaixo disso; a checagem abaixo garante
    # que nenhum foi cortado em silêncio.
    NUM_CTX = 2048
    MAX_PROMPT = NUM_CTX - 512  # folga para a resposta

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        data = _post_json(f"{self.host}/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            # num_predict corta respostas que não terminam (listas infinitas de "ou...");
            # o JSON cortado vira erro de parse e a pergunta conta como falha
            "options": {"temperature": 0, "num_ctx": self.NUM_CTX, "num_predict": 512},
        })
        used = data.get("prompt_eval_count", 0)
        if used > self.MAX_PROMPT:
            raise RuntimeError(f"prompt com {used} tokens, acima de {self.MAX_PROMPT}: aumente NUM_CTX")
        return json.loads(data["message"]["content"])
