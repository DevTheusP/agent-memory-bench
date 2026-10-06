"""Linha de base sem memória: mede o que o modelo responde só com o próprio conhecimento.

Serve para separar o efeito da memória do comportamento prévio do modelo (por
exemplo, sugerir verify=False por conta própria, sem injeção nenhuma).
"""

from __future__ import annotations

from datetime import datetime

from .base import Hit, MemoryPolicy, Observation


class NoMemory(MemoryPolicy):
    name = "v0_no_memory"

    def write(self, obs: Observation) -> None:
        pass

    def _rank(self, query_embedding: list[float], now: datetime) -> list[Hit]:
        return []
