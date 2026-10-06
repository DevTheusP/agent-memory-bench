"""Modelo de dados e interface comum das 4 políticas de memória."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .providers import Embedder

Kind = Literal["episodic", "semantic", "procedural"]
Tier = Literal["hot", "cold", "quarantine"]

TRUST = {"user": 1.0, "tool_output": 0.7, "agent_inference": 0.5, "web": 0.3}
QUARANTINE_BELOW = 0.5  # abaixo disso, a fonte não vira fato sem confirmação do usuário


@dataclass(frozen=True)
class Observation:
    """O que a memória recebe: texto, fonte e instante. Nunca o gabarito do cenário.

    origin é um id opaco da mensagem; a memória só o carrega adiante, para o
    avaliador saber de onde veio cada memória recuperada.
    """

    text: str
    source: str
    at: datetime
    origin: str | None = None


@dataclass
class Record:
    id: int | None
    content: str
    kind: Kind
    source: str
    trust: float
    created_at: datetime
    last_used_at: datetime | None = None
    use_count: int = 0
    subject_key: str | None = None
    value: str | None = None
    supersedes: int | None = None
    active: bool = True
    tier: Tier = "hot"
    origin: str | None = None  # id da mensagem que gerou a memória (só para avaliação)

    def last_touch(self) -> datetime:
        return self.last_used_at or self.created_at


@dataclass(frozen=True)
class Hit:
    record: Record
    similarity: float
    score: float


def estimate_tokens(text: str) -> int:
    """Aproximação de ~4 caracteres por token; serve para comparar versões entre si."""
    return max(1, len(text) // 4)


class MemoryPolicy(ABC):
    """Interface comum. O runner chama write a cada mensagem, sweep e retrieve a cada pergunta."""

    name: str

    def __init__(self, embedder: Embedder, *, budget_tokens: int = 1500, top_k: int = 8):
        from .store import VectorStore

        self.embedder = embedder
        self.store = VectorStore(embedder.dim)
        self.budget_tokens = budget_tokens
        self.top_k = top_k

    @abstractmethod
    def write(self, obs: Observation) -> None: ...

    @abstractmethod
    def _rank(self, query_embedding: list[float], now: datetime) -> list[Hit]:
        """Candidatos em ordem de preferência, antes do corte por orçamento."""

    def render_line(self, hit: Hit, now: datetime) -> str:
        return f"- {hit.record.content}"

    def retrieve(self, query: str, now: datetime) -> list[Hit]:
        """Melhores memórias que cabem em top_k e no orçamento de tokens."""
        ranked = self._rank(self.embedder.embed([query])[0], now)
        out: list[Hit] = []
        used = 0
        for hit in ranked:
            cost = estimate_tokens(self.render_line(hit, now))
            if len(out) == self.top_k or used + cost > self.budget_tokens:
                break
            out.append(hit)
            used += cost
        return out

    def render(self, hits: list[Hit], now: datetime) -> str:
        return "\n".join(self.render_line(h, now) for h in hits)

    def mark_used(self, ids: list[int], now: datetime) -> None:
        """Reforço no uso (a resposta de fato se apoiou na memória), não na recuperação."""
        for rec in self.store.get_many(ids):
            rec.use_count += 1
            rec.last_used_at = now
            self.store.update(rec)

    def sweep(self, now: datetime) -> None:
        """Manutenção periódica. Só a versão 4 faz algo aqui."""
