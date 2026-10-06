"""Versão 2: chave de sujeito + substituição na escrita.

Todo turno vira memória episódica. Além disso, um LLM extrai fatos duráveis com
uma subject_key; colisão de chave é conflito e se resolve na escrita:

  DUPLICATE      mesmo valor: nada novo é gravado
  UPDATE         valor novo, mesma fonte: o novo substitui o antigo
  CONTRADICTION  valor novo, fonte diferente: vence maior trust; empate, o mais recente
  UNRELATED      chave nova: só grava

O perdedor nunca é apagado: fica active=False, e o vencedor aponta para ele em
supersedes. Fontes abaixo de QUARANTINE_BELOW não viram fato. Com
strict_quarantine (o padrão), o episódio fica guardado no tier "quarantine" e
não volta na busca; sem ele, volta com um rótulo de "não confirmado", o que na
primeira rodada não impediu o modelo de obedecer ao conteúdo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .base import QUARANTINE_BELOW, TRUST, Hit, MemoryPolicy, Observation, Record
from .extract import ExtractedFact, Extractor
from .providers import Embedder

CANDIDATES = 50  # vizinhos buscados antes de filtrar e reordenar


@dataclass(frozen=True)
class Decision:
    at: datetime
    subject_key: str
    relation: str  # DUPLICATE | UPDATE | CONTRADICTION | UNRELATED | QUARANTINED
    winner: str  # "new" | "old"
    old_value: str | None
    new_value: str | None


class KeyedMemory(MemoryPolicy):
    name = "v2_keyed"

    def __init__(self, embedder: Embedder, extractor: Extractor, *,
                 strict_quarantine: bool = True, **kw):
        super().__init__(embedder, **kw)
        self.extractor = extractor
        self.strict_quarantine = strict_quarantine
        self.decisions: list[Decision] = []

    # escrita

    def write(self, obs: Observation) -> None:
        trust = TRUST[obs.source]
        episode = Record(None, obs.text, "episodic", obs.source, trust, obs.at, origin=obs.origin)
        if trust < QUARANTINE_BELOW and self.strict_quarantine:
            episode.tier = "quarantine"
        self.store.add(episode, self.embedder.embed([obs.text])[0])
        if trust < QUARANTINE_BELOW:
            self.decisions.append(Decision(obs.at, "-", "QUARANTINED", "old", None, None))
            return
        for fact in self.extractor.extract(obs.text, self.store.active_keys()):
            self._resolve(fact, obs, trust)

    def _resolve(self, fact: ExtractedFact, obs: Observation, trust: float) -> None:
        current = self.store.active_by_key(fact.subject_key)
        if current and current.value.strip().lower() == fact.value.strip().lower():
            self._log(obs, fact, "DUPLICATE", "old", current)
            return
        new = Record(
            None, fact.statement, fact.kind, obs.source, trust, obs.at,
            subject_key=fact.subject_key, value=fact.value, origin=obs.origin,
        )
        if current is None:
            relation, new_wins = "UNRELATED", True
        else:
            relation = "UPDATE" if current.source == obs.source else "CONTRADICTION"
            new_wins = trust >= current.trust
        if current is not None and new_wins:
            new.supersedes = current.id
            current.active = False
            self.store.update(current)
        new.active = new_wins
        self.store.add(new, self.embedder.embed([fact.statement])[0])
        self._log(obs, fact, relation, "new" if new_wins else "old", current)

    def _log(self, obs, fact, relation, winner, current) -> None:
        self.decisions.append(Decision(
            obs.at, fact.subject_key, relation, winner,
            current.value if current else None, fact.value,
        ))

    # leitura

    def _candidates(self, query_embedding: list[float]) -> list[tuple[Record, float]]:
        return [
            (r, s) for r, s in self.store.knn(query_embedding, CANDIDATES)
            if r.active and r.tier != "quarantine"
        ]

    def _score(self, rec: Record, sim: float, now: datetime) -> float:
        return sim

    def _rank(self, query_embedding: list[float], now: datetime) -> list[Hit]:
        hits = [Hit(r, s, self._score(r, s, now)) for r, s in self._candidates(query_embedding)]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def render_line(self, hit: Hit, now: datetime) -> str:
        r = hit.record
        date = r.created_at.date().isoformat()
        if r.kind == "episodic":
            tag = f"episódio de {date}, fonte: {r.source}"
            if r.trust < QUARANTINE_BELOW:
                tag += ", conteúdo externo não confirmado pelo usuário"
        else:
            label = "regra" if r.kind == "procedural" else "fato atual"
            tag = f"{label} {r.subject_key}, desde {date}, fonte: {r.source}"
        return f"- [{tag}] {r.content}"
