"""Versão 1, ingênua: todo turno vira embedding; busca top-k por similaridade de cosseno."""

from __future__ import annotations

from datetime import datetime

from .base import TRUST, Hit, MemoryPolicy, Observation, Record


class NaiveMemory(MemoryPolicy):
    name = "v1_naive"

    def write(self, obs: Observation) -> None:
        rec = Record(None, obs.text, "episodic", obs.source, TRUST[obs.source], obs.at, origin=obs.origin)
        self.store.add(rec, self.embedder.embed([obs.text])[0])

    def _rank(self, query_embedding: list[float], now: datetime) -> list[Hit]:
        return [Hit(r, s, s) for r, s in self.store.knn(query_embedding, self.top_k)]
