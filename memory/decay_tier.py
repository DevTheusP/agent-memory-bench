"""Versão 4: versão 2 + decaimento SÓ no rótulo e no tier hot/cold, nunca no ranking.

A varredura move para "cold" o que passou da meia-vida sem uso, sem apagar nada.
A busca ranqueia só por similaridade e consulta o cold quando o hot devolve
poucas memórias relevantes. O tempo aparece para o modelo como rótulo
("recente" / "pode estar desatualizado"), não como penalidade de score.
"""

from __future__ import annotations

from datetime import datetime

from .base import Hit
from .decay_rank import HALF_LIFE_DAYS, recency
from .extract import Extractor
from .keyed import KeyedMemory
from .providers import Embedder


class TieredMemory(KeyedMemory):
    name = "v4_decay_tier"

    def __init__(self, embedder: Embedder, extractor: Extractor, *,
                 min_sim: float = 0.35, min_hot_hits: int = 3, **kw):
        super().__init__(embedder, extractor, **kw)
        self.min_sim = min_sim
        self.min_hot_hits = min_hot_hits

    def sweep(self, now: datetime) -> None:
        for rec in self.store.all():
            age_days = (now - rec.last_touch()).total_seconds() / 86400
            if rec.active and rec.tier == "hot" and age_days > HALF_LIFE_DAYS[rec.kind]:
                rec.tier = "cold"
                self.store.update(rec)

    def _rank(self, query_embedding: list[float], now: datetime) -> list[Hit]:
        cands = self._candidates(query_embedding)
        hot = [(r, s) for r, s in cands if r.tier == "hot"]
        strong_hot = [1 for _, s in hot if s >= self.min_sim]
        pool = hot if len(strong_hot) >= self.min_hot_hits else cands
        return sorted((Hit(r, s, s) for r, s in pool), key=lambda h: h.score, reverse=True)

    def mark_used(self, ids: list[int], now: datetime) -> None:
        super().mark_used(ids, now)
        for rec in self.store.get_many(ids):
            if rec.tier == "cold":
                rec.tier = "hot"
                self.store.update(rec)

    def render_line(self, hit: Hit, now: datetime) -> str:
        fresh = "recente" if recency(hit.record, now) >= 0.5 else "pode estar desatualizado"
        line = super().render_line(hit, now)
        return line.replace("] ", f", {fresh}] ", 1)
