"""Versão 3: versão 2 + decaimento no RANKING.

score = 0.55*similaridade + 0.20*recência + 0.20*trust + 0.05*uso

recência é exponencial com meia-vida por tipo (30 dias para episódios, 180 para
fatos e regras), contada do último uso ou, sem uso, da criação. É a variante que,
segundo um relato nos comentários do dev.to, mata fatos raros e críticos.
"""

from __future__ import annotations

from datetime import datetime

from .base import Record
from .keyed import KeyedMemory

WEIGHTS = {"sim": 0.55, "recency": 0.20, "trust": 0.20, "use": 0.05}
HALF_LIFE_DAYS = {"episodic": 30.0, "semantic": 180.0, "procedural": 180.0}


def recency(rec: Record, now: datetime) -> float:
    age_days = max(0.0, (now - rec.last_touch()).total_seconds() / 86400)
    return 0.5 ** (age_days / HALF_LIFE_DAYS[rec.kind])


def usage(rec: Record) -> float:
    return 1 - 0.5 ** rec.use_count  # 0, 0.5, 0.75, ... satura em 1


class DecayRankMemory(KeyedMemory):
    name = "v3_decay_rank"

    def _score(self, rec: Record, sim: float, now: datetime) -> float:
        return (
            WEIGHTS["sim"] * sim
            + WEIGHTS["recency"] * recency(rec, now)
            + WEIGHTS["trust"] * rec.trust
            + WEIGHTS["use"] * usage(rec)
        )
