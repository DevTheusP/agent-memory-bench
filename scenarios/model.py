"""Tipos da simulação: mensagens da conversa e perguntas de avaliação (probes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

BRT = timezone(timedelta(hours=-3))
START = datetime(2026, 1, 5, tzinfo=BRT)  # segunda-feira da semana 1
WEEKS = 8

Source = Literal["user", "tool_output", "agent_inference", "web"]


def at(week: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Instante simulado. week e day começam em 1 (day 1 = segunda)."""
    if not 1 <= week <= WEEKS:
        raise ValueError(f"week fora do intervalo: {week}")
    if not 1 <= day <= 7:
        raise ValueError(f"day fora do intervalo: {day}")
    return START + timedelta(weeks=week - 1, days=day - 1, hours=hour, minutes=minute)


def week_of(dt: datetime) -> int:
    return (dt - START).days // 7 + 1


@dataclass(frozen=True)
class Fact:
    """Gabarito do que a mensagem afirma. Só o avaliador usa; a memória nunca vê."""

    subject_key: str
    value: str
    trusted: bool = True


@dataclass(frozen=True)
class Message:
    id: str
    at: datetime
    source: Source
    text: str
    scenario: str | None = None  # None = ruído
    fact: Fact | None = None
    reinforces: str | None = None  # subject_key repetido sem fato novo
    distractor_for: str | None = None  # ruído que parece relevante para um cenário

    @property
    def week(self) -> int:
        return week_of(self.at)


@dataclass(frozen=True)
class Probe:
    """Pergunta de avaliação. Não entra na memória: só lê dela."""

    id: str
    at: datetime
    scenario: str
    question: str
    response_schema: dict[str, str]
    grader: str
    params: dict[str, Any] = field(default_factory=dict)
    # Gabarito de recuperação, por id de mensagem de origem:
    #   need_all  todas precisam chegar ao contexto
    #   need_any  pelo menos uma precisa chegar
    #   forbid    nenhuma pode chegar
    #   stale     versões obsoletas; contar quando chegam sem a atual
    gold: dict[str, list[str]] = field(default_factory=dict)

    @property
    def week(self) -> int:
        return week_of(self.at)
