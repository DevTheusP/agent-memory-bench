"""Monta a linha do tempo da simulação: cenários + ruído + perguntas de avaliação.

Uso:
    uv run python -m scenarios.build                       # padrão: seed 42, 3 ruídos/semana
    uv run python -m scenarios.build --noise-per-week 20   # para testar acúmulo
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from .library import NOISE, core_messages, probes
from .model import START, WEEKS, Message, Probe, at, week_of

DEFAULT_OUT = Path(__file__).parent / "timeline.json"


@dataclass(frozen=True)
class Timeline:
    seed: int
    noise_per_week: int
    messages: list[Message]
    probes: list[Probe]
    # Pausa entre a semana 1 e o resto da conversa, no relógio da memória. Os fatos
    # da semana 1 (a regra rara entre eles) ficam com essa idade a mais, enquanto o
    # resto segue no ritmo normal: fato antigo, atividade recente densa.
    gap_days: float = 0.0

    def clock(self, dt: datetime) -> datetime:
        return dt if week_of(dt) == 1 else dt + timedelta(days=self.gap_days)

    def events(self) -> Iterator[Message | Probe]:
        """Mensagens e perguntas em ordem cronológica. No mesmo instante, mensagem vem antes."""
        tagged = [(m.at, 0, m) for m in self.messages] + [(p.at, 1, p) for p in self.probes]
        for _, _, ev in sorted(tagged, key=lambda t: (t[0], t[1])):
            yield ev

    def to_dict(self) -> dict:
        def enc(obj) -> dict:
            d = asdict(obj)
            d["at"] = obj.at.isoformat()
            d["week"] = obj.week
            return d

        return {
            "meta": {
                "seed": self.seed,
                "noise_per_week": self.noise_per_week,
                "gap_days": self.gap_days,
                "weeks": WEEKS,
                "start": START.isoformat(),
            },
            "messages": [enc(m) for m in self.messages],
            "probes": [enc(p) for p in self.probes],
        }


def _noise(rng: random.Random, per_week: int) -> list[Message]:
    """Ruído reprodutível. Esgota o banco antes de repetir uma frase."""
    out: list[Message] = []
    pool: list[tuple[str, str | None]] = []
    used_slots: set[datetime] = set()
    for week in range(1, WEEKS + 1):
        for i in range(per_week):
            if not pool:
                pool = NOISE.copy()
                rng.shuffle(pool)
            text, distractor = pool.pop()
            # dias 1 a 6: o domingo fica livre para as perguntas de avaliação
            while True:
                when = at(week, rng.randint(1, 6), rng.randint(8, 22), rng.choice((0, 15, 30, 45)))
                if when not in used_slots:
                    used_slots.add(when)
                    break
            out.append(Message(f"noise-w{week}-{i}", when, "user", text, distractor_for=distractor))
    return out


def build_timeline(seed: int = 42, noise_per_week: int = 3, gap_days: float = 0.0) -> Timeline:
    rng = random.Random(seed)
    messages = sorted(core_messages() + _noise(rng, noise_per_week), key=lambda m: m.at)
    return Timeline(seed, noise_per_week, messages, sorted(probes(), key=lambda p: p.at), gap_days)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-per-week", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    tl = build_timeline(args.seed, args.noise_per_week)
    args.out.write_text(json.dumps(tl.to_dict(), ensure_ascii=False, indent=2) + "\n")
    core = sum(m.scenario is not None for m in tl.messages)
    print(
        f"{args.out}: {len(tl.messages)} mensagens ({core} de cenário, "
        f"{len(tl.messages) - core} de ruído), {len(tl.probes)} perguntas, {WEEKS} semanas"
    )


if __name__ == "__main__":
    main()
