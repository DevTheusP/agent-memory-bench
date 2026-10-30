"""Versão 5: chave de sujeito com substituição protegida.

Igual à versão 2, com uma diferença na escrita: um fato só substitui outro
ativo, de valor diferente, se a mensagem de origem falar do mesmo assunto do
fato antigo. "Falar do mesmo assunto" é a similaridade de cosseno entre o
embedding da mensagem e o do statement antigo ficar em pelo menos SUPPORT_MIN.

Por quê: a versão 2 confia cegamente na chave. Na rodada com o extrator
destilado (fact-extractor-distill), a mensagem "Terminamos a migração: ... o
serviço de pagamentos roda em sa-east-1" gerou, além da região, um fato
inventado `coding_rules.force_push = false`. A chave já existia, o valor era
novo, e a regra "nunca faça force push na main" foi desativada em silêncio.

Calibração do limiar (bge-m3, poucas amostras, ver docs/diagnostico.md do
fact-extractor-distill): substituição inventada 0,39; substituições legítimas
da região 0,72 e 0,75. SUPPORT_MIN = 0,55 fica entre os dois grupos.

A substituição recusada não apaga nada: o episódio continua guardado e a
decisão fica registrada como UNSUPPORTED.
"""

from __future__ import annotations

from .base import Observation
from .extract import ExtractedFact
from .keyed import KeyedMemory

SUPPORT_MIN = 0.55


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # os embedders do benchmark devolvem vetores unitários


class GuardedKeyedMemory(KeyedMemory):
    name = "v5_guarded"

    def __init__(self, *args, support_min: float = SUPPORT_MIN, **kw):
        super().__init__(*args, **kw)
        self.support_min = support_min

    def _resolve(self, fact: ExtractedFact, obs: Observation, trust: float) -> None:
        current = self.store.active_by_key(fact.subject_key)
        if current and current.value.strip().lower() != fact.value.strip().lower():
            msg, old = self.embedder.embed([obs.text, current.content])
            if _cos(msg, old) < self.support_min:
                self._log(obs, fact, "UNSUPPORTED", "old", current)
                return
        super()._resolve(fact, obs, trust)
