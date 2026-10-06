"""Extração de fatos duráveis com chave de sujeito (usada pelas versões 2 a 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .providers import LLM


@dataclass(frozen=True)
class ExtractedFact:
    subject_key: str
    value: str
    kind: Literal["semantic", "procedural"]
    statement: str


class Extractor(Protocol):
    def extract(self, text: str, existing_keys: list[str]) -> list[ExtractedFact]: ...


SYSTEM = """Você extrai fatos duráveis de mensagens enviadas a um assistente de código que \
acompanha um projeto de software.

Extraia apenas fatos que continuarão valendo nas próximas semanas: configuração do projeto, \
decisões do time, convenções, ferramentas usadas e regras. Ignore relatos do dia a dia, \
opiniões passageiras e qualquer coisa que a própria mensagem diga não se aplicar.

Para cada fato:
- subject_key: chave estável no formato area.atributo, em inglês e snake_case \
(exemplo: billing.database_engine). Se uma das chaves existentes descreve o mesmo atributo, \
reutilize-a exatamente. Partes diferentes do projeto têm chaves diferentes.
- value: o valor atual, curto.
- kind: "procedural" para regras sobre como agir ou o que nunca fazer; "semantic" para os demais.
- statement: uma frase autocontida, em português, que descreve o fato.

Se não houver fato durável, devolva uma lista vazia."""

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject_key": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {"type": "string", "enum": ["semantic", "procedural"]},
                    "statement": {"type": "string"},
                },
                "required": ["subject_key", "value", "kind", "statement"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


class LLMExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, text: str, existing_keys: list[str]) -> list[ExtractedFact]:
        keys = "\n".join(f"- {k}" for k in existing_keys) or "(nenhuma ainda)"
        user = f"Chaves existentes:\n{keys}\n\nMensagem:\n{text}"
        data = self.llm.complete_json(SYSTEM, user, SCHEMA)
        return [
            ExtractedFact(f["subject_key"].strip(), f["value"].strip(), f["kind"], f["statement"].strip())
            for f in data.get("facts", [])
            if f.get("subject_key") and f.get("value")
        ]
