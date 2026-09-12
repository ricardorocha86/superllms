"""Conversão da resposta de logprobs da OpenAI em dados de visualização."""

from __future__ import annotations

import math
from typing import Any


def readable_token(token: str, byte_values: list[int] | None = None) -> str:
    """Preserva espaços visualmente e recupera tokens representados por bytes."""
    if token:
        return token.replace(" ", "␠").replace("\n", "↵")
    if byte_values:
        return bytes(byte_values).decode("utf-8", errors="replace").replace(" ", "␠")
    return "∅"


def probability(logprob: float) -> float:
    """Transforma log-probabilidade natural na probabilidade correspondente."""
    return math.exp(logprob)


def parse_chat_logprobs(response: Any) -> list[dict[str, Any]]:
    """Extrai tokens e top alternatives de um ChatCompletion do SDK."""
    choice = response.choices[0]
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        raise ValueError("A resposta não trouxe logprobs de tokens.")

    return parse_token_items(content)


def parse_response_logprobs(response: Any) -> list[dict[str, Any]]:
    """Extrai logprobs dos blocos ``output_text`` da Responses API."""
    content = []
    for output in getattr(response, "output", []) or []:
        for block in getattr(output, "content", []) or []:
            content.extend(getattr(block, "logprobs", None) or [])
    if not content:
        raise ValueError("A resposta não trouxe logprobs de tokens.")
    return parse_token_items(content)


def parse_token_items(content: list[Any]) -> list[dict[str, Any]]:
    """Normaliza itens de logprob compartilhados pelas duas APIs."""
    steps = []
    for index, item in enumerate(content):
        selected = str(item.token)
        alternatives = []
        seen = set()
        candidates = [item, *(getattr(item, "top_logprobs", None) or [])]
        for candidate in candidates:
            candidate_token = str(candidate.token)
            if candidate_token in seen:
                continue
            seen.add(candidate_token)
            alternatives.append(
                {
                    "token": candidate_token,
                    "label": readable_token(candidate_token, getattr(candidate, "bytes", None)),
                    "probability": probability(float(candidate.logprob)),
                    "selected": candidate_token == selected,
                }
            )
        alternatives.sort(key=lambda row: row["probability"], reverse=True)
        steps.append(
            {
                "index": index,
                "token": selected,
                "label": readable_token(selected, getattr(item, "bytes", None)),
                "logprob": float(item.logprob),
                "probability": probability(float(item.logprob)),
                "alternatives": alternatives,
            }
        )
    return steps
