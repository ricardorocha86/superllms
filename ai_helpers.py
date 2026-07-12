"""Adaptadores pequenos para chamadas de texto em provedores compatíveis."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - depende da instalação do projeto
    Anthropic = None


def get_secret(name: str) -> str:
    """Lê secrets do Streamlit e, como fallback, variáveis de ambiente."""
    try:
        value = st_secrets_get(name)
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def st_secrets_get(name: str) -> Any:
    # Import tardio evita carregar Streamlit em scripts que só usam este módulo.
    import streamlit as st

    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return ""


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def generate_text(
    *,
    model_info: dict[str, Any],
    api_key: str,
    messages: list[dict[str, str]],
    instructions: str = "",
    temperature: float | None = None,
) -> str:
    """Gera texto com Responses, Chat Completions ou Anthropic."""
    if not api_key:
        raise RuntimeError("A chave de API deste provedor não foi configurada.")

    api_type = model_info.get("api_tipo", "chat_completions")
    model_id = model_info["modelo_id"]
    base_url = str(model_info.get("base_url") or "").strip()

    if api_type == "openai_responses":
        client = OpenAI(api_key=api_key, timeout=90.0)
        try:
            kwargs: dict[str, Any] = {
                "model": model_id,
                "input": messages,
                "store": False,
            }
            if instructions:
                kwargs["instructions"] = instructions
            response = client.responses.create(**kwargs)
            return response_text(response)
        finally:
            client.close()

    if api_type == "anthropic_messages":
        if Anthropic is None:
            raise RuntimeError("Instale o pacote anthropic para usar este modelo.")
        client = Anthropic(api_key=api_key, timeout=90.0)
        response = client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=instructions or None,
            messages=messages,
        )
        return "\n".join(
            str(block.text)
            for block in getattr(response, "content", [])
            if getattr(block, "text", None)
        ).strip()

    kwargs = {"api_key": api_key, "timeout": 90.0}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    try:
        chat_messages = list(messages)
        if instructions:
            chat_messages.insert(0, {"role": "system", "content": instructions})
        create_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": chat_messages,
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**create_kwargs)
        return (response.choices[0].message.content or "").strip()
    finally:
        client.close()
