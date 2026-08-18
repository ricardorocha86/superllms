"""Adaptadores pequenos para chamadas de texto em provedores compatíveis."""

from __future__ import annotations

import os
from datetime import datetime, timezone
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


def _para_dicionario(valor: Any) -> dict[str, Any]:
    """Normaliza os objetos de usage dos SDKs para um dict simples."""
    if valor is None:
        return {}
    if isinstance(valor, dict):
        return dict(valor)
    for metodo in ("model_dump", "to_dict", "dict"):
        conversor = getattr(valor, metodo, None)
        if callable(conversor):
            try:
                convertido = conversor()
            except Exception:
                continue
            if isinstance(convertido, dict):
                return convertido
    return {}


def _primeiro(dados: dict[str, Any], *nomes: str) -> Any:
    """Cada provedor nomeia o mesmo número de um jeito; pega o primeiro que existir."""
    for nome in nomes:
        valor = dados.get(nome)
        if valor is not None:
            return valor
    return None


def resumir_uso(response: Any, *, model_info: dict[str, Any]) -> dict[str, Any]:
    """Extrai tokens, custo e id da geração de uma resposta de qualquer provedor.

    Só o OpenRouter devolve ``cost`` em dólares; para os demais o campo fica
    ``None`` e quem exibe decide se estima pelo preço de tabela.
    """
    bruto = _para_dicionario(getattr(response, "usage", None))
    detalhes_saida = _para_dicionario(_primeiro(bruto, "completion_tokens_details", "output_tokens_details"))
    detalhes_entrada = _para_dicionario(_primeiro(bruto, "prompt_tokens_details", "input_tokens_details"))

    return {
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provedor": model_info.get("provedor", ""),
        "empresa": model_info.get("empresa", ""),
        "modelo_nome": model_info.get("modelo_nome", ""),
        "modelo_id": model_info.get("modelo_id", ""),
        "generation_id": getattr(response, "id", "") or "",
        "tokens_entrada": _primeiro(bruto, "prompt_tokens", "input_tokens"),
        "tokens_saida": _primeiro(bruto, "completion_tokens", "output_tokens"),
        "tokens_total": _primeiro(bruto, "total_tokens"),
        "tokens_raciocinio": _primeiro(detalhes_saida, "reasoning_tokens"),
        "tokens_cache": _primeiro(
            detalhes_entrada, "cached_tokens", "cache_read_input_tokens"
        ),
        # Presente apenas no OpenRouter: custo real já debitado em créditos.
        "custo_usd": _primeiro(bruto, "cost"),
    }


def generate_text_detalhado(
    *,
    model_info: dict[str, Any],
    api_key: str,
    messages: list[dict[str, str]],
    instructions: str = "",
    temperature: float | None = None,
) -> dict[str, Any]:
    """Gera texto e devolve ``{"texto": ..., "uso": {...}}``.

    Mesma lógica de ``generate_text``, mas preservando os metadados de consumo
    que a resposta já traz — é o que alimenta o Histórico de Uso.
    """
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
            return {
                "texto": response_text(response),
                "uso": resumir_uso(response, model_info=model_info),
            }
        finally:
            client.close()

    if api_type == "anthropic_messages":
        if Anthropic is None:
            raise RuntimeError("Instale o pacote anthropic para usar este modelo.")
        with Anthropic(api_key=api_key, timeout=90.0) as client:
            response = client.messages.create(
                model=model_id,
                max_tokens=2048,
                system=instructions or None,
                messages=messages,
            )
            texto = "\n".join(
                str(block.text)
                for block in getattr(response, "content", [])
                if getattr(block, "text", None)
            ).strip()
            return {
                "texto": texto,
                "uso": resumir_uso(response, model_info=model_info),
            }

    kwargs = {"api_key": api_key, "timeout": 90.0}
    if base_url:
        kwargs["base_url"] = base_url
    if "openrouter.ai" in base_url:
        # Identifica o app nos rankings do OpenRouter.
        kwargs["default_headers"] = _headers_openrouter()
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
        return {
            "texto": (response.choices[0].message.content or "").strip(),
            "uso": resumir_uso(response, model_info=model_info),
        }
    finally:
        client.close()


def _headers_openrouter() -> dict[str, str]:
    headers = {"X-OpenRouter-Title": "SuperLLMs"}
    app_url = get_secret("APP_URL")
    if app_url:
        headers["HTTP-Referer"] = app_url
    return headers


def generate_text(
    *,
    model_info: dict[str, Any],
    api_key: str,
    messages: list[dict[str, str]],
    instructions: str = "",
    temperature: float | None = None,
) -> str:
    """Gera texto com Responses, Chat Completions ou Anthropic."""
    return generate_text_detalhado(
        model_info=model_info,
        api_key=api_key,
        messages=messages,
        instructions=instructions,
        temperature=temperature,
    )["texto"]
