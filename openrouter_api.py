"""Cliente fino para os endpoints do OpenRouter usados pelo app.

Cobre três coisas que antes eram feitas na mão:

- ``listar_modelos``  -> catálogo ao vivo (/models), com cache em disco.
- ``obter_creditos``  -> saldo da conta (/credits).
- ``obter_geracao``   -> tokens e custo real de uma resposta (/generation).

O módulo não importa Streamlit no topo de propósito: ``modelos.py`` depende
dele em tempo de import e precisa continuar utilizável fora do app.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://openrouter.ai/api/v1"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_MODELOS = CACHE_DIR / "openrouter_models.json"
CACHE_TTL_SEGUNDOS = 6 * 60 * 60
TIMEOUT_SEGUNDOS = 10.0


def get_secret(nome: str) -> str:
    """Lê secrets do Streamlit e, como fallback, variáveis de ambiente."""
    # Reaproveita o helper do app; o import é tardio porque ai_helpers puxa
    # os SDKs da OpenAI e da Anthropic.
    from ai_helpers import get_secret as _get_secret

    return _get_secret(nome)


def cabecalhos(*, autenticado: bool = True) -> dict[str, str]:
    """Headers padrão, incluindo a atribuição do app nos rankings."""
    headers = {"X-OpenRouter-Title": "SuperLLMs"}

    # HTTP-Referer só faz sentido apontando para o deploy real; sem ele o
    # OpenRouter aceita o título mas não cria a página do app.
    app_url = get_secret("APP_URL")
    if app_url:
        headers["HTTP-Referer"] = app_url

    if autenticado:
        chave = get_secret("OPENROUTER_API_KEY")
        if chave:
            headers["Authorization"] = f"Bearer {chave}"

    return headers


def preco_por_milhao(valor: Any) -> str:
    """Converte o preço por token da API para o formato ``$0.00`` do catálogo."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return "N/D"

    if numero < 0:  # o Auto Router usa -1 como sentinela de "preço variável".
        return "N/D"

    por_milhao = numero * 1_000_000
    if por_milhao == 0:
        return "$0.00"
    if por_milhao < 0.01:
        return f"${por_milhao:.4f}"
    return f"${por_milhao:.2f}"


# Grafias que o .title() erraria quando o nome do modelo não traz o autor.
AUTORES_POR_SLUG = {
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "nvidia": "NVIDIA",
    "z-ai": "Z.ai",
    "x-ai": "X.AI",
    "meta-llama": "Meta",
    "deepseek": "DeepSeek",
    "liquid": "LiquidAI",
    "moonshotai": "MoonshotAI",
}


def _data_utc(timestamp: Any) -> str:
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _ler_cache() -> tuple[list[dict], float] | None:
    if not CACHE_MODELOS.exists():
        return None
    try:
        conteudo = json.loads(CACHE_MODELOS.read_text(encoding="utf-8"))
        return conteudo["data"], float(conteudo["buscado_em"])
    except (OSError, ValueError, KeyError):
        return None


def _gravar_cache(dados: list[dict]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_MODELOS.write_text(
            json.dumps({"buscado_em": time.time(), "data": dados}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Cache é otimização, não requisito: disco cheio ou somente-leitura
        # não pode derrubar o app.
        pass


def listar_modelos(*, forcar_atualizacao: bool = False) -> dict[str, Any]:
    """Retorna o catálogo do OpenRouter e a origem dos dados.

    O resultado sempre tem a forma ``{"modelos": [...], "origem": ..., "buscado_em": ...}``.
    ``origem`` é ``"api"``, ``"cache"`` ou ``"indisponivel"`` — quem chama decide
    o que fazer quando a rede falha.
    """
    cache = _ler_cache()
    if not forcar_atualizacao and cache:
        modelos, buscado_em = cache
        if time.time() - buscado_em < CACHE_TTL_SEGUNDOS:
            return {"modelos": modelos, "origem": "cache", "buscado_em": buscado_em, "erro": ""}

    try:
        resposta = requests.get(
            f"{BASE_URL}/models",
            headers=cabecalhos(autenticado=False),
            timeout=TIMEOUT_SEGUNDOS,
        )
        resposta.raise_for_status()
        modelos = resposta.json().get("data", [])
        if not modelos:
            raise ValueError("resposta sem a chave 'data'")
    except (requests.RequestException, ValueError) as exc:
        if cache:  # cache vencido ainda é melhor que nada.
            modelos, buscado_em = cache
            return {
                "modelos": modelos,
                "origem": "cache",
                "buscado_em": buscado_em,
                "erro": f"{exc.__class__.__name__}: {exc}",
            }
        return {
            "modelos": [],
            "origem": "indisponivel",
            "buscado_em": 0.0,
            "erro": f"{exc.__class__.__name__}: {exc}",
        }

    _gravar_cache(modelos)
    return {"modelos": modelos, "origem": "api", "buscado_em": time.time(), "erro": ""}


def _preco_zero(pricing: dict) -> bool:
    for campo in ("prompt", "completion"):
        try:
            if float(pricing.get(campo)) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def modelos_gratuitos(modelos: list[dict]) -> list[dict]:
    """Filtra tudo que não consome créditos.

    O critério é o preço, não o sufixo ``:free`` — assim entra também o
    ``openrouter/free`` (roteador que sorteia entre os gratuitos disponíveis),
    que não usa o sufixo.

    A saída precisa ser exclusivamente texto: os modelos de música do Google
    (Lyria) anunciam token a $0 mas cobram por música gerada, então entrariam
    aqui indevidamente.
    """
    gratuitos = []
    for modelo in modelos:
        if not _preco_zero(modelo.get("pricing") or {}):
            continue
        saidas = (modelo.get("architecture") or {}).get("output_modalities") or ["text"]
        if list(saidas) != ["text"]:
            continue
        gratuitos.append(modelo)
    return gratuitos


def normalizar(modelo: dict) -> dict[str, Any]:
    """Extrai da resposta da API só os campos que o catálogo do app usa."""
    identificador = str(modelo.get("id", ""))
    nome_completo = str(modelo.get("name", identificador))

    # "Z.ai: GLM 5.2 (free)" -> autor "Z.ai", nome "GLM 5.2".
    autor, _, nome = nome_completo.partition(": ")
    if not nome:
        # Sem o prefixo "Autor: ", só resta o slug — que vem todo minúsculo.
        slug = identificador.split("/")[0]
        autor, nome = AUTORES_POR_SLUG.get(slug, slug.title()), nome_completo
    nome = nome.replace(" (free)", "").strip()

    pricing = modelo.get("pricing") or {}
    benchmarks = (modelo.get("benchmarks") or {}).get("artificial_analysis") or {}
    reasoning = modelo.get("reasoning") or {}

    return {
        "id": identificador,
        "autor": autor.strip(),
        "nome": nome or identificador,
        "contexto_tokens": modelo.get("context_length"),
        "custo_input_1M": preco_por_milhao(pricing.get("prompt")),
        "custo_output_1M": preco_por_milhao(pricing.get("completion")),
        "criado_em": _data_utc(modelo.get("created")),
        "expira_em": modelo.get("expiration_date") or "",
        "modalidades_entrada": (modelo.get("architecture") or {}).get("input_modalities") or [],
        "parametros": modelo.get("supported_parameters") or [],
        "indice_inteligencia": benchmarks.get("intelligence_index"),
        "indice_codigo": benchmarks.get("coding_index"),
        "indice_agentico": benchmarks.get("agentic_index"),
        "esforcos_reasoning": reasoning.get("supported_efforts") or [],
    }


def obter_creditos() -> dict[str, Any]:
    """Saldo da conta. Retorna ``{}`` se não houver chave ou a chamada falhar."""
    if not get_secret("OPENROUTER_API_KEY"):
        return {}
    try:
        resposta = requests.get(
            f"{BASE_URL}/credits", headers=cabecalhos(), timeout=TIMEOUT_SEGUNDOS
        )
        resposta.raise_for_status()
        return resposta.json().get("data", {})
    except (requests.RequestException, ValueError):
        return {}


def obter_geracao(generation_id: str) -> dict[str, Any]:
    """Metadados de custo/tokens de uma geração já concluída.

    O OpenRouter leva alguns instantes para consolidar a geração, então vale
    tentar de novo se vier vazio logo após a resposta.
    """
    if not generation_id or not get_secret("OPENROUTER_API_KEY"):
        return {}
    try:
        resposta = requests.get(
            f"{BASE_URL}/generation",
            headers=cabecalhos(),
            params={"id": generation_id},
            timeout=TIMEOUT_SEGUNDOS,
        )
        resposta.raise_for_status()
        return resposta.json().get("data", {})
    except (requests.RequestException, ValueError):
        return {}
