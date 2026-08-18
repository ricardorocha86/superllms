# Base atualizada dos modelos de IA usados pelo app.
#
# Fontes consultadas em 2026-07-11:
# - OpenAI API docs: https://platform.openai.com/docs/models
# - Google Gemini API docs: https://ai.google.dev/gemini-api/docs/models
# - Anthropic Claude Platform docs: https://docs.anthropic.com/en/docs/about-claude/models/overview
# - Groq supported models: https://console.groq.com/docs/models
# - OpenRouter public models API: https://openrouter.ai/api/v1/models
# - X.AI models API: https://api.x.ai/v1/models
# - DeepSeek models API: https://api.deepseek.com/models

from datetime import datetime, timezone

import pandas as pd


DATA_ATUALIZACAO = "2026-07-11"

OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def _modelo(
    *,
    empresa,
    modelo_id,
    modelo_nome,
    provedor,
    base_url,
    api_tipo,
    api_key_secret,
    custo_input_1M,
    custo_output_1M,
    tier,
    creditos,
    contexto_tokens,
    status,
    fonte,
    logo,
    cor,
    observacao="",
    selecionar_padrao=False,
):
    return {
        "uid": f"{provedor}::{modelo_id}",
        "empresa": empresa,
        "modelo_id": modelo_id,
        "modelo_nome": modelo_nome,
        "provedor": provedor,
        "base_url": base_url,
        "api_tipo": api_tipo,
        "api_key_secret": api_key_secret,
        "custo_input_1M": custo_input_1M,
        "custo_output_1M": custo_output_1M,
        "tier": tier,
        "creditos": creditos,
        "contexto_tokens": contexto_tokens,
        "status": status,
        "fonte": fonte,
        "logo": logo,
        "cor": cor,
        "observacao": observacao,
        "selecionar_padrao": selecionar_padrao,
    }


modelos_db = [
    # OpenAI - família GPT-5.6.
    _modelo(
        empresa="OpenAI",
        modelo_id="gpt-5.6-sol",
        modelo_nome="GPT-5.6 Sol",
        provedor="OpenAI",
        base_url=OPENAI_BASE_URL,
        api_tipo="openai_responses",
        api_key_secret="OPENAI_API_KEY",
        custo_input_1M="$5.00",
        custo_output_1M="$30.00",
        tier="Elite",
        creditos=10,
        contexto_tokens=1_050_000,
        status="Atual",
        fonte="OpenAI API docs",
        logo="openai.jpg",
        cor="#1f77b4",
    ),
    _modelo(
        empresa="OpenAI",
        modelo_id="gpt-5.6-terra",
        modelo_nome="GPT-5.6 Terra",
        provedor="OpenAI",
        base_url=OPENAI_BASE_URL,
        api_tipo="openai_responses",
        api_key_secret="OPENAI_API_KEY",
        custo_input_1M="$2.50",
        custo_output_1M="$15.00",
        tier="Pro",
        creditos=5,
        contexto_tokens=1_050_000,
        status="Atual",
        fonte="OpenAI API docs",
        logo="openai.jpg",
        cor="#1f77b4",
    ),
    _modelo(
        empresa="OpenAI",
        modelo_id="gpt-5.6-luna",
        modelo_nome="GPT-5.6 Luna",
        provedor="OpenAI",
        base_url=OPENAI_BASE_URL,
        api_tipo="openai_responses",
        api_key_secret="OPENAI_API_KEY",
        custo_input_1M="$1.00",
        custo_output_1M="$6.00",
        tier="Básico",
        creditos=2,
        contexto_tokens=1_050_000,
        status="Atual",
        fonte="OpenAI API docs",
        logo="openai.jpg",
        cor="#1f77b4",
        selecionar_padrao=True,
    ),
    # Google Gemini - via endpoint OpenAI-compatible oficial.
    _modelo(
        empresa="Google",
        modelo_id="gemini-3.5-flash",
        modelo_nome="Gemini 3.5 Flash",
        provedor="Google Gemini",
        base_url=GEMINI_OPENAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GEMINI_API_KEY",
        custo_input_1M="$1.50",
        custo_output_1M="$9.00",
        tier="Pro",
        creditos=2,
        contexto_tokens=None,
        status="Stable",
        fonte="Google Gemini API docs",
        logo="google.JPG",
        cor="#ff7f0e",
    ),
    _modelo(
        empresa="Google",
        modelo_id="gemini-3.1-flash-lite",
        modelo_nome="Gemini 3.1 Flash-Lite",
        provedor="Google Gemini",
        base_url=GEMINI_OPENAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GEMINI_API_KEY",
        custo_input_1M="$0.25",
        custo_output_1M="$1.50",
        tier="Básico",
        creditos=1,
        contexto_tokens=None,
        status="Stable",
        fonte="Google Gemini API docs",
        logo="google.JPG",
        cor="#ff7f0e",
        selecionar_padrao=True,
    ),
    _modelo(
        empresa="Google",
        modelo_id="gemini-3.1-pro-preview",
        modelo_nome="Gemini 3.1 Pro Preview",
        provedor="Google Gemini",
        base_url=GEMINI_OPENAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GEMINI_API_KEY",
        custo_input_1M="$2.00",
        custo_output_1M="$12.00",
        tier="Preview",
        creditos=4,
        contexto_tokens=None,
        status="Preview",
        fonte="Google Gemini API docs",
        logo="google.JPG",
        cor="#ff7f0e",
    ),

    # Anthropic - Claude API nativa.
    _modelo(
        empresa="Anthropic",
        modelo_id="claude-fable-5",
        modelo_nome="Claude Fable 5",
        provedor="Anthropic",
        base_url="",
        api_tipo="anthropic_messages",
        api_key_secret="ANTHROPIC_API_KEY",
        custo_input_1M="$10.00",
        custo_output_1M="$50.00",
        tier="Elite",
        creditos=10,
        contexto_tokens=1_000_000,
        status="Atual",
        fonte="Anthropic Claude Platform docs",
        logo="anthropic.JPG",
        cor="#d62728",
    ),
    _modelo(
        empresa="Anthropic",
        modelo_id="claude-sonnet-5",
        modelo_nome="Claude Sonnet 5",
        provedor="Anthropic",
        base_url="",
        api_tipo="anthropic_messages",
        api_key_secret="ANTHROPIC_API_KEY",
        custo_input_1M="$2.00",
        custo_output_1M="$10.00",
        tier="Pro",
        creditos=2,
        contexto_tokens=1_000_000,
        status="Atual",
        fonte="Anthropic Claude Platform docs",
        logo="anthropic.JPG",
        cor="#d62728",
        observacao="Preço introdutório até 2026-08-31.",
    ),
    _modelo(
        empresa="Anthropic",
        modelo_id="claude-haiku-4-5-20251001",
        modelo_nome="Claude Haiku 4.5",
        provedor="Anthropic",
        base_url="",
        api_tipo="anthropic_messages",
        api_key_secret="ANTHROPIC_API_KEY",
        custo_input_1M="$1.00",
        custo_output_1M="$5.00",
        tier="Pro",
        creditos=2,
        contexto_tokens=200_000,
        status="Atual",
        fonte="Anthropic Claude Platform docs",
        logo="anthropic.JPG",
        cor="#d62728",
    ),

    # Groq - modelos de texto atuais da tabela oficial.
    _modelo(
        empresa="Meta",
        modelo_id="llama-3.1-8b-instant",
        modelo_nome="Llama 3.1 8B Instant",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.05",
        custo_output_1M="$0.08",
        tier="Básico",
        creditos=1,
        contexto_tokens=131_072,
        status="Produção",
        fonte="Groq docs",
        logo="meta.JPG",
        cor="#9b59b6",
    ),
    _modelo(
        empresa="Meta",
        modelo_id="llama-3.3-70b-versatile",
        modelo_nome="Llama 3.3 70B Versatile",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.59",
        custo_output_1M="$0.79",
        tier="Pro",
        creditos=2,
        contexto_tokens=131_072,
        status="Produção",
        fonte="Groq docs",
        logo="meta.JPG",
        cor="#9b59b6",
    ),
    _modelo(
        empresa="OpenAI",
        modelo_id="openai/gpt-oss-120b",
        modelo_nome="GPT-OSS 120B",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.15",
        custo_output_1M="$0.60",
        tier="Básico",
        creditos=1,
        contexto_tokens=131_072,
        status="Produção",
        fonte="Groq docs",
        logo="openai.jpg",
        cor="#16a085",
        selecionar_padrao=True,
    ),
    _modelo(
        empresa="OpenAI",
        modelo_id="openai/gpt-oss-20b",
        modelo_nome="GPT-OSS 20B",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.075",
        custo_output_1M="$0.30",
        tier="Básico",
        creditos=1,
        contexto_tokens=131_072,
        status="Produção",
        fonte="Groq docs",
        logo="openai.jpg",
        cor="#16a085",
    ),
    _modelo(
        empresa="Meta",
        modelo_id="meta-llama/llama-4-scout-17b-16e-instruct",
        modelo_nome="Llama 4 Scout 17B 16E",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.11",
        custo_output_1M="$0.34",
        tier="Preview",
        creditos=1,
        contexto_tokens=131_072,
        status="Preview",
        fonte="Groq docs",
        logo="meta.JPG",
        cor="#9b59b6",
    ),
    _modelo(
        empresa="Qwen",
        modelo_id="qwen/qwen3-32b",
        modelo_nome="Qwen3 32B",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.29",
        custo_output_1M="$0.59",
        tier="Preview",
        creditos=1,
        contexto_tokens=131_072,
        status="Preview",
        fonte="Groq docs",
        logo="qwen.JPG",
        cor="#27ae60",
    ),
    _modelo(
        empresa="Qwen",
        modelo_id="qwen/qwen3.6-27b",
        modelo_nome="Qwen3.6 27B",
        provedor="Groq",
        base_url=GROQ_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="GROQ_API_KEY",
        custo_input_1M="$0.60",
        custo_output_1M="$3.00",
        tier="Preview",
        creditos=2,
        contexto_tokens=131_072,
        status="Preview",
        fonte="Groq docs",
        logo="qwen.JPG",
        cor="#27ae60",
    ),

    # X.AI - modelos atuais retornados pelo endpoint /v1/models.
    _modelo(
        empresa="X.AI",
        modelo_id="grok-4.3",
        modelo_nome="Grok 4.3",
        provedor="X.AI",
        base_url=XAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="XAI_API_KEY",
        custo_input_1M="$12.50",
        custo_output_1M="$25.00",
        tier="Elite",
        creditos=10,
        contexto_tokens=1_000_000,
        status="Atual",
        fonte="X.AI models API",
        logo="xai.JPG",
        cor="#2ca02c",
    ),
    _modelo(
        empresa="X.AI",
        modelo_id="grok-4.20-0309-reasoning",
        modelo_nome="Grok 4.20 Reasoning",
        provedor="X.AI",
        base_url=XAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="XAI_API_KEY",
        custo_input_1M="$12.50",
        custo_output_1M="$25.00",
        tier="Elite",
        creditos=10,
        contexto_tokens=1_000_000,
        status="Atual",
        fonte="X.AI models API",
        logo="xai.JPG",
        cor="#2ca02c",
    ),
    _modelo(
        empresa="X.AI",
        modelo_id="grok-4.20-0309-non-reasoning",
        modelo_nome="Grok 4.20 Non-Reasoning",
        provedor="X.AI",
        base_url=XAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="XAI_API_KEY",
        custo_input_1M="$12.50",
        custo_output_1M="$25.00",
        tier="Elite",
        creditos=10,
        contexto_tokens=1_000_000,
        status="Atual",
        fonte="X.AI models API",
        logo="xai.JPG",
        cor="#2ca02c",
    ),
    _modelo(
        empresa="X.AI",
        modelo_id="grok-build-0.1",
        modelo_nome="Grok Build 0.1",
        provedor="X.AI",
        base_url=XAI_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="XAI_API_KEY",
        custo_input_1M="$10.00",
        custo_output_1M="$20.00",
        tier="Pro",
        creditos=4,
        contexto_tokens=256_000,
        status="Atual",
        fonte="X.AI models API",
        logo="xai.JPG",
        cor="#2ca02c",
    ),

    # DeepSeek - modelos atuais retornados pelo endpoint /models.
    _modelo(
        empresa="DeepSeek",
        modelo_id="deepseek-v4-flash",
        modelo_nome="DeepSeek V4 Flash",
        provedor="DeepSeek",
        base_url=DEEPSEEK_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="DEEPSEEK_API_KEY",
        custo_input_1M="N/D",
        custo_output_1M="N/D",
        tier="Básico",
        creditos=1,
        contexto_tokens=None,
        status="Atual",
        fonte="DeepSeek models API",
        logo="deepseek.JPG",
        cor="#9467bd",
    ),
    _modelo(
        empresa="DeepSeek",
        modelo_id="deepseek-v4-pro",
        modelo_nome="DeepSeek V4 Pro",
        provedor="DeepSeek",
        base_url=DEEPSEEK_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="DEEPSEEK_API_KEY",
        custo_input_1M="N/D",
        custo_output_1M="N/D",
        tier="Pro",
        creditos=2,
        contexto_tokens=None,
        status="Atual",
        fonte="DeepSeek models API",
        logo="deepseek.JPG",
        cor="#9467bd",
    ),
]


# ---------------------------------------------------------------------------
# OpenRouter Free: carregado ao vivo de /api/v1/models.
#
# A lista costumava ser fixa aqui, e envelhecia mal: modelos ":free" entram e
# saem do catálogo toda semana, então o app acabava oferecendo model_ids que já
# retornavam 404. Agora o catálogo vem da API (com cache em disco de 6h) e o
# snapshot abaixo é só a rede de segurança para quando não houver rede nem cache.
# ---------------------------------------------------------------------------

# Ordem de preferência para o que já vem marcado no Laboratório. Só entram os
# que estiverem realmente vivos no momento do carregamento.
OPENROUTER_PREFERIDOS = (
    # O roteador vem primeiro: ele sorteia entre os gratuitos que estão de pé,
    # então é o que menos esbarra no rate limit compartilhado do free tier.
    "openrouter/free",
    "z-ai/glm-5.2:free",
    "poolside/laguna-s-2.1:free",
    "openai/gpt-oss-20b:free",
)
OPENROUTER_MAX_PADRAO = 4

# Snapshot verificado em 2026-08-18 contra /api/v1/models.
# (empresa, modelo_id, nome, contexto, expira_em, criado_em)
OPENROUTER_FREE_SNAPSHOT = [
    ("OpenRouter", "openrouter/free", "Free Models Router", 200_000, "", ""),
    ("Dots Studio", "dots-studio/dots-3-note-preview:free", "Dots3-Note Preview", 512_000, "", "2026-08-14"),
    ("LiquidAI", "liquid/lfm-2.5-2.6b:free", "LFM2.5-2.6B", 128_000, "", "2026-08-11"),
    ("NVIDIA", "nvidia/nemotron-3.5-lightning:free", "Nemotron 3.5 Lightning", 1_000_000, "", "2026-08-11"),
    ("Poolside", "poolside/laguna-s-2.1:free", "Laguna S 2.1", 262_144, "", "2026-07-21"),
    ("Poolside", "poolside/laguna-xs-2.1:free", "Laguna XS 2.1", 262_144, "", "2026-06-25"),
    ("Cohere", "cohere/north-mini-code:free", "North Mini Code", 256_000, "", "2026-06-17"),
    ("Z.ai", "z-ai/glm-5.2:free", "GLM 5.2", 256_000, "", "2026-06-16"),
    ("NVIDIA", "nvidia/nemotron-3.5-content-safety:free", "Nemotron 3.5 Content Safety", 128_000, "", "2026-06-04"),
    ("NVIDIA", "nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra", 1_000_000, "", "2026-06-04"),
    ("NVIDIA", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Nemotron 3 Nano Omni", 256_000, "", "2026-04-28"),
    ("Google", "google/gemma-4-26b-a4b-it:free", "Gemma 4 26B A4B", 262_144, "", "2026-04-03"),
    ("Google", "google/gemma-4-31b-it:free", "Gemma 4 31B", 262_144, "", "2026-04-02"),
    ("NVIDIA", "nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super", 262_144, "", "2026-03-11"),
    ("NVIDIA", "nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 3 Nano 30B A3B", 256_000, "", "2025-12-14"),
    ("NVIDIA", "nvidia/nemotron-nano-12b-v2-vl:free", "Nemotron Nano 12B 2 VL", 128_000, "2026-08-24", "2025-10-28"),
    ("NVIDIA", "nvidia/nemotron-nano-9b-v2:free", "Nemotron Nano 9B V2", 128_000, "", "2025-09-05"),
    ("OpenAI", "openai/gpt-oss-20b:free", "gpt-oss-20b", 131_072, "", "2025-08-05"),
]

# Os logos são por empresa e nem toda empresa do OpenRouter tem um arquivo.
LOGOS_POR_EMPRESA = {
    "anthropic": "anthropic.JPG",
    "deepseek": "deepseek.JPG",
    "google": "google.JPG",
    "meta": "meta.JPG",
    "meta-llama": "meta.JPG",
    "mistral": "mistral.JPG",
    "mistralai": "mistral.JPG",
    "moonshotai": "moonshot.jpg",
    "openai": "openai.jpg",
    "qwen": "qwen.JPG",
    "x.ai": "xai.JPG",
    "z.ai": "zai.JPG",
}


def _logo_da_empresa(empresa):
    return LOGOS_POR_EMPRESA.get(str(empresa).strip().lower(), "openrouter.JPG")


def _observacao_openrouter(criado, expira, indices=None):
    partes = []
    if criado:
        partes.append(f"Criado no OpenRouter em {criado}.")
    if expira:
        partes.append(f"Expira em {expira}.")
    if indices:
        rotulos = [
            ("Inteligência", indices.get("indice_inteligencia")),
            ("Código", indices.get("indice_codigo")),
            ("Agêntico", indices.get("indice_agentico")),
        ]
        marcados = [f"{rotulo} {valor:g}" for rotulo, valor in rotulos if valor is not None]
        if marcados:
            partes.append("Artificial Analysis: " + ", ".join(marcados) + ".")
    return " ".join(partes)


def _modelo_openrouter(
    *, empresa, modelo_id, nome, contexto, observacao, custo_input="$0.00", custo_output="$0.00"
):
    return _modelo(
        empresa=empresa,
        modelo_id=modelo_id,
        modelo_nome=nome,
        provedor="OpenRouter Free",
        base_url=OPENROUTER_BASE_URL,
        api_tipo="chat_completions",
        api_key_secret="OPENROUTER_API_KEY",
        custo_input_1M=custo_input,
        custo_output_1M=custo_output,
        tier="Free",
        creditos=0,
        contexto_tokens=contexto,
        status="Free",
        fonte="OpenRouter public models API",
        logo=_logo_da_empresa(empresa),
        cor="#e74c3c",
        observacao=observacao,
    )


def _marcar_padroes(modelos):
    """Pré-seleciona os preferidos que existirem na lista carregada."""
    disponiveis = {modelo["modelo_id"] for modelo in modelos}
    padroes = [
        modelo_id for modelo_id in OPENROUTER_PREFERIDOS if modelo_id in disponiveis
    ][:OPENROUTER_MAX_PADRAO]
    if not padroes and modelos:
        padroes = [modelos[0]["modelo_id"]]
    for modelo in modelos:
        modelo["selecionar_padrao"] = modelo["modelo_id"] in padroes
    return modelos


def _carregar_openrouter_free():
    """Monta a fatia OpenRouter do catálogo, ao vivo quando possível."""
    try:
        from openrouter_api import listar_modelos, modelos_gratuitos, normalizar

        resultado = listar_modelos()
        brutos = modelos_gratuitos(resultado.get("modelos") or [])
    except Exception as exc:  # rede, import, JSON malformado: cai no snapshot.
        resultado = {"origem": "indisponivel", "erro": f"{exc.__class__.__name__}: {exc}"}
        brutos = []

    if not brutos:
        modelos = [
            _modelo_openrouter(
                empresa=empresa,
                modelo_id=modelo_id,
                nome=nome,
                contexto=contexto,
                observacao=_observacao_openrouter(criado, expira),
            )
            for empresa, modelo_id, nome, contexto, expira, criado in OPENROUTER_FREE_SNAPSHOT
        ]
        status = {
            "origem": "snapshot",
            "erro": resultado.get("erro", ""),
            "buscado_em": 0.0,
            "total": len(modelos),
        }
        return _marcar_padroes(modelos), status

    modelos = []
    for bruto in brutos:
        dados = normalizar(bruto)
        modelos.append(
            _modelo_openrouter(
                empresa=dados["autor"],
                modelo_id=dados["id"],
                nome=dados["nome"],
                contexto=dados["contexto_tokens"],
                custo_input=dados["custo_input_1M"],
                custo_output=dados["custo_output_1M"],
                observacao=_observacao_openrouter(
                    dados["criado_em"], dados["expira_em"], dados
                ),
            )
        )

    _marcar_padroes(modelos)

    status = {
        "origem": resultado.get("origem", "api"),
        "erro": resultado.get("erro", ""),
        "buscado_em": resultado.get("buscado_em", 0.0),
        "total": len(modelos),
    }
    return modelos, status


modelos_openrouter, OPENROUTER_STATUS = _carregar_openrouter_free()
modelos_db.extend(modelos_openrouter)


def resumo_openrouter():
    """Frase curta sobre a procedência da fatia OpenRouter, para as legendas."""
    origem = OPENROUTER_STATUS.get("origem")
    total = OPENROUTER_STATUS.get("total", 0)
    if origem == "api":
        return f"OpenRouter Free: {total} modelos lidos ao vivo de /api/v1/models."
    if origem == "cache":
        momento = OPENROUTER_STATUS.get("buscado_em") or 0
        quando = (
            datetime.fromtimestamp(momento, tz=timezone.utc).strftime("%d/%m %H:%M UTC")
            if momento
            else "recentemente"
        )
        return f"OpenRouter Free: {total} modelos do cache local (lido em {quando})."
    return (
        f"OpenRouter Free: {total} modelos do snapshot embutido — "
        "não foi possível consultar /api/v1/models."
    )

df_modelos = pd.DataFrame(modelos_db)


def obter_info_modelo(uid_ou_modelo_id):
    """Retorna informacoes de um modelo pelo uid ou pelo modelo_id."""
    modelo = df_modelos[
        (df_modelos["uid"] == uid_ou_modelo_id)
        | (df_modelos["modelo_id"] == uid_ou_modelo_id)
    ]
    if not modelo.empty:
        return modelo.iloc[0].to_dict()
    return None


def obter_modelos_empresa(empresa):
    """Retorna todos os modelos de uma empresa."""
    return df_modelos[df_modelos["empresa"] == empresa]


def obter_modelos_tier(tier):
    """Retorna todos os modelos de um tier especifico."""
    return df_modelos[df_modelos["tier"] == tier]
