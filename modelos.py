# Base de dados dos modelos de IA
import pandas as pd

modelos_db = [
    {"empresa": "OpenAI", "modelo_id": "gpt-5-nano", "modelo_nome": "GPT-5 Nano", "base_url": "https://api.openai.com/v1", "custo_input_1M": "$0.05", "custo_output_1M": "$0.40", "tier": "🥉 Básico", "creditos": 1, "cor": "#1f77b4", "logo": "openai.jpg"},
    {"empresa": "OpenAI", "modelo_id": "gpt-5-mini", "modelo_nome": "GPT-5 Mini", "base_url": "https://api.openai.com/v1", "custo_input_1M": "$0.25", "custo_output_1M": "$2.00", "tier": "🥈 Pro", "creditos": 2, "cor": "#1f77b4", "logo": "openai.jpg"},
    {"empresa": "OpenAI", "modelo_id": "gpt-5", "modelo_nome": "GPT-5", "base_url": "https://api.openai.com/v1", "custo_input_1M": "$1.25", "custo_output_1M": "$10.00", "tier": "🥇 Elite", "creditos": 10, "cor": "#1f77b4", "logo": "openai.jpg"},
    
    {"empresa": "Google", "modelo_id": "gemini-2.5-flash-lite", "modelo_nome": "Gemini 2.5 Flash Lite", "base_url": "https://generativelanguage.googleapis.com/v1beta", "custo_input_1M": "$0.10", "custo_output_1M": "$0.40", "tier": "🥉 Básico", "creditos": 1, "cor": "#ff7f0e", "logo": "google.jpg"},
    {"empresa": "Google", "modelo_id": "gemini-2.5-flash", "modelo_nome": "Gemini 2.5 Flash", "base_url": "https://generativelanguage.googleapis.com/v1beta", "custo_input_1M": "$0.30", "custo_output_1M": "$2.50", "tier": "🥈 Pro", "creditos": 2, "cor": "#ff7f0e", "logo": "google.jpg"},
    {"empresa": "Google", "modelo_id": "gemini-2.5-pro", "modelo_nome": "Gemini 2.5 Pro", "base_url": "https://generativelanguage.googleapis.com/v1beta", "custo_input_1M": "$1.25", "custo_output_1M": "$10.00", "tier": "🥇 Elite", "creditos": 10, "cor": "#ff7f0e", "logo": "google.jpg"},
    
    {"empresa": "X.AI", "modelo_id": "grok-3-mini", "modelo_nome": "Grok 3 Mini", "base_url": "https://api.x.ai/v1", "custo_input_1M": "$0.30", "custo_output_1M": "$0.50", "tier": "🥉 Básico", "creditos": 1, "cor": "#2ca02c", "logo": "xai.jpg"},
    {"empresa": "X.AI", "modelo_id": "grok-code-fast-1", "modelo_nome": "Grok Code Fast", "base_url": "https://api.x.ai/v1", "custo_input_1M": "$0.20", "custo_output_1M": "$1.50", "tier": "🥈 Pro", "creditos": 2, "cor": "#2ca02c", "logo": "xai.jpg"},
    {"empresa": "X.AI", "modelo_id": "grok-4-0709", "modelo_nome": "Grok 4", "base_url": "https://api.x.ai/v1", "custo_input_1M": "$3.00", "custo_output_1M": "$15.00", "tier": "🥇 Elite", "creditos": 10, "cor": "#2ca02c", "logo": "xai.jpg"},
    
    {"empresa": "Anthropic", "modelo_id": "claude-3-5-haiku-20241022", "modelo_nome": "Claude 3.5 Haiku", "base_url": "https://api.anthropic.com/v1/", "custo_input_1M": "$0.80", "custo_output_1M": "$4.00", "tier": "🥈 Pro", "creditos": 2, "cor": "#d62728", "logo": "anthropic.jpg"},
    {"empresa": "Anthropic", "modelo_id": "claude-sonnet-4-20250514", "modelo_nome": "Claude Sonnet 4", "base_url": "https://api.anthropic.com/v1/", "custo_input_1M": "$3.00", "custo_output_1M": "$15.00", "tier": "🥇 Elite", "creditos": 10, "cor": "#d62728", "logo": "anthropic.jpg"},
    
    {"empresa": "DeepSeek", "modelo_id": "deepseek-chat", "modelo_nome": "DeepSeek V3.1 Chat", "base_url": "https://api.deepseek.com", "custo_input_1M": "$0.56", "custo_output_1M": "$1.68", "tier": "🥈 Pro", "creditos": 2, "cor": "#9467bd", "logo": "deepseek.jpg"},
    {"empresa": "DeepSeek", "modelo_id": "deepseek-reasoner", "modelo_nome": "DeepSeek V3.1 Reasoner", "base_url": "https://api.deepseek.com", "custo_input_1M": "$0.56", "custo_output_1M": "$1.68", "tier": "🥈 Pro", "creditos": 2, "cor": "#9467bd", "logo": "deepseek.jpg"},
    
    {"empresa": "Z.AI", "modelo_id": "z-ai/glm-4.5-air:free", "modelo_nome": "GLM 4.5 Air", "base_url": "https://openrouter.ai/api/v1", "custo_input_1M": "$0.14", "custo_output_1M": "$0.86", "tier": "🥈 Pro", "creditos": 2, "cor": "#8e44ad", "logo": "zai.jpg"},
    
    {"empresa": "OpenRouter", "modelo_id": "openrouter/sonoma-sky-alpha", "modelo_nome": "Sonoma Sky Alpha", "base_url": "https://openrouter.ai/api/v1", "custo_input_1M": "$3.00", "custo_output_1M": "$15.00", "tier": "🥇 Elite", "creditos": 10, "cor": "#e74c3c", "logo": "openrouter.jpg"},
    {"empresa": "OpenRouter", "modelo_id": "openrouter/sonoma-dusk-alpha", "modelo_nome": "Sonoma Dusk Alpha", "base_url": "https://openrouter.ai/api/v1", "custo_input_1M": "$0.20", "custo_output_1M": "$1.50", "tier": "🥈 Pro", "creditos": 2, "cor": "#e74c3c", "logo": "openrouter.jpg"},
    
    {"empresa": "Qwen", "modelo_id": "qwen/qwen3-32b", "modelo_nome": "Qwen 3 32B", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$0.29", "custo_output_1M": "$0.59", "tier": "🥉 Básico", "creditos": 1, "cor": "#27ae60", "logo": "qwen.jpg"},
    
    {"empresa": "Moonshot AI", "modelo_id": "moonshotai/kimi-k2-instruct", "modelo_nome": "Kimi K2", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$1.00", "custo_output_1M": "$3.00", "tier": "🥈 Pro", "creditos": 2, "cor": "#f39c12", "logo": "moonshot.jpg"},
    
    {"empresa": "Mistral AI", "modelo_id": "mistralai/mistral-small-3.2-24b-instruct:free", "modelo_nome": "Mistral Small 3.2", "base_url": "https://openrouter.ai/api/v1", "custo_input_1M": "$0.10", "custo_output_1M": "$0.30", "tier": "🥉 Básico", "creditos": 1, "cor": "#3498db", "logo": "mistral.jpg"},
    {"empresa": "Mistral AI", "modelo_id": "mistralai/mistral-nemo:free", "modelo_nome": "Mistral Nemo", "base_url": "https://openrouter.ai/api/v1", "custo_input_1M": "$0.15", "custo_output_1M": "$0.15", "tier": "🥉 Básico", "creditos": 1, "cor": "#3498db", "logo": "mistral.jpg"},
    
    {"empresa": "Meta", "modelo_id": "meta-llama/llama-4-maverick-17b-128e-instruct", "modelo_nome": "Llama 4 Maverick", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$0.20", "custo_output_1M": "$0.60", "tier": "🥈 Pro", "creditos": 2, "cor": "#9b59b6", "logo": "meta.jpg"},
    {"empresa": "Meta", "modelo_id": "meta-llama/llama-4-scout-17b-16e-instruct", "modelo_nome": "Llama 4 Scout", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$0.11", "custo_output_1M": "$0.34", "tier": "🥉 Básico", "creditos": 1, "cor": "#9b59b6", "logo": "meta.jpg"},
    
    {"empresa": "OpenAI", "modelo_id": "openai/gpt-oss-120b", "modelo_nome": "GPT-OSS 120B", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$0.15", "custo_output_1M": "$0.75", "tier": "🥈 Pro", "creditos": 2, "cor": "#16a085", "logo": "openai.jpg"},
    {"empresa": "OpenAI", "modelo_id": "openai/gpt-oss-20b", "modelo_nome": "GPT-OSS 20B", "base_url": "https://api.groq.com/openai/v1", "custo_input_1M": "$0.10", "custo_output_1M": "$0.50", "tier": "🥉 Básico", "creditos": 1, "cor": "#16a085", "logo": "openai.jpg"}
]

# Criar DataFrame
df_modelos = pd.DataFrame(modelos_db)

# Função para obter informações de um modelo específico
def obter_info_modelo(modelo_id):
    """Retorna informações de um modelo específico"""
    modelo = df_modelos[df_modelos['modelo_id'] == modelo_id]
    if not modelo.empty:
        return modelo.iloc[0].to_dict()
    return None

# Função para obter modelos por empresa
def obter_modelos_empresa(empresa):
    """Retorna todos os modelos de uma empresa"""
    return df_modelos[df_modelos['empresa'] == empresa]

# Função para obter modelos por tier
def obter_modelos_tier(tier):
    """Retorna todos os modelos de um tier específico"""
    return df_modelos[df_modelos['tier'] == tier]