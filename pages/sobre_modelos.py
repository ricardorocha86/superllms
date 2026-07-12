import pandas as pd
import streamlit as st

from modelos import DATA_ATUALIZACAO, df_modelos


def formatar_contexto(valor):
    if pd.isna(valor) or valor is None:
        return "N/D"
    valor = int(valor)
    if valor >= 1_000_000:
        return f"{valor / 1_000_000:g}M"
    if valor >= 1_000:
        return f"{valor // 1_000}k"
    return str(valor)


st.title("Sobre os Modelos")
st.caption(f"Base de modelos atualizada em {DATA_ATUALIZACAO}.")

st.markdown(
    """
Esta página agora é gerada a partir da mesma base usada pelo laboratório.
Assim, o `model_id`, o provedor, os custos e o status exibidos aqui são os mesmos usados na hora de testar.
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Modelos", len(df_modelos))
col2.metric("Provedores", df_modelos["provedor"].nunique())
col3.metric("OpenRouter Free", int((df_modelos["provedor"] == "OpenRouter Free").sum()))
col4.metric("Preview", int((df_modelos["status"] == "Preview").sum()))

st.markdown("### Fontes consultadas")
st.markdown(
    """
- [OpenAI API Models](https://platform.openai.com/docs/models)
- [Google Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [Groq Supported Models](https://console.groq.com/docs/models)
- [OpenRouter Models API](https://openrouter.ai/api/v1/models)
"""
)

st.markdown("### Consulta rápida")
provedores = sorted(df_modelos["provedor"].unique())
status = sorted(df_modelos["status"].unique())

col_filtro1, col_filtro2 = st.columns(2)
with col_filtro1:
    provedores_escolhidos = st.multiselect(
        "Provedores", provedores, default=provedores
    )
with col_filtro2:
    status_escolhidos = st.multiselect("Status", status, default=status)

df_filtrado = df_modelos[
    df_modelos["provedor"].isin(provedores_escolhidos)
    & df_modelos["status"].isin(status_escolhidos)
].copy()
df_filtrado["contexto"] = df_filtrado["contexto_tokens"].apply(formatar_contexto)

colunas = [
    "provedor",
    "empresa",
    "modelo_nome",
    "modelo_id",
    "status",
    "tier",
    "custo_input_1M",
    "custo_output_1M",
    "contexto",
    "fonte",
    "observacao",
]

st.dataframe(
    df_filtrado[colunas].rename(
        columns={
            "provedor": "Provedor",
            "empresa": "Empresa",
            "modelo_nome": "Modelo",
            "modelo_id": "Model ID",
            "status": "Status",
            "tier": "Tier",
            "custo_input_1M": "Input/1M",
            "custo_output_1M": "Output/1M",
            "contexto": "Contexto",
            "fonte": "Fonte",
            "observacao": "Observação",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.markdown("### Por provedor")
for provedor in provedores:
    with st.expander(provedor, expanded=provedor in {"Groq", "OpenRouter Free"}):
        df_provedor = df_modelos[df_modelos["provedor"] == provedor].copy()
        df_provedor["contexto"] = df_provedor["contexto_tokens"].apply(formatar_contexto)
        st.dataframe(
            df_provedor[
                [
                    "empresa",
                    "modelo_nome",
                    "modelo_id",
                    "status",
                    "custo_input_1M",
                    "custo_output_1M",
                    "contexto",
                    "observacao",
                ]
            ].rename(
                columns={
                    "empresa": "Empresa",
                    "modelo_nome": "Modelo",
                    "modelo_id": "Model ID",
                    "status": "Status",
                    "custo_input_1M": "Input/1M",
                    "custo_output_1M": "Output/1M",
                    "contexto": "Contexto",
                    "observacao": "Observação",
                }
            ),
            hide_index=True,
            width="stretch",
        )
