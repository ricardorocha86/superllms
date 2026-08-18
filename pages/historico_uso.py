"""Histórico de uso da sessão, com tokens e custo real reportados pela API."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import historico
from modelos import df_modelos
from openrouter_api import obter_creditos, obter_geracao


def preco_para_float(valor):
    try:
        return float(str(valor).replace("$", "").replace(",", ".").strip())
    except (TypeError, ValueError):
        return 0.0


@st.cache_data(ttl=300, show_spinner=False)
def carregar_creditos():
    return obter_creditos()


def custo_estimado(evento):
    """Estima o custo pelo preço de tabela, para provedores que não reportam.

    O OpenRouter devolve o custo real em ``custo_usd``; OpenAI, Anthropic, Groq
    e afins não devolvem nada, então o melhor disponível é o preço do catálogo.
    """
    modelo = df_modelos[df_modelos["modelo_id"] == evento.get("modelo_id")]
    if modelo.empty:
        return None
    linha = modelo.iloc[0]
    entrada = (evento.get("tokens_entrada") or 0) / 1_000_000
    saida = (evento.get("tokens_saida") or 0) / 1_000_000
    return entrada * preco_para_float(linha["custo_input_1M"]) + saida * preco_para_float(
        linha["custo_output_1M"]
    )


st.title("Histórico de Uso")

eventos = historico.obter()

creditos = carregar_creditos()
if creditos:
    total = float(creditos.get("total_credits") or 0)
    usado = float(creditos.get("total_usage") or 0)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Créditos OpenRouter", f"US$ {total:.2f}")
    col_b.metric("Já consumido", f"US$ {usado:.4f}")
    col_c.metric("Saldo", f"US$ {total - usado:.2f}")
    if total - usado <= 0:
        st.warning(
            "Sem saldo no OpenRouter: apenas modelos `:free` respondem. "
            "Os demais retornam HTTP 402.",
            icon=":material/account_balance_wallet:",
        )
else:
    st.caption("Saldo do OpenRouter indisponível (sem `OPENROUTER_API_KEY` ou sem rede).")

st.divider()

if not eventos:
    st.info(
        "Nenhuma chamada registrada nesta sessão. Use o Chatbot ou a Engenharia de "
        "Prompt e volte aqui — cada resposta registra tokens e custo automaticamente.",
        icon=":material/history:",
    )
    st.stop()

df = pd.DataFrame(eventos)
df["custo_estimado_usd"] = df.apply(custo_estimado, axis=1)
# O custo real só vem do OpenRouter; para o resto vale a estimativa de tabela.
df["custo_usd_final"] = df["custo_usd"].fillna(df["custo_estimado_usd"])
df["custo_e_real"] = df["custo_usd"].notna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Chamadas", len(df))
col2.metric("Tokens de entrada", f"{int(df['tokens_entrada'].fillna(0).sum()):,}".replace(",", "."))
col3.metric("Tokens de saída", f"{int(df['tokens_saida'].fillna(0).sum()):,}".replace(",", "."))
col4.metric("Custo total", f"US$ {df['custo_usd_final'].fillna(0).sum():.6f}")

reais = int(df["custo_e_real"].sum())
st.caption(
    f"{reais} de {len(df)} chamadas têm custo real reportado pela API (OpenRouter). "
    "As demais são estimadas pelo preço de tabela do catálogo."
)

raciocinio = int(df["tokens_raciocinio"].fillna(0).sum())
if raciocinio:
    st.caption(f"Tokens de raciocínio incluídos na saída: {raciocinio:,}".replace(",", "."))

st.markdown("### Por modelo")
por_modelo = (
    df.groupby(["provedor", "modelo_nome", "modelo_id"], as_index=False)
    .agg(
        chamadas=("modelo_id", "size"),
        tokens_entrada=("tokens_entrada", "sum"),
        tokens_saida=("tokens_saida", "sum"),
        custo_usd=("custo_usd_final", "sum"),
    )
    .sort_values("custo_usd", ascending=False)
)
st.dataframe(
    por_modelo.rename(
        columns={
            "provedor": "Provedor",
            "modelo_nome": "Modelo",
            "modelo_id": "Model ID",
            "chamadas": "Chamadas",
            "tokens_entrada": "Tokens in",
            "tokens_saida": "Tokens out",
            "custo_usd": "Custo (US$)",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={"Custo (US$)": st.column_config.NumberColumn(format="%.6f")},
)

st.markdown("### Chamadas")
colunas = [
    "quando",
    "origem",
    "provedor",
    "modelo_nome",
    "tokens_entrada",
    "tokens_saida",
    "tokens_raciocinio",
    "custo_usd_final",
    "custo_e_real",
]
st.dataframe(
    df[colunas]
    .sort_values("quando", ascending=False)
    .rename(
        columns={
            "quando": "Quando (UTC)",
            "origem": "Origem",
            "provedor": "Provedor",
            "modelo_nome": "Modelo",
            "tokens_entrada": "Tokens in",
            "tokens_saida": "Tokens out",
            "tokens_raciocinio": "Raciocínio",
            "custo_usd_final": "Custo (US$)",
            "custo_e_real": "Custo real",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={"Custo (US$)": st.column_config.NumberColumn(format="%.6f")},
)

col_export, col_limpar = st.columns(2)
with col_export:
    st.download_button(
        "Exportar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="superllms_historico_uso.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
with col_limpar:
    if st.button("Limpar histórico", icon=":material/delete_sweep:", width="stretch"):
        historico.limpar()
        st.rerun()

with st.expander("Auditar uma geração no OpenRouter"):
    st.caption(
        "O endpoint `/generation` devolve os metadados consolidados de uma chamada "
        "(tokens nativos, custo, provedor que atendeu e latência)."
    )
    ids = [
        evento["generation_id"]
        for evento in eventos
        if evento.get("generation_id") and evento.get("provedor") == "OpenRouter Free"
    ]
    if not ids:
        st.caption("Nenhuma chamada via OpenRouter nesta sessão.")
    else:
        escolhido = st.selectbox("Generation ID", options=list(reversed(ids)))
        if st.button("Consultar", icon=":material/search:"):
            with st.spinner("Consultando o OpenRouter..."):
                dados = obter_geracao(escolhido)
            if dados:
                st.json(dados)
            else:
                st.warning(
                    "Sem dados ainda. O OpenRouter leva alguns segundos para "
                    "consolidar a geração — tente de novo em instantes."
                )
