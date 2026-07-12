"""Chatbot efêmero com a mesma configuração do Laboratório de Modelos."""

from __future__ import annotations

import streamlit as st

from ai_helpers import generate_text, get_secret
from modelos import DATA_ATUALIZACAO, df_modelos
from personas import PERSONALIDADES, construir_prompt_final


AVATARES = {"user": "🙂", "assistant": "⚡"}


def modelo_label(modelo):
    return f"{modelo['provedor']} | {modelo['empresa']} — {modelo['modelo_nome']} | {modelo['modelo_id']}"


with st.sidebar:
    st.markdown("### Configurações de personalidade")
    selecoes_personalidade = {}
    for chave, dados in PERSONALIDADES.items():
        selecoes_personalidade[chave] = st.checkbox(
            dados["label"],
            value=dados.get("default", False),
            key=f"chat_personalidade_{chave}",
            disabled=dados.get("disabled", False),
            help=dados["instrucao"],
        )

    tamanho_resposta = st.slider(
        "Tamanho da resposta (palavras)",
        10,
        200,
        80,
        10,
        key="chat_tamanho_resposta",
    )

    st.divider()
    st.markdown("### Modelo")
    st.caption(f"Base compartilhada com o Laboratório de Modelos · atualizada em {DATA_ATUALIZACAO}.")
    df_modelos_chat = df_modelos.sort_values(
        ["provedor", "status", "empresa", "modelo_nome"], kind="stable"
    ).copy()
    opcoes_modelos = df_modelos_chat["uid"].tolist()
    uids_padrao = df_modelos_chat.loc[
        df_modelos_chat["selecionar_padrao"].fillna(False).astype(bool), "uid"
    ].tolist()
    uid_padrao = uids_padrao[0] if uids_padrao else opcoes_modelos[0]
    uid_modelo = st.selectbox(
        "Modelo",
        options=opcoes_modelos,
        index=opcoes_modelos.index(uid_padrao),
        format_func=lambda uid: modelo_label(df_modelos_chat.loc[df_modelos_chat["uid"] == uid].iloc[0].to_dict()),
        key="chat_modelo_uid",
    )
    modelo_info = df_modelos_chat.loc[df_modelos_chat["uid"] == uid_modelo].iloc[0].to_dict()
    st.caption(f"Chave usada automaticamente: `{modelo_info['api_key_secret']}`")
    with st.expander("Chaves esperadas"):
        st.code(
            "\n".join(sorted(df_modelos["api_key_secret"].dropna().unique())),
            language="text",
        )

    if st.button("Nova conversa", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()
    st.divider()
    st.caption("A conversa fica somente na memória desta sessão e não é salva em banco.")

config_signature = (
    modelo_info["uid"],
    tuple((chave, valor) for chave, valor in selecoes_personalidade.items()),
    tamanho_resposta,
)
if st.session_state.get("chat_config_signature") != config_signature:
    st.session_state["chat_config_signature"] = config_signature
    st.session_state["chat_messages"] = []

st.title("Chatbot")
st.caption(f"{modelo_label(modelo_info)} · conversa temporária")

for mensagem in st.session_state.get("chat_messages", []):
    with st.chat_message(mensagem["role"], avatar=AVATARES.get(mensagem["role"])):
        st.markdown(mensagem["content"])

if not st.session_state.get("chat_messages"):
    st.info("Escreva uma mensagem para iniciar. Trocar modelo ou configurações começa uma nova conversa.")

prompt = st.chat_input("Digite sua mensagem...")
if prompt:
    mensagens = st.session_state.setdefault("chat_messages", [])
    with st.chat_message("user", avatar=AVATARES["user"]):
        st.markdown(prompt)
    mensagens.append({"role": "user", "content": prompt})

    # Este é exatamente o mesmo montador usado pelo Laboratório de Modelos:
    # somente tamanho e personas selecionadas entram no prompt enviado.
    prompt_final = construir_prompt_final(prompt, tamanho_resposta, selecoes_personalidade)
    mensagens_para_api = mensagens[:-1] + [{"role": "user", "content": prompt_final}]
    api_key = get_secret(modelo_info["api_key_secret"])

    with st.chat_message("assistant", avatar=AVATARES["assistant"]):
        with st.spinner("Pensando...", show_time=True):
            try:
                resposta = generate_text(
                    model_info=modelo_info,
                    api_key=api_key,
                    messages=mensagens_para_api,
                    instructions="",
                    temperature=None,
                )
                if not resposta:
                    resposta = "O modelo não retornou texto. Tente novamente com outro modelo."
                st.markdown(resposta)
                mensagens.append({"role": "assistant", "content": resposta})
            except Exception as exc:
                mensagens.pop()
                st.error(f"Não foi possível gerar a resposta: {exc}")
