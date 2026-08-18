"""Registro de consumo das chamadas feitas durante a sessão.

Fica em ``st.session_state`` de propósito: o app não tem banco, e a promessa
feita ao usuário no Chatbot é que nada é persistido. Quem quiser guardar exporta
o CSV na página de Histórico.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


CHAVE_SESSAO = "historico_uso"
LIMITE_EVENTOS = 500


def registrar(uso: dict[str, Any], *, origem: str) -> None:
    """Guarda um evento de uso. ``origem`` é a página que fez a chamada."""
    if not uso:
        return
    eventos = st.session_state.setdefault(CHAVE_SESSAO, [])
    eventos.append({**uso, "origem": origem})
    # Sessões longas não podem crescer sem limite na memória do servidor.
    if len(eventos) > LIMITE_EVENTOS:
        del eventos[: len(eventos) - LIMITE_EVENTOS]


def obter() -> list[dict[str, Any]]:
    return list(st.session_state.get(CHAVE_SESSAO, []))


def limpar() -> None:
    st.session_state[CHAVE_SESSAO] = []
