"""Entrada independente do laboratório, útil enquanto a navegação principal evolui."""

import runpy
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Probabilidades de Tokens · SuperLLMs", layout="wide")
runpy.run_path(Path(__file__).parent / "pages" / "token_probabilities_v2.py", run_name="__main__")
