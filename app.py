import streamlit as st


st.set_page_config(
    page_title="SuperLLMs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A ordem da navegação é também a ordem de descoberta mais útil para quem chega.
pages = {
    "Comece aqui": [
        st.Page("pages/home.py", title="Home", icon=":material/home:"),
        st.Page("pages/chatbot.py", title="Chatbot", icon=":material/chat:"),
        st.Page("pages/otimizador_prompt.py", title="Engenharia de Prompt", icon=":material/auto_fix_high:"),
    ],
    "Laboratórios": [
        st.Page("pages/llms.py", title="Laboratório de Modelos", icon=":material/science:"),
        st.Page("pages/embedding_lab.py", title="Embedding Lab", icon=":material/hub:"),
        st.Page("pages/playground_imagem.py", title="Playground de Imagem", icon=":material/palette:"),
    ],
    "Referência": [
        st.Page("pages/sobre_modelos.py", title="Modelos e Provedores", icon=":material/menu_book:"),
    ],
    "Pessoal": [
        st.Page("pages/meu_perfil.py", title="Meu Perfil", icon=":material/person:"),
        st.Page("pages/historico_uso.py", title="Histórico de Uso", icon=":material/analytics:"),
    ],
}

pg = st.navigation(pages)
pg.run()
