import streamlit as st

# Configuração da página principal
st.set_page_config(
    page_title="SuperLLMs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir as páginas
pages = {
    "SuperLLMs": [
        st.Page("pages/llms.py", title="Laboratório de Modelos", icon="🔬"),
        st.Page("pages/sobre_modelos.py", title="Sobre os Modelos", icon="📚"),
        st.Page("pages/otimizador_prompt.py", title="Otimizador de Prompt", icon="🎯"),
        st.Page("pages/galeria.py", title="Galeria de Exemplos", icon="🖼️"),
        st.Page("pages/chatbot.py", title="ChatBot", icon="💬")
    ],
    "Pessoal": [
        st.Page("pages/meu_perfil.py", title="Meu Perfil", icon="👤"),
        st.Page("pages/historico_uso.py", title="Histórico de Uso", icon="📊")
    ]
}

# Criar a navegação
pg = st.navigation(pages)

# Executar a página selecionada
pg.run()