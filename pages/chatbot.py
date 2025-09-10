import streamlit as st

# Página: ChatBot
st.title("💬 ChatBot")
st.caption("Converse com os modelos de IA em formato de chat")

st.info("🚧 Esta página está em desenvolvimento. Em breve você poderá ter conversas contínuas com os modelos de IA em formato de chat.")

# Placeholder para conteúdo futuro
with st.expander("💭 Funcionalidades Planejadas"):
    st.markdown("""
    - **Chat Contínuo:** Manter contexto de conversa com os modelos
    - **Múltiplos Modelos:** Conversar simultaneamente com diferentes modelos
    - **Histórico de Chat:** Salvar e revisar conversas anteriores
    - **Exportar Conversas:** Baixar conversas em diferentes formatos
    - **Configurações Avançadas:** Ajustar temperatura, tokens e outros parâmetros
    """)
