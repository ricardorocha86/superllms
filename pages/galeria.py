import streamlit as st

# Página: Galeria de Exemplos
st.title("🖼️ Galeria de Exemplos")
st.caption("Explore exemplos de prompts e respostas dos diferentes modelos")

st.info("🚧 Esta página está em desenvolvimento. Em breve você encontrará aqui uma galeria com exemplos práticos de uso dos modelos de IA.")

# Placeholder para conteúdo futuro
with st.expander("🎨 Conteúdo Planejado"):
    st.markdown("""
    - **Exemplos por Categoria:** Prompts organizados por tipo (criativo, técnico, analítico, etc.)
    - **Comparações Visuais:** Ver como diferentes modelos respondem ao mesmo prompt
    - **Casos de Sucesso:** Exemplos de prompts que geraram excelentes resultados
    - **Filtros Avançados:** Buscar exemplos por modelo, categoria ou complexidade
    - **Contribuições da Comunidade:** Exemplos enviados pelos usuários
    """)
