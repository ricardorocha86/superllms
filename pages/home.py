import streamlit as st


st.title("SuperLLMs")
st.write(
    "Um espaço para aprender e experimentar com inteligência artificial. "
    "Compare modelos de linguagem, explore embeddings, crie imagens e entenda "
    "como os modelos escolhem os próximos tokens de uma resposta."
)

pages = [
    (":material/science:", "Laboratório de Modelos", "Compare respostas de diferentes modelos de IA.", "pages/llms.py"),
    (":material/hub:", "Embedding Lab", "Explore similaridade e busca semântica.", "pages/embedding_lab.py"),
    (":material/palette:", "Playground de Imagem", "Crie e edite imagens com IA.", "pages/playground_imagem.py"),
    (":material/percent:", "Probabilidades de Tokens", "Entenda a geração de texto e o efeito da temperatura.", "pages/token_probabilities_v2.py"),
]

for start in range(0, len(pages), 2):
    for column, (icon, title, description, target) in zip(st.columns(2, gap="large"), pages[start:start + 2]):
        with column:
            with st.container(border=True):
                st.page_link(target, label=title, icon=icon, use_container_width=True)
                st.caption(description)
