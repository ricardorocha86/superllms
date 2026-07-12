import streamlit as st


st.set_page_config(page_title="SuperLLMs — Home", page_icon="⚡", layout="wide")

st.title("⚡ SuperLLMs")
st.subheader("Um espaço para aprender, testar e construir com modelos de IA")
st.write(
    "Escolha um caminho abaixo para começar. A plataforma reúne laboratório de modelos, "
    "chat, engenharia de prompt, embeddings e geração de imagens em um só lugar."
)

st.markdown("## Comece por aqui")
cards = [
    ("💬", "Chatbot", "Converse com uma persona usando qualquer modelo configurado na base.", "pages/chatbot.py"),
    ("✨", "Engenharia de Prompt", "Aprenda a escrever instruções melhores e use o otimizador de prompt com IA.", "pages/otimizador_prompt.py"),
    ("🔬", "Laboratório de Modelos", "Compare provedores, modelos, estilos de resposta e custos.", "pages/llms.py"),
]
CARD_IMAGES = {
    "Chatbot": "assets/home/chatbot.png",
    "Engenharia de Prompt": "assets/home/prompt-engineering.png",
    "Laboratório de Modelos": "assets/home/models-lab.png",
}

for start in range(0, len(cards), 2):
    columns = st.columns(2, gap="large")
    for column, (icon, title, description, target) in zip(columns, cards[start : start + 2]):
        with column:
            with st.container(border=True):
                st.image(CARD_IMAGES[title], use_container_width=True)
                st.markdown(f"### {icon} {title}")
                st.write(description)
                st.page_link(target, label=f"Abrir {title}", icon=icon, use_container_width=True)

st.markdown("## O que você pode fazer")
use_cases = [
    ("Aprender", "Leia as aulas de engenharia de prompt e embedding, com exemplos executáveis e explicações curtas."),
    ("Experimentar", "Rode o mesmo pedido em modelos diferentes e observe qualidade, latência e custo."),
    ("Construir", "Transforme um pedido informal em um prompt com contexto, critérios de sucesso e formato de saída."),
    ("Criar", "Gere imagens com referências ou use o chatbot para explorar ideias antes de implementá-las."),
]
columns = st.columns(4)
for column, (title, description) in zip(columns, use_cases):
    with column:
        st.markdown(f"### {title}")
        st.caption(description)

st.markdown("## Exemplos rápidos")
with st.expander("📝 Organizar uma reunião"):
    st.code(
        "Você é um secretário executivo. A partir das notas abaixo, produza: "
        "(1) decisões, (2) responsáveis, (3) prazos e (4) perguntas em aberto. "
        "Não invente informações; marque lacunas como 'não informado'.\n\n[notas]",
        language="text",
    )
with st.expander("📊 Explorar uma base textual"):
    st.code(
        "Use o Laboratório de Modelos para comparar respostas, provedores, "
        "personas, tempo de geração e custo estimado.",
        language="text",
    )
with st.expander("🧩 Melhorar um pedido vago"):
    st.code(
        "Pedido vago: 'faça um post sobre meu produto'\n\n"
        "No Otimizador, informe público, objetivo, contexto, restrições e formato "
        "para receber uma versão testável e reutilizável.",
        language="text",
    )

st.divider()
st.caption("Dica: comece pelo Chatbot para testar uma ideia e depois leve o pedido ao Otimizador de Prompt.")
