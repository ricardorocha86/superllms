"""Aula e ferramenta prática de engenharia de prompt e de contexto."""

from __future__ import annotations

import streamlit as st

from ai_helpers import generate_text, get_secret


st.set_page_config(page_title="Engenharia de Prompt", page_icon="✨", layout="wide")


def montar_prompt_local(pedido: str, contexto: str, publico: str, formato: str, restricoes: str, idioma: str) -> str:
    """Fallback determinístico: mantém a ferramenta útil sem uma chave de API."""
    blocos = [
        "# Papel", "Você é um especialista no domínio do pedido e deve priorizar precisão, clareza e utilidade.",
        "\n# Objetivo", pedido.strip(),
        "\n# Contexto disponível", contexto.strip() or "Nenhum contexto adicional foi fornecido.",
        "\n# Público", publico.strip() or "Defina o nível de conhecimento do público antes de responder.",
        "\n# Tarefa", "Execute o objetivo em etapas curtas, valide as premissas e destaque o que não estiver informado.",
        "\n# Formato de saída", formato.strip() or "Use títulos curtos, listas quando ajudarem e uma conclusão acionável.",
        "\n# Restrições", restricoes.strip() or "Não invente fatos, fontes, números ou requisitos.",
        "\n# Idioma", idioma,
    ]
    return "\n".join(blocos)


def limpar_prompt_gerado(texto: str) -> str:
    """Remove apenas molduras de Markdown para deixar a saída copiável."""
    resultado = (texto or "").strip()
    if "```" in resultado:
        partes = [parte.strip() for parte in resultado.split("```") if parte.strip()]
        if partes:
            resultado = max(partes, key=len)
            if resultado.splitlines() and resultado.splitlines()[0].lower() in {"markdown", "text", "txt"}:
                resultado = "\n".join(resultado.splitlines()[1:]).strip()
    for prefixo in ("Prompt otimizado:", "Prompt final:", "Novo prompt:"):
        if resultado.lower().startswith(prefixo.lower()):
            resultado = resultado[len(prefixo):].lstrip(" \n:")
    return resultado.strip()


def otimizar_com_llm(pedido: str, contexto: str, publico: str, formato: str, restricoes: str, idioma: str) -> str:
    model_info = {
        "modelo_id": "gpt-5.6",
        "api_tipo": "openai_responses",
        "base_url": "",
    }
    briefing = f"""Pedido original:
{pedido}

Contexto:
{contexto or "não informado"}

Público:
{publico or "não informado"}

Formato desejado:
{formato or "não informado"}

Restrições:
{restricoes or "não informado"}

Idioma: {idioma}
"""
    instructions = """Transforme o pedido em um único prompt final, pronto para copiar e colar em qualquer modelo.
Retorne somente o texto do novo prompt, sem título, explicação, análise, lista de melhorias, introdução ou conclusão.
Não use bloco de código Markdown.
Se alguma informação estiver ausente, use um marcador [preencher] em vez de inventar.
Escreva o prompt em português do Brasil."""
    return generate_text(
        model_info=model_info,
        api_key=get_secret("OPENAI_API_KEY"),
        messages=[{"role": "user", "content": briefing}],
        instructions=instructions,
    )


st.title("✨ Engenharia de Prompt e de Contexto")
st.caption("Aprenda a transformar uma intenção vaga em uma instrução testável, reutilizável e fácil de avaliar.")

aba_aula, aba_ferramenta, aba_checklist = st.tabs(["Aula", "Otimizador com IA", "Checklist"])

with aba_aula:
    st.markdown("## 1. Prompt é contrato, não mágica")
    st.write(
        "Um bom prompt reduz ambiguidades. Ele explica o objetivo, fornece contexto suficiente, "
        "define limites e descreve como reconhecer uma boa resposta. O modelo ainda pode errar; "
        "a diferença é que você consegue observar, testar e corrigir o comportamento."
    )

    st.markdown("## 2. Engenharia de prompt")
    st.markdown(
        """
        A engenharia de prompt organiza a instrução que você dá ao modelo:

        - **Papel:** qual perspectiva ou competência deve ser usada.
        - **Objetivo:** qual resultado precisa ser produzido.
        - **Critérios:** o que torna a resposta correta ou útil.
        - **Formato:** tabela, lista, JSON, roteiro, código ou outro formato.
        - **Restrições:** limites de tamanho, tom, fontes, idioma e segurança.
        - **Exemplos:** casos de entrada e saída quando o padrão for difícil de explicar.
        """
    )
    st.code(
        "Você é um analista de produto.\n"
        "Objetivo: resumir as entrevistas abaixo em temas recorrentes.\n"
        "Critérios: separe evidência de interpretação e não invente dados.\n"
        "Formato: tabela com tema, evidência, frequência e próxima pergunta.\n"
        "Contexto: [cole as entrevistas aqui]",
        language="text",
    )

    st.markdown("## 3. Engenharia de contexto")
    st.write(
        "Contexto é tudo que o sistema precisa saber antes de responder: documentos, histórico, "
        "dados do usuário, regras do negócio, ferramentas disponíveis e estado atual da tarefa. "
        "A pergunta pode estar bem escrita e ainda assim falhar se o contexto estiver incompleto, "
        "desatualizado ou misturado sem hierarquia."
    )
    st.markdown(
        """
        Uma arquitetura simples de contexto é:

        1. **Instruções estáveis:** regras da aplicação e critérios de segurança.
        2. **Contexto confiável:** fatos, documentos e dados recuperados.
        3. **Estado da tarefa:** o que já foi decidido e o que falta fazer.
        4. **Pedido atual:** a ação que o usuário quer agora.
        5. **Saída esperada:** como o resultado será consumido ou avaliado.
        """
    )
    with st.expander("Exemplo: prompt sem contexto vs. prompt com contexto", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sem contexto**")
            st.code("Resuma esta política e diga o que devo fazer.", language="text")
        with col2:
            st.markdown("**Com contexto**")
            st.code(
                "Você é um atendente de suporte. Use somente a política abaixo.\n"
                "Cliente: pessoa física, compra feita há 12 dias.\n"
                "Objetivo: informar se a devolução é elegível.\n"
                "Saída: decisão, trecho da regra usada e próximo passo.\n"
                "Política: [documento versionado]",
                language="text",
            )

    st.markdown("## 4. Como avaliar")
    st.write(
        "Não avalie um prompt por uma resposta bonita. Monte um pequeno conjunto de casos, "
        "defina critérios observáveis e compare versões. Exemplos de critérios: completude, "
        "fidelidade ao contexto, formato válido, tom adequado e ação correta."
    )

with aba_ferramenta:
    st.markdown("## Transforme seu pedido em um prompt pronto")
    st.info(
        "Quando `OPENAI_API_KEY` estiver configurada, o botão usa a Responses API para revisar o pedido. "
        "Sem chave, o app oferece um modo local baseado em um template de engenharia de contexto."
    )
    pedido = st.text_area(
        "O que você quer que a IA faça?",
        value="Quero criar um post sobre meu produto.",
        height=120,
        key="otimizador_pedido",
    )
    col1, col2 = st.columns(2)
    with col1:
        contexto = st.text_area("Contexto e dados disponíveis", placeholder="Produto, fatos, documentos, histórico...", height=130)
        publico = st.text_input("Para quem é a resposta?", placeholder="Ex.: clientes iniciantes, equipe técnica...")
        idioma = st.selectbox("Idioma", ["Português do Brasil", "Inglês", "Espanhol"], key="otimizador_idioma")
    with col2:
        formato = st.text_area("Formato esperado", placeholder="Ex.: tabela, roteiro de 5 cenas, JSON...", height=130)
        restricoes = st.text_area("Restrições e critérios de qualidade", placeholder="Ex.: até 150 palavras, sem promessas, com CTA...", height=130)

    if st.button("✨ Melhorar meu prompt", type="primary", use_container_width=True):
        if not pedido.strip():
            st.warning("Escreva pelo menos um pedido para começar.")
        else:
            with st.spinner("Estruturando objetivo, contexto e critérios...", show_time=True):
                try:
                    if get_secret("OPENAI_API_KEY"):
                        resultado = otimizar_com_llm(pedido, contexto, publico, formato, restricoes, idioma)
                    else:
                        resultado = montar_prompt_local(pedido, contexto, publico, formato, restricoes, idioma)
                    st.session_state["ultimo_prompt_otimizado"] = limpar_prompt_gerado(resultado)
                except Exception as exc:
                    st.error(f"Não foi possível chamar o modelo. O modo local continua disponível: {exc}")

    if st.session_state.get("ultimo_prompt_otimizado"):
        st.markdown("### Prompt pronto para copiar")
        st.code(st.session_state["ultimo_prompt_otimizado"], language="markdown")

with aba_checklist:
    st.markdown("## Checklist antes de publicar um prompt")
    itens = [
        "O objetivo está descrito em um verbo observável?",
        "O modelo recebeu contexto suficiente e confiável?",
        "As lacunas estão marcadas em vez de serem preenchidas por invenção?",
        "O formato de saída é explícito e fácil de validar?",
        "As restrições têm limites mensuráveis?",
        "Existe pelo menos um caso de teste que representa o uso real?",
        "Você sabe o que fazer quando a informação não estiver disponível?",
    ]
    for index, item in enumerate(itens):
        st.checkbox(item, key=f"checklist_prompt_{index}")

    st.divider()
    st.markdown("### Regra de ouro")
    st.success("Prompt bom diz o que fazer. Contexto bom diz com quais fatos, limites e critérios fazer.")
