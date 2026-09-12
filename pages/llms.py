import asyncio
import base64
import socket
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import tiktoken
from openai import AsyncOpenAI
from personas import PERSONALIDADES, construir_prompt_final

try:
    from anthropic import AsyncAnthropic
except ImportError:  # O app mostra um erro claro se um modelo Anthropic for usado.
    AsyncAnthropic = None

from modelos import DATA_ATUALIZACAO, df_modelos, resumo_openrouter


ROOT_DIR = Path(__file__).resolve().parents[1]
LOGOS_DIR = ROOT_DIR / "logos"


def contar_tokens(texto, modelo="gpt-4o"):
    """Estima tokens de um texto usando tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(modelo)
        return len(encoding.encode(texto or ""))
    except Exception:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(texto or ""))
        except Exception:
            # Fallback offline: evita quebrar o app se o cache do tiktoken ainda
            # não existir e a rede local estiver bloqueada.
            return max(1, int(len(texto or "") / 4))


def preco_para_float(valor):
    try:
        return float(str(valor).replace("$", "").replace(",", ".").strip())
    except Exception:
        return 0.0


FAIXAS_DE_PRECO = {
    "Todos os preços": None,
    "Baratos (até US$ 10)": "Baratos",
    "Intermediários (US$ 10 a 25)": "Intermediários",
    "Caros (acima de US$ 25)": "Caros",
}


def classificar_faixa_de_preco(modelo):
    """Classifica o custo de 1M de tokens de entrada + 1M de saída."""
    custo_total = preco_para_float(modelo["custo_input_1M"]) + preco_para_float(
        modelo["custo_output_1M"]
    )
    if custo_total <= 10:
        return "Baratos"
    if custo_total <= 25:
        return "Intermediários"
    return "Caros"


def formatar_contexto(valor):
    if pd.isna(valor) or valor is None:
        return "N/D"
    valor = int(valor)
    if valor >= 1_000_000:
        return f"{valor / 1_000_000:g}M"
    if valor >= 1_000:
        return f"{valor // 1_000}k"
    return str(valor)


def obter_secret(nome):
    try:
        return st.secrets[nome]
    except Exception:
        return None


def checar_rede_externa():
    try:
        with socket.create_connection(("openrouter.ai", 443), timeout=4):
            return True, ""
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def resolver_logo(nome_arquivo):
    if not nome_arquivo:
        return None

    caminho = LOGOS_DIR / str(nome_arquivo)
    if caminho.exists():
        return caminho

    nome_lower = str(nome_arquivo).lower()
    for candidato in LOGOS_DIR.iterdir():
        if candidato.name.lower() == nome_lower:
            return candidato
    return None


def image_to_data_url(nome_arquivo):
    caminho = resolver_logo(nome_arquivo)
    if not caminho:
        return None
    try:
        encoded = base64.b64encode(caminho.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def formatar_erro_api(exc):
    partes = [f"{exc.__class__.__name__}: {exc}"]
    status_code = getattr(exc, "status_code", None)
    if status_code:
        partes.append(f"status HTTP {status_code}")

    causa = getattr(exc, "__cause__", None)
    if causa:
        partes.append(f"causa: {causa.__class__.__name__}: {causa}")

    body = getattr(exc, "body", None)
    if body:
        partes.append(f"detalhe: {body}")

    return " | ".join(partes)


def extrair_usage(usage, entrada_nomes, saida_nomes):
    if usage is None:
        return None, None

    def primeiro_int(nomes):
        for nome in nomes:
            valor = getattr(usage, nome, None)
            if valor is not None:
                return int(valor)
        return None

    return primeiro_int(entrada_nomes), primeiro_int(saida_nomes)


def extrair_texto_responses(response):
    texto = getattr(response, "output_text", None)
    if texto:
        return texto.strip()

    partes = []
    for item in getattr(response, "output", []) or []:
        for conteudo in getattr(item, "content", []) or []:
            texto_bloco = getattr(conteudo, "text", None)
            if texto_bloco:
                partes.append(texto_bloco)
    return "\n".join(partes).strip()


def extrair_texto_anthropic(response):
    partes = []
    for bloco in getattr(response, "content", []) or []:
        texto = getattr(bloco, "text", None)
        if texto:
            partes.append(texto)
    return "\n".join(partes).strip()


async def testar_modelo(prompt, api_key, modelo_info):
    inicio = time.time()
    api_tipo = modelo_info.get("api_tipo", "chat_completions")
    modelo_id = modelo_info["modelo_id"]

    try:
        if api_tipo == "openai_responses":
            client = AsyncOpenAI(api_key=api_key, timeout=60.0)
            try:
                response = await client.responses.create(model=modelo_id, input=prompt)
            finally:
                await client.close()

            texto = extrair_texto_responses(response)
            tokens_input, tokens_output = extrair_usage(
                getattr(response, "usage", None),
                ["input_tokens", "prompt_tokens"],
                ["output_tokens", "completion_tokens"],
            )

        elif api_tipo == "anthropic_messages":
            if AsyncAnthropic is None:
                raise RuntimeError(
                    "Pacote 'anthropic' não instalado. Rode: pip install -r requirements.txt"
                )

            client = AsyncAnthropic(api_key=api_key, timeout=60.0)
            try:
                response = await client.messages.create(
                    model=modelo_id,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
            finally:
                await client.close()

            texto = extrair_texto_anthropic(response)
            tokens_input, tokens_output = extrair_usage(
                getattr(response, "usage", None),
                ["input_tokens"],
                ["output_tokens"],
            )

        else:
            kwargs = {"api_key": api_key}
            base_url = str(modelo_info.get("base_url") or "").strip()
            if base_url:
                kwargs["base_url"] = base_url

            client = AsyncOpenAI(timeout=60.0, **kwargs)
            try:
                response = await client.chat.completions.create(
                    model=modelo_id,
                    messages=[{"role": "user", "content": prompt}],
                )
            finally:
                await client.close()

            texto = (response.choices[0].message.content or "").strip()
            tokens_input, tokens_output = extrair_usage(
                getattr(response, "usage", None),
                ["prompt_tokens", "input_tokens"],
                ["completion_tokens", "output_tokens"],
            )

        return {
            "ok": True,
            "texto": texto,
            "tempo": time.time() - inicio,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "erro": "",
        }

    except Exception as exc:
        return {
            "ok": False,
            "texto": "",
            "tempo": time.time() - inicio,
            "tokens_input": None,
            "tokens_output": None,
            "erro": formatar_erro_api(exc),
            "codigo_erro": getattr(exc, "status_code", None) or getattr(exc, "code", None) or type(exc).__name__,
            "mensagem_erro": getattr(exc, "message", None) or str(exc),
        }


def executar_corrotina(corrotina):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(corrotina)
    finally:
        loop.close()


def resumir_erro_api(mensagem):
    mensagem = " ".join(str(mensagem or "").split())
    mensagem_lower = mensagem.lower()

    if "ratelimiterror" in mensagem_lower or "429" in mensagem_lower:
        return "Rate limit do provedor. Tente novamente em instantes ou use uma chave própria nesse provedor."
    if "authenticationerror" in mensagem_lower or "401" in mensagem_lower:
        return "Falha de autenticação. Verifique a chave desse provedor."
    if "permissiondenied" in mensagem_lower or "403" in mensagem_lower:
        return "Sem permissão para esse modelo/provedor com a chave atual."
    if "notfounderror" in mensagem_lower or "404" in mensagem_lower:
        return "Modelo indisponível ou ID não aceito pelo provedor."
    if "apiconnectionerror" in mensagem_lower or "connecterror" in mensagem_lower:
        return "Falha de conexão com o provedor."
    if "timeout" in mensagem_lower or "timed out" in mensagem_lower:
        return "Timeout: o provedor demorou demais para responder."
    if "chave" in mensagem_lower and "não encontrada" in mensagem_lower:
        return "Chave ausente em st.secrets."

    return mensagem[:220] + ("..." if len(mensagem) > 220 else "")


def linha_erro_resumida(resultado, modelo_info):
    return {
        "Provedor": modelo_info["provedor"],
        "Empresa": modelo_info["empresa"],
        "Modelo": modelo_info["modelo_nome"],
        "ID": modelo_info["modelo_id"],
        "Erro": resumir_erro_api(resultado["erro"]),
        "Tempo": f"{resultado['tempo']:.2f}s",
        "Detalhe técnico": resultado["erro"],
    }


async def testar_modelo_com_posicao(pos, prompt, api_key, modelo_info):
    info = modelo_info.to_dict() if hasattr(modelo_info, "to_dict") else dict(modelo_info)
    resultado = await testar_modelo(prompt, api_key, info)
    return pos, info, resultado


async def executar_modelos_progressivo(
    modelos_selecionados,
    prompt_final,
    resultados_container,
    status_slot,
    progresso_barra,
):
    total = len(modelos_selecionados)
    concluidos = 0
    sucessos = 0
    resultados = [None] * total
    erros_resumidos = []
    tarefas = []
    em_andamento = {}
    with resultados_container:
        colunas = st.columns(3, gap="medium")
    alturas_estimadas = [0, 0, 0]

    def publicar_resultado(resultado, info):
        # Aproxima a altura por linhas e quebra de texto; não reserva espaços.
        coluna = min(range(3), key=lambda indice: alturas_estimadas[indice])
        with colunas[coluna]:
            exibir_resultado(resultado, info)
        texto = str(resultado.get("texto") or resultado.get("erro") or "")
        if not resultado["ok"]:
            texto = resumir_erro_api(texto)[:150]
        linhas = sum(max(1, (len(linha) + 44) // 45) for linha in texto.splitlines())
        alturas_estimadas[coluna] += 6 + linhas

    def atualizar_progresso():
        progresso_barra.progress(concluidos / total if total else 0)
        resumo = (
            f"{concluidos}/{total} finalizados · {sucessos} sucesso(s) · "
            f"{len(erros_resumidos)} erro(s)"
        )
        agora = time.perf_counter()
        pendentes = [
            f"⏳ Gerando **{nome}** · {int(agora - inicio)}s"
            for nome, inicio in em_andamento.values()
        ]
        status_slot.info(resumo + ("\n\n" + "  \n".join(pendentes) if pendentes else ""))

    atualizar_progresso()

    for pos, (_, modelo_info) in enumerate(modelos_selecionados.iterrows()):
        secret_name = modelo_info["api_key_secret"]
        api_key = obter_secret(secret_name)
        if not api_key:
            resultado = {
                "ok": False,
                "texto": "",
                "tempo": 0.0,
                "tokens_input": None,
                "tokens_output": None,
                "erro": f"Chave {secret_name} não encontrada em st.secrets.",
                "codigo_erro": "CHAVE_AUSENTE",
            }
            info = modelo_info.to_dict()
            resultados[pos] = resultado
            publicar_resultado(resultado, info)
            erros_resumidos.append(linha_erro_resumida(resultado, info))
            concluidos += 1
            atualizar_progresso()
            continue

        em_andamento[pos] = (modelo_info["modelo_nome"], time.perf_counter())
        tarefas.append(
            asyncio.create_task(
                testar_modelo_com_posicao(pos, prompt_final, api_key, modelo_info)
            )
        )

    atualizar_progresso()
    for tarefa in asyncio.as_completed(tarefas):
        proxima_resposta = asyncio.ensure_future(tarefa)
        while not proxima_resposta.done():
            await asyncio.wait({proxima_resposta}, timeout=1.0)
            atualizar_progresso()
        pos, modelo_info, resultado = await proxima_resposta
        em_andamento.pop(pos, None)
        resultados[pos] = resultado
        concluidos += 1

        if resultado["ok"]:
            sucessos += 1
        else:
            erros_resumidos.append(linha_erro_resumida(resultado, modelo_info))

        publicar_resultado(resultado, modelo_info)
        atualizar_progresso()

    status_slot.caption(
        f"Finalizado: {sucessos}/{total} modelo(s) com sucesso, "
        f"{len(erros_resumidos)} erro(s)."
    )
    return resultados, erros_resumidos


def exibir_resultado(resultado, modelo_info):
    empresa = modelo_info["empresa"]
    modelo_nome = modelo_info["modelo_nome"]
    with st.container(border=True, key=f"resposta_modelo_{modelo_info['uid']}"):
        st.markdown(f"**{empresa} · {modelo_nome}**")
        if not resultado["ok"]:
            codigo = resultado.get("codigo_erro") or "ERRO"
            mensagem = " ".join(str(resultado.get("mensagem_erro") or resultado["erro"]).split())
            mensagem_pt = resumir_erro_api(mensagem)
            if mensagem_pt == mensagem[:220] + ("..." if len(mensagem) > 220 else ""):
                texto_erro = mensagem.lower()
                if "402" in str(codigo) or "credit" in texto_erro or "balance" in texto_erro:
                    mensagem_pt = "Saldo ou créditos insuficientes para executar este modelo."
                elif "400" in str(codigo) or "unsupported" in texto_erro or "invalid" in texto_erro:
                    mensagem_pt = "O provedor não aceitou um dos parâmetros enviados para este modelo."
                elif str(codigo).startswith("5"):
                    mensagem_pt = "O provedor apresentou uma falha interna. Tente novamente em instantes."
                else:
                    mensagem_pt = "Não foi possível concluir a solicitação ao provedor."
            st.error(f"Deu erro. Código: {codigo}. Mensagem: {mensagem_pt}")
            return
        texto_resposta = resultado["texto"] or "_Resposta vazia._"
        st.markdown("\n".join(f"> {linha}" for linha in texto_resposta.splitlines()))
        tokens_output = resultado["tokens_output"] or contar_tokens(resultado["texto"])
        st.caption(f"{resultado['tempo']:.2f}s · {tokens_output} tokens de saída")


def calcular_relatorio_custos(resultados, modelos_selecionados, prompt_final):
    relatorio = []
    tokens_input_estimados = contar_tokens(prompt_final)

    for resultado, (_, modelo_info) in zip(resultados, modelos_selecionados.iterrows()):
        if resultado["ok"]:
            tokens_input = resultado["tokens_input"] or tokens_input_estimados
            tokens_output = resultado["tokens_output"] or contar_tokens(resultado["texto"])
            custo_input = preco_para_float(modelo_info["custo_input_1M"])
            custo_output = preco_para_float(modelo_info["custo_output_1M"])
            custo_input_total = (tokens_input / 1_000_000) * custo_input
            custo_output_total = (tokens_output / 1_000_000) * custo_output
            custo_total = custo_input_total + custo_output_total
            status = "Sucesso"
            creditos = modelo_info["creditos"]
        else:
            tokens_input = tokens_input_estimados
            tokens_output = 0
            custo_input_total = 0
            custo_output_total = 0
            custo_total = 0
            status = "Erro"
            creditos = 0

        relatorio.append(
            {
                "Provedor": modelo_info["provedor"],
                "Empresa": modelo_info["empresa"],
                "Modelo": modelo_info["modelo_nome"],
                "ID": modelo_info["modelo_id"],
                "Status": status,
                "Tokens Input": tokens_input,
                "Tokens Output": tokens_output,
                "Custo Input": custo_input_total,
                "Custo Output": custo_output_total,
                "Custo Total": custo_total,
                "Tempo": resultado["tempo"],
                "Créditos": creditos,
            }
        )

    return relatorio


with st.sidebar:
    st.markdown("### Configurações de personalidade")
    selecoes_personalidade = {}
    colunas_personalidade = st.columns(2)
    for indice, (chave, dados) in enumerate(PERSONALIDADES.items()):
        selecoes_personalidade[chave] = colunas_personalidade[indice % 2].checkbox(
            dados["label"],
            value=dados.get("default", False),
            key=f"personalidade_{chave}",
            disabled=dados.get("disabled", False),
        )

    tamanho_resposta = st.slider("Tamanho da resposta (palavras)", 10, 100, 80, 10)

    st.divider()
    st.markdown("### Base de modelos")
    st.caption(f"Atualizada em {DATA_ATUALIZACAO}.")
    st.caption(resumo_openrouter())

    with st.expander("Chaves esperadas"):
        st.code(
            "\n".join(
                sorted(
                    secret
                    for secret in df_modelos["api_key_secret"].dropna().unique()
                    if secret
                )
            ),
            language="text",
        )


st.html("""<style>
[class*="st-key-resposta_modelo_"] {
    background-color: color-mix(in srgb, var(--text-color, #31333f) 4%, var(--background-color, #fff));
    border-radius: 10px;
}
[class*="st-key-resposta_modelo_"] blockquote {
    color: inherit !important;
    opacity: 1;
    font-style: normal;
    border-left: 3px solid #0068c9;
    padding: 0 0 0 0.75rem;
    margin: 0.5rem 0;
    background: transparent;
}
[class*="st-key-resposta_modelo_"] blockquote p {
    color: inherit !important;
    opacity: 1;
}
</style>""")

st.title("Laboratório de Modelos")
st.caption(
    f"Compare {len(df_modelos)} modelos atuais em "
    f"{', '.join(sorted(df_modelos['provedor'].unique()))}."
)

col_prompt, col_configuracoes = st.columns(2, gap="large")
with col_prompt:
    prompt = st.text_area("Digite seu prompt:", height=140)
    prompt_preview = construir_prompt_final(prompt, tamanho_resposta, selecoes_personalidade)
    st.markdown("**Prompt enviado**")
    st.code(prompt_preview, language="text", wrap_lines=True)
    st.caption(f"{contar_tokens(prompt_preview)} tokens estimados")

with col_configuracoes:
    st.subheader("Seleção de modelos")
    df_base = df_modelos.sort_values(
        ["empresa", "modelo_nome", "provedor"], kind="stable"
    ).drop_duplicates("uid").copy()
    opcoes = df_base["uid"].tolist()
    # Um representante por fabricante, priorizando a categoria do catálogo.
    ranking = df_base.assign(
        prioridade=df_base["tier"].map({"Elite": 5, "Pro": 4, "Preview": 3, "Básico": 2, "Free": 1}).fillna(0),
        direto=~df_base["provedor"].str.contains("OpenRouter", case=False),
    ).sort_values(["prioridade", "direto"], ascending=[False, False], kind="stable")
    uids_padrao = ranking.drop_duplicates("empresa")["uid"].tolist()
    rotulos = {
        row["uid"]: f"{row['modelo_nome']} · {row['provedor']} · {row['modelo_id']}"
        for _, row in df_base.iterrows()
    }
    chave_seletor = "seletor_modelos_fabricantes"
    if chave_seletor not in st.session_state:
        st.session_state[chave_seletor] = [
            uid for uid in uids_padrao
            if uid in rotulos
        ]
    else:
        st.session_state[chave_seletor] = [
            uid for uid in st.session_state[chave_seletor] if uid in rotulos
        ]

    def restaurar_selecao(uids):
        st.session_state[chave_seletor] = uids
        st.session_state["uids_modelos_selecionados"] = uids.copy()

    uids_selecionados = st.multiselect(
        "Modelos para comparar",
        opcoes,
        format_func=lambda uid: rotulos[uid],
        key=chave_seletor,
        placeholder="Busque pelo nome do modelo ou provedor",
        help="Digite para buscar e clique para adicionar. Use o × ao lado do nome para remover.",
    )
    padroes, limpar = st.columns(2)
    padroes.button("Um por fabricante", on_click=restaurar_selecao, args=(uids_padrao,), width="stretch", help="Prioriza Elite e Pro do catálogo; essa classificação não é um ranking de desempenho.")
    limpar.button("Limpar seleção", on_click=restaurar_selecao, args=([],), width="stretch")
    def alterar_fabricante(empresa, adicionar):
        atuais = st.session_state[chave_seletor]
        grupo = df_base.loc[df_base["empresa"] == empresa, "uid"].tolist()
        restaurar_selecao(list(dict.fromkeys(atuais + grupo)) if adicionar else [uid for uid in atuais if uid not in grupo])

    with st.expander("Selecionar por fabricante", expanded=False):
        fabricante = st.selectbox("Fabricante", sorted(df_base["empresa"].unique()))
        adicionar, remover = st.columns(2)
        adicionar.button(f"Adicionar todos · {fabricante}", on_click=alterar_fabricante, args=(fabricante, True), width="stretch")
        remover.button(f"Remover todos · {fabricante}", on_click=alterar_fabricante, args=(fabricante, False), width="stretch")

st.session_state["uids_modelos_selecionados"] = uids_selecionados.copy()
ordem = {uid: pos for pos, uid in enumerate(uids_selecionados)}
modelos_selecionados = df_modelos[df_modelos["uid"].isin(uids_selecionados)].copy()
if not modelos_selecionados.empty:
    modelos_selecionados["ordem"] = modelos_selecionados["uid"].map(ordem)
    modelos_selecionados = modelos_selecionados.sort_values("ordem").drop(columns=["ordem"])


col_contagem, col_testar = st.columns([1, 3], vertical_alignment="center")
with col_contagem:
    st.info(f"{len(modelos_selecionados)} modelo(s) selecionado(s)")
with col_testar:
    botao_teste = st.button(
        "Testar modelos selecionados", type="primary", width="stretch",
        disabled=modelos_selecionados.empty,
    )


if botao_teste:
    if not prompt.strip():
        st.warning("Digite um prompt primeiro.")
    elif modelos_selecionados.empty:
        st.warning("Selecione pelo menos um modelo.")
    else:
        rede_ok, erro_rede = checar_rede_externa()
        if not rede_ok:
            st.warning(
                "Não foi possível verificar a conectividade com openrouter.ai. "
                "Os testes continuarão, pois os outros provedores podem estar acessíveis. "
                f"Detalhe: {erro_rede}"
            )

        prompt_final = construir_prompt_final(prompt, tamanho_resposta, selecoes_personalidade)

        st.markdown("### Resultados concluídos")
        status_slot = st.empty()
        progresso_barra = st.progress(0)
        resultados_container = st.container()

        with st.spinner("Aguardando respostas dos modelos selecionados...", show_time=True):
            resultados, erros_resumidos = executar_corrotina(
                executar_modelos_progressivo(
                    modelos_selecionados,
                    prompt_final,
                    resultados_container,
                    status_slot,
                    progresso_barra,
                )
            )

        st.markdown("---")
        st.markdown("### Resumo dos erros")
        if erros_resumidos:
            df_erros = pd.DataFrame(erros_resumidos)
            st.dataframe(
                df_erros[
                    ["Provedor", "Empresa", "Modelo", "ID", "Erro", "Tempo"]
                ],
                hide_index=True,
                width="stretch",
            )
            with st.expander("Detalhes técnicos dos erros", expanded=False):
                st.dataframe(
                    df_erros[
                        [
                            "Provedor",
                            "Empresa",
                            "Modelo",
                            "ID",
                            "Detalhe técnico",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.success("Nenhum erro nos modelos selecionados.")

        st.markdown("---")
        st.markdown("### Relatório de custos")
        relatorio = calcular_relatorio_custos(resultados, modelos_selecionados, prompt_final)
        df_relatorio = pd.DataFrame(relatorio)

        if not df_relatorio.empty:
            modelos_sucesso = df_relatorio[df_relatorio["Status"] == "Sucesso"].copy()
            modelos_erro = df_relatorio[df_relatorio["Status"] == "Erro"].copy()

            custo_total = float(df_relatorio["Custo Total"].sum())
            creditos = int(df_relatorio["Créditos"].sum())

            df_exibicao = df_relatorio.copy()
            for coluna in ["Custo Input", "Custo Output", "Custo Total"]:
                df_exibicao[coluna] = df_exibicao[coluna].apply(lambda x: f"${x:.6f}")
            df_exibicao["Tempo"] = df_exibicao["Tempo"].apply(lambda x: f"{x:.2f}s")
            df_exibicao["ordem_status"] = df_relatorio["Status"].map(
                {"Sucesso": 0, "Erro": 1}
            ).fillna(2)
            df_exibicao = df_exibicao.sort_values(
                ["ordem_status", "Provedor", "Empresa", "Modelo"], kind="stable"
            ).drop(columns=["ordem_status"])

            st.dataframe(df_exibicao, hide_index=True, width="stretch")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sucesso", f"{len(modelos_sucesso)}/{len(df_relatorio)}")
                st.metric("Créditos", creditos)
            with col2:
                st.metric("Custo total", f"${custo_total:.6f}")
                if not modelos_sucesso.empty:
                    st.metric(
                        "Tempo médio",
                        f"{modelos_sucesso['Tempo'].mean():.2f}s",
                    )
            with col3:
                if not modelos_sucesso.empty:
                    mais_barato = modelos_sucesso.sort_values("Custo Total").iloc[0]
                    mais_rapido = modelos_sucesso.sort_values("Tempo").iloc[0]
                    st.metric("Mais barato", mais_barato["Modelo"])
                    st.metric("Mais rápido", mais_rapido["Modelo"])

            if not modelos_erro.empty:
                st.caption(
                    f"Erros: {len(modelos_erro)}/{len(df_relatorio)} modelos. "
                    "Veja o resumo compacto acima."
                )
