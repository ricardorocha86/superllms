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

from modelos import DATA_ATUALIZACAO, df_modelos


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

    def atualizar_progresso():
        progresso_barra.progress(concluidos / total if total else 0)
        status_slot.info(
            f"{concluidos}/{total} finalizados | {sucessos} sucesso(s) | "
            f"{len(erros_resumidos)} erro(s)"
        )

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
            }
            info = modelo_info.to_dict()
            resultados[pos] = resultado
            erros_resumidos.append(linha_erro_resumida(resultado, info))
            concluidos += 1
            atualizar_progresso()
            continue

        tarefas.append(
            asyncio.create_task(
                testar_modelo_com_posicao(pos, prompt_final, api_key, modelo_info)
            )
        )

    for tarefa in asyncio.as_completed(tarefas):
        pos, modelo_info, resultado = await tarefa
        resultados[pos] = resultado
        concluidos += 1

        if resultado["ok"]:
            sucessos += 1
            with resultados_container:
                exibir_resultado(resultado, modelo_info)
        else:
            erros_resumidos.append(linha_erro_resumida(resultado, modelo_info))

        atualizar_progresso()

    status_slot.success(
        f"Finalizado: {sucessos}/{total} modelo(s) com sucesso, "
        f"{len(erros_resumidos)} erro(s)."
    )
    return resultados, erros_resumidos


def exibir_resultado(resultado, modelo_info):
    empresa = modelo_info["empresa"]
    modelo_nome = modelo_info["modelo_nome"]
    provedor = modelo_info["provedor"]
    if not resultado["ok"]:
        return

    tokens_output = resultado["tokens_output"] or contar_tokens(resultado["texto"])
    st.caption(f"{provedor} | {empresa} — {modelo_nome} · {resultado['tempo']:.2f}s · {tokens_output} tokens")
    st.markdown(resultado["texto"] or "_Resposta vazia._")


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
    for chave, dados in PERSONALIDADES.items():
        selecoes_personalidade[chave] = st.checkbox(
            dados["label"],
            value=dados.get("default", False),
            key=f"personalidade_{chave}",
            disabled=dados.get("disabled", False),
            help=dados["instrucao"],
        )

    tamanho_resposta = st.slider("Tamanho da resposta (palavras)", 10, 200, 80, 10)

    st.divider()
    st.markdown("### Base de modelos")
    st.caption(f"Atualizada em {DATA_ATUALIZACAO}.")
    st.caption("OpenRouter Free vem do endpoint público /api/v1/models.")

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


st.title("Laboratório de Modelos")
st.caption(
    f"Compare {len(df_modelos)} modelos atuais em "
    f"{', '.join(sorted(df_modelos['provedor'].unique()))}."
)

col_prompt, col_configuracoes = st.columns(2, gap="large")
with col_prompt:
    prompt = st.text_area("Digite seu prompt:", height=320)


with col_configuracoes, st.expander("Seleção de modelos", expanded=False):
    df_base = df_modelos.sort_values(
        ["provedor", "status", "empresa", "modelo_nome"], kind="stable"
    ).copy()
    df_base["faixa_preco"] = df_base.apply(classificar_faixa_de_preco, axis=1)
    uids_padrao = df_base.loc[
        df_base["selecionar_padrao"].fillna(False).astype(bool), "uid"
    ].tolist()

    if "uids_modelos_selecionados" not in st.session_state:
        st.session_state["uids_modelos_selecionados"] = uids_padrao
    if "versao_seletor_modelos" not in st.session_state:
        st.session_state["versao_seletor_modelos"] = 0

    col_filtro_empresa, col_filtro_preco = st.columns(2)
    with col_filtro_empresa:
        empresa_escolhida = st.selectbox(
            "Empresa",
            ["Todas as empresas", *sorted(df_base["empresa"].unique())],
            help="Exiba modelos de uma única empresa.",
        )
    with col_filtro_preco:
        preco_escolhido = st.selectbox(
            "Faixa de preço predefinida",
            list(FAIXAS_DE_PRECO),
            help="Considera 1M de tokens de entrada + 1M de saída.",
        )

    df_filtrado = df_base
    if empresa_escolhida != "Todas as empresas":
        df_filtrado = df_filtrado[df_filtrado["empresa"] == empresa_escolhida]
    faixa_escolhida = FAIXAS_DE_PRECO[preco_escolhido]
    if faixa_escolhida:
        df_filtrado = df_filtrado[df_filtrado["faixa_preco"] == faixa_escolhida]

    col_sel_1, col_sel_2, col_sel_3, col_sel_4 = st.columns([1.6, 1.35, 1.15, 2.2])
    with col_sel_1:
        if st.button("Selecionar somente os filtrados", use_container_width=True):
            st.session_state["uids_modelos_selecionados"] = df_filtrado["uid"].tolist()
            st.session_state["versao_seletor_modelos"] += 1
    with col_sel_2:
        if st.button("Adicionar filtrados", use_container_width=True):
            selecionados = set(st.session_state["uids_modelos_selecionados"])
            st.session_state["uids_modelos_selecionados"] = [
                *selecionados,
                *[uid for uid in df_filtrado["uid"] if uid not in selecionados],
            ]
            st.session_state["versao_seletor_modelos"] += 1
    with col_sel_3:
        if st.button("Selecionar padrões", use_container_width=True):
            st.session_state["uids_modelos_selecionados"] = uids_padrao
            st.session_state["versao_seletor_modelos"] += 1
    with col_sel_4:
        if st.button("Limpar seleção", use_container_width=True):
            st.session_state["uids_modelos_selecionados"] = []
            st.session_state["versao_seletor_modelos"] += 1

    st.caption(
        f"Exibindo {len(df_filtrado)} de {len(df_base)} modelos | "
        f"{len(st.session_state['uids_modelos_selecionados'])} selecionado(s)."
    )

    df_filtrado = df_filtrado.copy()
    df_filtrado["Selecionar"] = df_filtrado["uid"].isin(
        st.session_state["uids_modelos_selecionados"]
    )
    df_filtrado["contexto"] = df_filtrado["contexto_tokens"].apply(formatar_contexto)
    df_filtrado["logo_img"] = df_filtrado["logo"].apply(image_to_data_url)

    colunas = [
        "Selecionar",
        "logo_img",
        "provedor",
        "empresa",
        "modelo_nome",
        "modelo_id",
        "status",
        "tier",
        "custo_input_1M",
        "custo_output_1M",
        "faixa_preco",
        "contexto",
        "observacao",
        "uid",
    ]

    df_editor = df_filtrado[colunas]
    df_editado = st.data_editor(
        df_editor,
        key=(
            "model_selector_v20260711_"
            f"{empresa_escolhida}_{preco_escolhido}_"
            f"{st.session_state['versao_seletor_modelos']}"
        ),
        column_config={
            "uid": None,
            "Selecionar": st.column_config.CheckboxColumn(
                "Selecionar",
                help="Marque os modelos que deseja testar.",
                default=False,
            ),
            "logo_img": st.column_config.ImageColumn("Logo", width="small"),
            "provedor": st.column_config.TextColumn("Provedor", disabled=True),
            "empresa": st.column_config.TextColumn("Empresa", disabled=True),
            "modelo_nome": st.column_config.TextColumn("Modelo", disabled=True),
            "modelo_id": st.column_config.TextColumn("Model ID", disabled=True),
            "status": st.column_config.TextColumn("Status", disabled=True),
            "tier": st.column_config.TextColumn("Tier", disabled=True),
            "custo_input_1M": st.column_config.TextColumn("Input/1M", disabled=True),
            "custo_output_1M": st.column_config.TextColumn("Output/1M", disabled=True),
            "faixa_preco": st.column_config.TextColumn("Preço", disabled=True),
            "contexto": st.column_config.TextColumn("Contexto", disabled=True),
            "observacao": st.column_config.TextColumn("Observação", disabled=True),
        },
        hide_index=True,
        width="stretch",
        height=620,
    )


uids_visiveis = set(df_filtrado["uid"])
uids_selecionados = [
    *[
        uid
        for uid in st.session_state["uids_modelos_selecionados"]
        if uid not in uids_visiveis
    ],
    *df_editado.loc[df_editado["Selecionar"], "uid"].tolist(),
]
st.session_state["uids_modelos_selecionados"] = uids_selecionados
ordem = {uid: pos for pos, uid in enumerate(uids_selecionados)}
modelos_selecionados = df_modelos[df_modelos["uid"].isin(uids_selecionados)].copy()
if not modelos_selecionados.empty:
    modelos_selecionados["ordem"] = modelos_selecionados["uid"].map(ordem)
    modelos_selecionados = modelos_selecionados.sort_values("ordem").drop(columns=["ordem"])


with col_configuracoes, st.expander("Preview do prompt", expanded=False):
    if prompt.strip():
        prompt_preview = construir_prompt_final(prompt, tamanho_resposta, selecoes_personalidade)
        st.markdown(f"**Prompt enviado:** ({contar_tokens(prompt_preview)} tokens estimados)")
        st.text_area(
            "Preview do prompt enviado",
            value=prompt_preview,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("Digite um prompt para ver o preview.")


botao_teste = st.button(
    "Testar modelos selecionados", type="primary", use_container_width=True
)

if modelos_selecionados.empty:
    st.warning("Selecione pelo menos um modelo para testar.")
else:
    st.success(f"Modelos Selecionados: {len(modelos_selecionados)}")


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
