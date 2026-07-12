"""Playground visual para a API Images com GPT Image 2."""

import base64
import time

import streamlit as st
from openai import OpenAI


MODELO = "gpt-image-2"
DOCS_URL = "https://developers.openai.com/api/docs/guides/image-generation"
MODELO_URL = "https://developers.openai.com/api/docs/models/gpt-image-2"
CALCULADORA_URL = "https://developers.openai.com/api/docs/guides/image-generation#calculating-costs"
CUSTO_PARCIAL_USD = 0.003  # 100 image output tokens a US$ 30 / 1M tokens.
PRECO_TEXTO_INPUT_1M = 5.00
PRECO_IMAGEM_INPUT_1M = 8.00
PRECO_IMAGEM_OUTPUT_1M = 30.00

# Valores de saída por imagem da tabela oficial em 2026-07-11. Eles não incluem
# tokens do prompt nem das imagens de referência em edições.
PRECOS_SAIDA = {
    "1024x1024": {"low": 0.006, "medium": 0.053, "high": 0.211},
    "1024x1536": {"low": 0.005, "medium": 0.041, "high": 0.165},
    "1536x1024": {"low": 0.005, "medium": 0.041, "high": 0.165},
}

TAMANHOS_POPULARES = {
    "Auto (o modelo decide)": "auto",
    "Quadrado — 1024 × 1024": "1024x1024",
    "Retrato — 1024 × 1536": "1024x1536",
    "Paisagem — 1536 × 1024": "1536x1024",
    "2K quadrado — 2048 × 2048": "2048x2048",
    "2K paisagem — 2048 × 1152": "2048x1152",
    "4K paisagem — 3840 × 2160": "3840x2160",
    "4K retrato — 2160 × 3840": "2160x3840",
    "Personalizado": "customizado",
}


def obter_chave_openai():
    """Prefere a chave temporária da sessão e depois o secrets.toml."""
    chave_temporaria = st.session_state.get("imagem_openai_api_key", "").strip()
    if chave_temporaria:
        return chave_temporaria
    try:
        return str(st.secrets["OPENAI_API_KEY"])
    except (KeyError, FileNotFoundError):
        return ""


def tamanho_valido(largura, altura):
    if largura > 3840 or altura > 3840:
        return False, "Cada lado deve ter no máximo 3840 px."
    if largura % 16 or altura % 16:
        return False, "Largura e altura devem ser múltiplos de 16 px."
    if max(largura, altura) / min(largura, altura) > 3:
        return False, "A proporção entre os lados não pode ultrapassar 3:1."
    pixels = largura * altura
    if not 655_360 <= pixels <= 8_294_400:
        return False, "A imagem deve ter entre 655.360 e 8.294.400 pixels."
    return True, ""


def estimar_custo_saida(tamanho, qualidade, quantidade):
    if tamanho in PRECOS_SAIDA and qualidade in {"low", "medium", "high"}:
        unitario = PRECOS_SAIDA[tamanho][qualidade]
        return unitario, unitario * quantidade
    return None, None


def calcular_custo_por_tokens(uso, quantidade_parciais):
    """Estima o custo padrão a partir do uso retornado pelo evento final."""
    if not uso or not any(uso.values()):
        return None

    texto = uso.get("texto_input", 0)
    imagens = uso.get("imagem_input", 0)
    saida = uso.get("imagem_output", 0)
    return (
        texto * PRECO_TEXTO_INPUT_1M / 1_000_000
        + imagens * PRECO_IMAGEM_INPUT_1M / 1_000_000
        + saida * PRECO_IMAGEM_OUTPUT_1M / 1_000_000
        + quantidade_parciais * CUSTO_PARCIAL_USD
    )


def arquivos_para_api(arquivos):
    return [
        (arquivo.name, arquivo.getvalue(), arquivo.type or "application/octet-stream")
        for arquivo in arquivos
    ]


def extensao_e_mime(formato):
    return {
        "png": ("png", "image/png"),
        "jpeg": ("jpg", "image/jpeg"),
        "webp": ("webp", "image/webp"),
    }[formato]


def resumo_erro_api(exc):
    partes = [str(exc)]
    codigo = getattr(exc, "code", None)
    request_id = getattr(exc, "request_id", None)
    causa = getattr(exc, "__cause__", None)
    if codigo:
        partes.append(f"código: {codigo}")
    if request_id:
        partes.append(f"request ID: {request_id}")
    if causa:
        partes.append(f"causa: {causa}")
    return " | ".join(partes)


st.set_page_config(page_title="Playground GPT Image 2", page_icon="🎨", layout="wide")

st.title("🎨 Playground GPT Image 2")
st.caption("Gere, edite e compare imagens usando a API oficial da OpenAI.")

with st.sidebar:
    st.header("Conexão")
    st.text_input(
        "Chave temporária da OpenAI (opcional)",
        type="password",
        key="imagem_openai_api_key",
        help="Se preenchida, tem prioridade sobre OPENAI_API_KEY em st.secrets e vale apenas nesta sessão.",
    )
    if obter_chave_openai():
        st.success("Chave da API disponível")
    else:
        st.warning("Configure OPENAI_API_KEY em .streamlit/secrets.toml ou informe uma chave temporária.")

    if st.button("Testar conexão com a OpenAI", use_container_width=True):
        if not obter_chave_openai():
            st.error("Não há uma chave configurada para testar.")
        else:
            try:
                client = OpenAI(api_key=obter_chave_openai(), timeout=20.0)
                try:
                    modelos = client.models.list()
                finally:
                    client.close()
                st.success(f"Conexão e autenticação OK ({len(modelos.data)} modelos recebidos).")
            except Exception as exc:
                st.error("Não foi possível conectar à API.")
                st.code(resumo_erro_api(exc), language="text")

    st.divider()
    st.caption("Modelo: `gpt-image-2`")
    st.caption("A API pode exigir verificação da organização antes do primeiro uso.")

aba_playground, aba_custos, aba_referencias = st.tabs(
    ["Playground", "Custos", "Guia rápido"]
)

with aba_playground:
    esquerda, direita = st.columns([1.1, 0.9], gap="large")

    with esquerda:
        modo = st.radio(
            "Modo",
            ["Gerar do zero", "Editar / usar referências"],
            horizontal=True,
            help="Edição aceita uma ou mais imagens de referência; a máscara é opcional.",
        )
        prompt = st.text_area(
            "Prompt",
            height=180,
            placeholder=(
                "Ex.: Fotografia editorial de um tênis futurista sobre uma base de mármore, "
                "luz lateral suave, fundo azul petróleo, sem texto."
            ),
        )

        imagens_referencia = []
        mascara = None
        if modo == "Editar / usar referências":
            imagens_referencia = st.file_uploader(
                "Imagens de referência",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                help="Envie uma ou mais imagens. O GPT Image 2 preserva os detalhes das entradas em alta fidelidade.",
            )
            mascara = st.file_uploader(
                "Máscara (opcional, PNG com transparência)",
                type=["png"],
                help="A máscara se aplica à primeira imagem. Ela deve ter o mesmo tamanho e formato da imagem a editar.",
            )
            if imagens_referencia:
                st.caption(f"{len(imagens_referencia)} imagem(ns) de referência anexada(s).")
            if mascara:
                st.caption("Máscara anexada. A área mascarada guia a região a ser alterada.")

    with direita:
        st.subheader("Configurações de saída")
        tamanho_rotulo = st.selectbox("Tamanho", list(TAMANHOS_POPULARES))
        tamanho = TAMANHOS_POPULARES[tamanho_rotulo]
        if tamanho == "customizado":
            col_largura, col_altura = st.columns(2)
            with col_largura:
                largura = st.number_input("Largura", min_value=16, max_value=3840, value=1024, step=16)
            with col_altura:
                altura = st.number_input("Altura", min_value=16, max_value=3840, value=1024, step=16)
            valido, motivo = tamanho_valido(largura, altura)
            if valido:
                tamanho = f"{largura}x{altura}"
                st.success(f"Tamanho válido: {tamanho}")
            else:
                tamanho = None
                st.error(motivo)

        qualidade = st.select_slider(
            "Qualidade", options=["auto", "low", "medium", "high"], value="low"
        )
        formato = st.selectbox("Formato", ["png", "jpeg", "webp"])
        compressao = None
        if formato in {"jpeg", "webp"}:
            compressao = st.slider("Compressão", 0, 100, 85)
            st.caption("JPEG tende a ser mais rápido que PNG.")
        background = st.selectbox("Fundo", ["auto", "opaque"])
        moderacao = st.selectbox("Moderação", ["auto", "low"], help="`auto` é o padrão recomendado.")
        quantidade = st.number_input("Quantidade", min_value=1, max_value=4, value=1, step=1)
        parciais_solicitadas = st.select_slider(
            "Pré-visualizações parciais",
            options=[0, 1, 2, 3],
            value=0,
            format_func=lambda valor: (
                "Desligadas (sem custo adicional)"
                if valor == 0
                else f"{valor} prévia(s)"
            ),
            help="Ativa streaming. A API pode retornar menos parciais se a imagem final ficar pronta rapidamente.",
        )

        unitario, total_saida = estimar_custo_saida(tamanho, qualidade, quantidade)
        if total_saida is not None:
            st.metric("Estimativa de saída", f"US$ {total_saida:.3f}", f"US$ {unitario:.3f} por imagem")
            st.caption("Exclui tokens do prompt e, em edições, tokens das imagens de entrada.")
        else:
            st.info("A estimativa exata para `auto` ou tamanhos flexíveis está na calculadora oficial.")
        if parciais_solicitadas:
            custo_parciais_maximo = CUSTO_PARCIAL_USD * parciais_solicitadas * quantidade
            st.caption(
                f"Pré-visualizações: até US$ {custo_parciais_maximo:.3f} adicionais "
                f"({parciais_solicitadas} × US$ {CUSTO_PARCIAL_USD:.3f} por imagem)."
            )

    if modo == "Editar / usar referências" and imagens_referencia:
        with st.expander("Prévia das referências", expanded=False):
            colunas = st.columns(min(4, len(imagens_referencia)))
            for indice, arquivo in enumerate(imagens_referencia):
                with colunas[indice % len(colunas)]:
                    st.image(arquivo, caption=arquivo.name, use_container_width=True)
            if mascara:
                st.image(mascara, caption="Máscara", width=220)

    with st.expander("Payload que será enviado", expanded=False):
        args = {
            "model": MODELO,
            "prompt": prompt or "<seu prompt>",
            "size": tamanho or "<tamanho inválido>",
            "quality": qualidade,
            "output_format": formato,
            "background": background,
            "moderation": moderacao,
            "n": int(quantidade),
        }
        args["stream"] = True
        args["partial_images"] = parciais_solicitadas
        if compressao is not None:
            args["output_compression"] = compressao
        if modo == "Editar / usar referências":
            args["endpoint"] = "/v1/images/edits"
            args["imagens"] = len(imagens_referencia)
            args["máscara"] = bool(mascara)
        else:
            args["endpoint"] = "/v1/images/generations"
        st.json(args)

    acao = "Editar imagem" if modo == "Editar / usar referências" else "Gerar imagem"
    if st.button(acao, type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Descreva o que você quer criar ou editar no prompt.")
        elif not tamanho:
            st.warning("Corrija o tamanho personalizado antes de enviar.")
        elif modo == "Editar / usar referências" and not imagens_referencia:
            st.warning("Envie ao menos uma imagem de referência para o modo de edição.")
        elif not obter_chave_openai():
            st.error("Não encontrei OPENAI_API_KEY. Configure a chave para fazer a solicitação.")
        else:
            parametros = {
                "model": MODELO,
                "prompt": prompt.strip(),
                "size": tamanho,
                "quality": qualidade,
                "output_format": formato,
                "background": background,
                "moderation": moderacao,
                "n": int(quantidade),
            }
            parametros["stream"] = True
            parametros["partial_images"] = parciais_solicitadas
            if compressao is not None:
                parametros["output_compression"] = compressao

            inicio = time.perf_counter()
            parciais = []
            imagens = []
            uso_tokens = {"texto_input": 0, "imagem_input": 0, "imagem_output": 0}
            previa_slot = st.empty()
            try:
                with st.spinner("A API está renderizando a imagem…", show_time=True):
                    client = OpenAI(api_key=obter_chave_openai(), timeout=180.0)
                    try:
                        if modo == "Editar / usar referências":
                            # A assinatura atual de images.edit ainda não declara
                            # ``moderation``, embora a API o aceite. extra_body
                            # preserva o campo com segurança entre versões do SDK.
                            parametros_edicao = parametros.copy()
                            parametros_edicao.pop("moderation")
                            parametros_edicao["image"] = arquivos_para_api(imagens_referencia)
                            parametros_edicao["extra_body"] = {
                                "moderation": moderacao
                            }
                            if mascara:
                                parametros_edicao["mask"] = arquivos_para_api([mascara])[0]
                            resposta = client.images.edit(**parametros_edicao)
                        else:
                            resposta = client.images.generate(**parametros)

                        for evento in resposta:
                            tipo_evento = getattr(evento, "type", "")
                            if tipo_evento.endswith(".partial_image"):
                                parcial = {
                                    "bytes": base64.b64decode(evento.b64_json),
                                    "indice": getattr(evento, "partial_image_index", len(parciais)),
                                }
                                parciais.append(parcial)
                                previa_slot.image(
                                    parcial["bytes"],
                                    caption=f"Prévia parcial {parcial['indice'] + 1}",
                                    width=260,
                                )
                            elif tipo_evento.endswith(".completed"):
                                imagens.append(
                                    {
                                        "bytes": base64.b64decode(evento.b64_json),
                                        "revised_prompt": None,
                                    }
                                )
                                usage = getattr(evento, "usage", None)
                                detalhes_input = getattr(usage, "input_tokens_details", None)
                                uso_tokens["texto_input"] += int(
                                    getattr(detalhes_input, "text_tokens", 0) or 0
                                )
                                uso_tokens["imagem_input"] += int(
                                    getattr(detalhes_input, "image_tokens", 0) or 0
                                )
                                uso_tokens["imagem_output"] += int(
                                    getattr(usage, "output_tokens", 0) or 0
                                )
                    finally:
                        client.close()

                st.session_state["imagem_resultados"] = {
                    "imagens": imagens,
                    "parciais": parciais,
                    "formato": formato,
                    "tempo": time.perf_counter() - inicio,
                    "tamanho": tamanho,
                    "qualidade": qualidade,
                    "modo": modo,
                    "custo_saida": total_saida,
                    "custo_parciais": CUSTO_PARCIAL_USD * len(parciais),
                    "uso_tokens": uso_tokens,
                    "custo_tokens": calcular_custo_por_tokens(uso_tokens, len(parciais)),
                }
                st.success(f"Concluído em {time.perf_counter() - inicio:.1f}s.")
            except Exception as exc:
                st.error("A solicitação não foi concluída.")
                st.code(resumo_erro_api(exc), language="text")

    resultado = st.session_state.get("imagem_resultados")
    if resultado:
        st.divider()
        st.subheader("Resultado mais recente")
        meta_1, meta_2, meta_3, meta_4, meta_5 = st.columns(5)
        meta_1.metric("Imagens", len(resultado["imagens"]))
        meta_2.metric("Prévias recebidas", len(resultado.get("parciais", [])))
        meta_3.metric("Tempo", f"{resultado['tempo']:.1f}s")
        meta_4.metric("Saída", f"{resultado['tamanho']} · {resultado['qualidade']}")
        meta_5.metric(
            "Custo estimado pela API",
            (
                f"US$ {resultado['custo_tokens']:.4f}"
                if resultado.get("custo_tokens") is not None
                else (
                    f"US$ {resultado['custo_saida'] + resultado.get('custo_parciais', 0):.3f}"
                    if resultado["custo_saida"] is not None
                    else "Consulte a calculadora"
                )
            ),
        )

        uso_tokens = resultado.get("uso_tokens") or {}
        if any(uso_tokens.values()):
            st.caption(
                "Tokens retornados pela API — "
                f"texto de entrada: {uso_tokens['texto_input']:,} · "
                f"imagens de entrada: {uso_tokens['imagem_input']:,} · "
                f"imagem final: {uso_tokens['imagem_output']:,}."
            )

        if resultado.get("parciais"):
            st.markdown("#### Pré-visualizações parciais")
            st.caption(
                f"{len(resultado['parciais'])} recebida(s) · custo adicional: "
                f"US$ {resultado.get('custo_parciais', 0):.3f}."
            )
            colunas_parciais = st.columns(min(3, len(resultado["parciais"])))
            for indice, parcial in enumerate(resultado["parciais"]):
                with colunas_parciais[indice % len(colunas_parciais)]:
                    st.image(
                        parcial["bytes"],
                        caption=f"Prévia {parcial['indice'] + 1}",
                        use_container_width=True,
                    )

        extensao, mime = extensao_e_mime(resultado["formato"])
        for indice, imagem in enumerate(resultado["imagens"], start=1):
            col_imagem, col_info = st.columns([2, 1])
            with col_imagem:
                st.image(imagem["bytes"], caption=f"Imagem {indice}", use_container_width=True)
            with col_info:
                st.download_button(
                    f"Baixar imagem {indice}",
                    data=imagem["bytes"],
                    file_name=f"gpt-image-2-{indice}.{extensao}",
                    mime=mime,
                    use_container_width=True,
                )
                if imagem["revised_prompt"]:
                    st.caption("Prompt revisado pela API")
                    st.write(imagem["revised_prompt"])

with aba_custos:
    st.subheader("Custos de saída do GPT Image 2")
    st.caption("Por imagem; tabela oficial para os tamanhos abaixo. A entrada é cobrada à parte.")
    linhas = []
    for tamanho, precos in PRECOS_SAIDA.items():
        linhas.append(
            {
                "Tamanho": tamanho,
                "Low": f"US$ {precos['low']:.3f}",
                "Medium": f"US$ {precos['medium']:.3f}",
                "High": f"US$ {precos['high']:.3f}",
            }
        )
    st.table(linhas)
    st.info(
        "Para rascunhos e iterações, use `low`. `medium` equilibra qualidade e custo; "
        "`high` é indicado para o arquivo final. Tamanhos quadrados tendem a renderizar mais rápido."
    )
    st.markdown(
        f"Para resoluções flexíveis, `auto` e custos de imagens de referência, consulte a "
        f"[seção oficial de cálculo de custos]({CALCULADORA_URL}). Em edições, cada imagem de entrada "
        "é processada em alta fidelidade pelo GPT Image 2, elevando o custo de entrada."
    )

with aba_referencias:
    st.subheader("O que este playground cobre")
    st.markdown(
        """
        - **Geração direta:** texto para imagem com `POST /v1/images/generations`.
        - **Edição e composição:** uma ou mais imagens de referência com `POST /v1/images/edits`.
        - **Máscara:** delimita a região a editar; deve ser PNG com canal alpha e ter o mesmo tamanho/formato da imagem base.
        - **Saída flexível:** qualidade, resolução, PNG/JPEG/WebP e nível de compressão para JPEG/WebP.
        - **Segurança:** seleção de moderação `auto` ou `low`, com mensagens de erro e request ID quando disponíveis.
        """
    )
    st.warning(
        "O GPT Image 2 não aceita fundo transparente. Além disso, prompts complexos podem levar até cerca de dois minutos; "
        "texto pequeno, posicionamento rigoroso e consistência perfeita entre imagens ainda podem variar."
    )
    st.markdown(
        f"Fontes: [guia de geração de imagens]({DOCS_URL}) e "
        f"[página do modelo GPT Image 2]({MODELO_URL})."
    )
