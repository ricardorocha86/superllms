"""Playground visual para a API Images com GPT Image 2 e 2.5."""

import base64
import time

import streamlit as st
from openai import OpenAI


MODELOS = {
    "GPT Image 2.5 Flare": "gpt-image-2.5-flare",
    "GPT Image 2.5 Sunburst": "gpt-image-2.5-sunburst",
    "GPT Image 2.0 (anterior)": "gpt-image-2",
}
DOCS_URL = "https://developers.openai.com/api/docs/guides/image-generation"
CALCULADORA_URL = "https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency"
CUSTO_PARCIAL_USD = 0.003  # 100 image output tokens a US$ 30 / 1M tokens.
PRECO_TEXTO_INPUT_1M = 5.00
PRECO_IMAGEM_INPUT_1M = 8.00
PRECO_IMAGEM_OUTPUT_1M = 30.00

# Tarifas oficiais verificadas em 2026-09-08, iguais para Flare e Sunburst.
# O consumo por imagem varia: não reutilizar a tabela de saída do GPT Image 2.

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
    """Usa a chave configurada nos secrets do aplicativo."""
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


st.title("Playground GPT Image")
st.caption("Gere, edite e compare imagens usando a API oficial da OpenAI.")
modelo_nome = st.selectbox("Modelo de imagem", list(MODELOS))
MODELO = MODELOS[modelo_nome]
modelo_25 = MODELO.startswith("gpt-image-2.5-")
MODELO_URL = f"https://developers.openai.com/api/docs/models/{MODELO}"
st.badge(f"Modelo escolhido: {modelo_nome}", icon=":material/check_circle:")
st.caption(
    "Flare prioriza velocidade para criar e iterar. Sunburst oferece maior precisão "
    "para trabalhos detalhados, com geração mais demorada. GPT Image 2.0 mantém a versão anterior disponível."
)

aba_playground, aba_custos, aba_referencias = st.tabs(
    ["Playground", "Custos", "Guia rápido"]
)

with aba_playground:
    esquerda, direita = st.columns([1, 2], gap="large")

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
                help="Envie uma ou mais imagens para editar com o modelo escolhido.",
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
        dimensoes, aparencia, geracao = st.columns(3, gap="medium")
        with dimensoes:
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

            qualidades = ["auto", "low", "medium", "high"]
            if modelo_25:
                qualidades += ["xhigh", "max"]
            qualidade = st.select_slider("Qualidade", options=qualidades, value="low")
            st.caption("`low` para rascunhos; qualidades superiores podem aumentar tempo e custo.")
            if tamanho and tamanho != "auto":
                largura_saida, altura_saida = map(int, tamanho.split("x"))
                if largura_saida * altura_saida > 2560 * 1440:
                    st.caption("Resoluções acima de 2560 × 1440 são experimentais.")
        with aparencia:
            formato = st.selectbox("Formato", ["png", "jpeg", "webp"])
            compressao = None
            if formato in {"jpeg", "webp"}:
                compressao = st.slider("Compressão", 0, 100, 85)
                st.caption("JPEG tende a ser mais rápido que PNG.")
            fundos = ["auto", "opaque", "transparent"] if modelo_25 and formato != "jpeg" else ["auto", "opaque"]
            background = st.selectbox("Fundo", fundos)
            st.caption(
                "Para fundo transparente, use PNG ou WebP."
                if modelo_25 else "GPT Image 2.0 aceita fundo automático ou opaco e qualidade até high."
            )
        with geracao:
            moderacao = st.selectbox("Moderação", ["auto", "low"], help="`auto` é o padrão recomendado.")
            quantidade = st.number_input("Quantidade", min_value=1, max_value=4, value=1, step=1)
        total_saida = None

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
        args["stream"] = False
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
            parametros["stream"] = False
            if compressao is not None:
                parametros["output_compression"] = compressao

            inicio = time.perf_counter()
            parciais = []
            imagens = []
            uso_tokens = {"texto_input": 0, "imagem_input": 0, "imagem_output": 0}
            try:
                with st.spinner("A API está renderizando a imagem…", show_time=True):
                    client = OpenAI(api_key=obter_chave_openai(), timeout=600.0, max_retries=0)
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

                        for item in resposta.data:
                            imagens.append({
                                "bytes": base64.b64decode(item.b64_json),
                                "revised_prompt": getattr(item, "revised_prompt", None),
                            })
                        usage = getattr(resposta, "usage", None)
                        detalhes_input = getattr(usage, "input_tokens_details", None)
                        uso_tokens["texto_input"] = int(getattr(detalhes_input, "text_tokens", 0) or 0)
                        uso_tokens["imagem_input"] = int(getattr(detalhes_input, "image_tokens", 0) or 0)
                        uso_tokens["imagem_output"] = int(getattr(usage, "output_tokens", 0) or 0)
                    finally:
                        client.close()

                if not imagens:
                    raise RuntimeError("A API encerrou a resposta sem retornar uma imagem final.")
                st.session_state["imagem_resultados"] = {
                    "modelo": MODELO,
                    "modelo_nome": modelo_nome,
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
                if "organization must be verified" in str(exc).lower():
                    st.warning(
                        "A OpenAI exige a verificação da organização para liberar este modelo. "
                        "Após verificar, a liberação pode levar até 15 minutos."
                    )
                    st.markdown(
                        "[Verificar organização na OpenAI]"
                        "(https://platform.openai.com/settings/organization/general)"
                    )
                st.code(resumo_erro_api(exc), language="text")

    resultado = st.session_state.get("imagem_resultados")
    if resultado:
        st.divider()
        st.subheader("Resultado mais recente")
        st.caption(f"Modelo usado: {resultado.get('modelo_nome', 'GPT Image 2')}")
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
        colunas_imagens = st.columns(len(resultado["imagens"]), gap="large")
        for indice, (coluna, imagem) in enumerate(zip(colunas_imagens, resultado["imagens"]), start=1):
            with coluna:
                st.image(imagem["bytes"], caption=f"Imagem {indice}", use_container_width=True)
                st.download_button(
                    f"Baixar imagem {indice}",
                    data=imagem["bytes"],
                    file_name=f"{resultado.get('modelo', 'gpt-image-2')}-{indice}.{extensao}",
                    mime=mime,
                    use_container_width=True,
                )
                if imagem["revised_prompt"]:
                    st.caption("Prompt revisado pela API")
                    st.write(imagem["revised_prompt"])

with aba_custos:
    st.subheader(f"Custos — {modelo_nome}")
    st.caption("USD por 1 milhão de tokens · verificado em 08/09/2026.")
    st.table([
        {"Tipo": "Texto de entrada", "Preço": f"US$ {PRECO_TEXTO_INPUT_1M:.2f}"},
        {"Tipo": "Imagem de entrada", "Preço": f"US$ {PRECO_IMAGEM_INPUT_1M:.2f}"},
        {"Tipo": "Imagem de saída", "Preço": f"US$ {PRECO_IMAGEM_OUTPUT_1M:.2f}"},
    ])
    st.caption(
        "A estimativa usa tarifas de entrada sem desconto de cache. Tarifas iguais por token "
        "não significam custo igual por imagem: modelo, qualidade e tamanho alteram o consumo. "
        "Cada prévia parcial acrescenta 100 tokens de saída (US$ 0,003)."
    )
    st.info(
        "Para rascunhos e iterações, use `low`. `medium` equilibra qualidade e custo; "
        "`high` é indicado para o arquivo final; na linha 2.5, há também `xhigh` e `max`."
    )
    st.markdown(
        f"Para resoluções flexíveis, `auto` e custos de imagens de referência, consulte a "
        f"[seção oficial de cálculo de custos]({CALCULADORA_URL}). Em edições, cada imagem de entrada "
        "também contribui para o custo de entrada."
    )

with aba_referencias:
    st.subheader("O que este playground cobre")
    st.markdown(
        """
        - **Geração direta:** texto para imagem com `POST /v1/images/generations`.
        - **Edição e composição:** uma ou mais imagens de referência com `POST /v1/images/edits`.
        - **Máscara:** delimita a região a editar; deve ser PNG com canal alpha e ter o mesmo tamanho/formato da imagem base.
        - **Saída flexível:** qualidade, resolução, PNG/JPEG/WebP e nível de compressão para JPEG/WebP.
        - **Novidades 2.5:** Flare e Sunburst, qualidades `xhigh` e `max` e fundo transparente em PNG/WebP.
        - **Segurança:** seleção de moderação `auto` ou `low`, com mensagens de erro e request ID quando disponíveis.
        """
    )
    st.warning(
        "Sunburst e qualidades superiores podem levar mais tempo para gerar. "
        "Texto pequeno, posicionamento rigoroso e consistência perfeita entre imagens ainda podem variar."
    )
    st.markdown(
        f"Fontes: [guia de geração de imagens]({DOCS_URL}) e "
        f"[página do modelo {modelo_nome}]({MODELO_URL})."
    )
