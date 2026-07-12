# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import os
import unicodedata
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="EmbeddingLab | Aula com Gemini",
    page_icon=":material/hub:",
    layout="wide",
)


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SAMPLE_IMAGES = {
    "Cachorro": {
        "url": "https://picsum.photos/id/1025/800/520",
        "mime_type": "image/jpeg",
        "caption": "Exemplo de imagem com um animal.",
    },
    "Paisagem litorânea": {
        "url": "https://picsum.photos/id/1018/800/520",
        "mime_type": "image/jpeg",
        "caption": "Exemplo de paisagem aberta.",
    },
    "Cena cotidiana": {
        "url": "https://picsum.photos/id/292/800/520",
        "mime_type": "image/jpeg",
        "caption": "Exemplo visual para testar objetos e contexto.",
    },
}

IMDB_URL = (
    "https://raw.githubusercontent.com/ricardorocha86/Datasets/refs/heads/master/"
    "reviews-homem-aranha.csv"
)

SCRIPT_METADATA = {
    "01_congresso_busca_semantica.py": {
        "label": "Busca semântica em títulos do Congresso UFBA",
        "summary": "Replica o experimento de recomendação de trabalhos a partir de uma consulta textual.",
        "output_hint": "Salva um CSV ranqueado em outputs/congresso_ufba_resultados.csv.",
    },
    "02_reviews_zero_shot.py": {
        "label": "Classificação zero-shot de reviews do Homem-Aranha",
        "summary": "Classifica reviews como Promotor, Neutro ou Detrator usando descrições das classes.",
        "output_hint": "Salva previsões e métricas em outputs/reviews_spiderman_zero_shot.csv.",
    },
    "03_similaridade_cosseno.py": {
        "label": "Laboratório de similaridade do cosseno",
        "summary": "Roda exemplos dirigidos de proximidade semântica inspirados na aula teórica.",
        "output_hint": "Salva os pares avaliados em outputs/similaridade_cosseno.csv.",
    },
}


#load_dotenv()
#API_KEY = os.getenv("GEMINI_API_KEY")


st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #f5f2eb;
        --panel: #fffdfa;
        --ink: #172033;
        --muted: #556070;
        --brand: #0f766e;
        --brand-2: #1d4ed8;
        --accent: #c2410c;
        --line: #ddd4c5;
        --soft: rgba(255, 255, 255, 0.84);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 32%),
            radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 28%),
            linear-gradient(180deg, #f8f5ef 0%, #f2ece1 100%);
        color: var(--ink);
        font-family: "Source Sans 3", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 253, 250, 0.94);
        border-right: 1px solid var(--line);
    }

    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,250,241,0.96));
        border: 1px solid rgba(221, 212, 197, 0.95);
        border-radius: 26px;
        padding: 30px 32px 26px 32px;
        box-shadow: 0 18px 40px rgba(84, 67, 42, 0.08);
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-family: "Space Grotesk", sans-serif;
        font-size: 3rem;
        line-height: 1.0;
        margin: 0 0 0.45rem 0;
        color: #0f172a;
    }

    .hero p {
        margin: 0.35rem 0;
        color: var(--muted);
        font-size: 1.08rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.38rem 0.8rem;
        font-size: 0.9rem;
        font-weight: 700;
        background: rgba(15, 118, 110, 0.10);
        color: var(--brand);
        border: 1px solid rgba(15, 118, 110, 0.18);
    }

    .card {
        background: var(--soft);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 18px 18px 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 18px rgba(84, 67, 42, 0.05);
    }

    .card h3 {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.05rem;
        margin: 0 0 0.4rem 0;
    }

    .card p, .card li {
        color: var(--muted);
        margin-bottom: 0.3rem;
    }

    .section-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .metric-strip {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 12px 14px;
        background: var(--soft);
        margin-bottom: 12px;
    }

    .gallery-note {
        font-size: 0.92rem;
        color: var(--muted);
    }

    code {
        color: #9a3412 !important;
        background: #fff3e8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def nps_label(rating: float) -> str:
    if rating >= 9:
        return "Promotor"
    if rating >= 7:
        return "Neutro"
    return "Detrator"


@st.cache_data(show_spinner=False)
def load_imdb_reviews() -> pd.DataFrame:
    df = pd.read_csv(IMDB_URL)
    df["NPS"] = df["Rating"].apply(nps_label)
    return df


@st.cache_data(show_spinner=False)
def build_search_demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Título": [
                "Modelos preditivos para evasão escolar",
                "Mineração de textos jurídicos com aprendizado de máquina",
                "Aplicações de inteligência artificial em diagnóstico médico",
                "Educação ambiental em comunidades pesqueiras",
                "Visualização de dados para políticas públicas urbanas",
                "Análise estatística de séries temporais climáticas",
                "Ensino de programação com jogos digitais",
                "Reconhecimento de padrões em imagens de satélite",
            ],
            "Resumo": [
                "Estudo sobre sinais acadêmicos e socioeconômicos para prever evasão no ensino superior.",
                "Uso de embeddings e classificação para organizar documentos jurídicos extensos.",
                "Discussão de modelos capazes de apoiar triagem e análise de exames clínicos.",
                "Projeto de extensão com foco em sustentabilidade e práticas locais de preservação.",
                "Painéis interativos para leitura de mobilidade, violência e infraestrutura urbana.",
                "Métodos estatísticos para tendência, sazonalidade e eventos extremos em clima.",
                "Sequências didáticas com gamificação para introdução a algoritmos e lógica.",
                "Extração de características visuais para mapear uso do solo e cobertura vegetal.",
            ],
        }
    )


@st.cache_data(show_spinner=False)
def build_reviews_demo_df() -> pd.DataFrame:
    data = [
        {
            "Rating": 10,
            "Title": "Espetacular do início ao fim",
            "Review": "O filme é emocionante, divertido e fecha a história com muita energia. Eu recomendaria sem hesitar.",
        },
        {
            "Rating": 9,
            "Title": "Muito acima do esperado",
            "Review": "Saí da sessão querendo rever. Atuações fortes, ritmo bom e vários momentos memoráveis.",
        },
        {
            "Rating": 8,
            "Title": "Bom, mas não perfeito",
            "Review": "Gostei bastante da experiência, embora alguns trechos pareçam longos. Ainda assim, funciona bem.",
        },
        {
            "Rating": 7,
            "Title": "Entretenimento competente",
            "Review": "É um filme agradável e cumpre o que promete, mas não me marcou tanto quanto eu imaginava.",
        },
        {
            "Rating": 6,
            "Title": "Esperava mais",
            "Review": "Tem ideias interessantes, porém a execução não me convenceu. Não é ruim, mas também não empolga.",
        },
        {
            "Rating": 5,
            "Title": "Regular para fraco",
            "Review": "Algumas cenas são boas, só que o conjunto parece bagunçado e cansativo. Eu não recomendaria.",
        },
        {
            "Rating": 3,
            "Title": "Decepcionante",
            "Review": "Achei confuso e exagerado. Em vários momentos eu só queria que o filme acabasse logo.",
        },
        {
            "Rating": 2,
            "Title": "Quase nada funciona",
            "Review": "Não me conectei com a história e a experiência foi frustrante. Passaria longe de uma nova sessão.",
        },
        {
            "Rating": 9,
            "Title": "Entrega emoção e nostalgia",
            "Review": "O filme entende o fã e entrega uma experiência muito satisfatória. Vale muito a recomendação.",
        },
        {
            "Rating": 8,
            "Title": "Sólido e divertido",
            "Review": "Tem bons personagens e boas ideias. Talvez não seja brilhante o tempo todo, mas funciona.",
        },
        {
            "Rating": 6,
            "Title": "Morno",
            "Review": "Não achei péssimo, só não vi nada realmente especial. É o tipo de filme que eu esqueço rápido.",
        },
        {
            "Rating": 4,
            "Title": "Abaixo do esperado",
            "Review": "Fui com expectativa alta e saí frustrado. O resultado me pareceu inconsistente e pouco envolvente.",
        },
    ]
    df = pd.DataFrame(data)
    df["NPS"] = df["Rating"].apply(nps_label)
    return df


@st.cache_data(show_spinner=False)
def build_complaints_demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pedido": [
                "Fone bluetooth",
                "Fone bluetooth",
                "Liquidificador",
                "Liquidificador",
                "Cadeira de escritório",
                "Cadeira de escritório",
                "Livro técnico",
                "Livro técnico",
                "Smartwatch",
                "Cafeteira",
            ],
            "Título": [
                "Entrega atrasou muito",
                "Cobrança duplicada no cartão",
                "Produto veio com defeito",
                "Atendimento não responde",
                "Peça errada no pacote",
                "Troca está demorando",
                "Livro veio amassado",
                "Preço anunciado estava diferente",
                "Relógio descarrega rápido",
                "Suporte não resolveu",
            ],
            "Comentário": [
                "Meu pedido chegou muito depois do prazo prometido e eu não tive atualização nenhuma.",
                "A loja lançou a cobrança duas vezes e até agora não estornou o valor extra.",
                "O aparelho veio quebrado e não liga desde o primeiro uso.",
                "Já mandei mensagem várias vezes e ninguém responde no chat nem por e-mail.",
                "Recebi um acessório diferente do que estava descrito no anúncio.",
                "Solicitei a troca há dias, mas o processo não avança e ninguém dá retorno.",
                "O conteúdo é bom, porém o livro chegou com capas dobradas e várias marcas.",
                "No anúncio aparecia um preço, mas na hora de pagar o valor subiu.",
                "A bateria acaba em poucas horas, bem diferente do que foi prometido.",
                "O suporte respondeu uma vez e depois sumiu sem resolver o problema.",
            ],
        }
    )


@st.cache_data(show_spinner=False)
def fetch_remote_image(url: str) -> bytes:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.content


@st.cache_resource(show_spinner=False)
def get_client(api_key: str | None):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def find_column(df: pd.DataFrame, target: str) -> str | None:
    target_slug = slugify(target)
    for column in df.columns:
        if slugify(column) == target_slug:
            return column
    return None


def build_embed_config(model_name: str, task_type: str | None, output_dim: int):
    kwargs = {"output_dimensionality": output_dim}
    if model_name == "gemini-embedding-001" and task_type:
        kwargs["task_type"] = task_type
    return types.EmbedContentConfig(**kwargs)


def prepare_text_for_model(
    model_name: str,
    content: str,
    mode: str | None = None,
    title: str | None = None,
) -> str:
    text = str(content).strip()
    if not text:
        return ""

    if model_name != "gemini-embedding-2-preview":
        return text

    if mode == "similarity":
        return f"task: sentence similarity | query: {text}"
    if mode == "classification":
        return f"task: classification | query: {text}"
    if mode == "clustering":
        return f"task: clustering | query: {text}"
    if mode == "retrieval_query":
        return f"task: search result | query: {text}"
    if mode == "retrieval_document":
        safe_title = title.strip() if title else "none"
        return f"title: {safe_title} | text: {text}"
    return text


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.array(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_batch(client, model_name: str, contents, config):
    return client.models.embed_content(
        model=model_name,
        contents=contents,
        config=config,
    )


def format_embedding_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    message = str(exc)
    if status_code == 429 or "RESOURCE_EXHAUSTED" in message or "429" in message:
        return (
            "A API de embeddings do Gemini atingiu o limite de uso no momento. "
            "Aguarde um pouco e tente novamente."
        )
    return f"Não foi possível gerar embeddings agora: {message}"


def embed_texts(
    client,
    model_name: str,
    texts: list[str],
    output_dim: int,
    mode: str | None = None,
    titles: list[str] | None = None,
    progress_text: str | None = None,
) -> np.ndarray:
    cleaned = []
    for idx, raw_text in enumerate(texts):
        if not str(raw_text).strip():
            continue
        title = titles[idx] if titles else None
        cleaned.append(prepare_text_for_model(model_name, raw_text, mode=mode, title=title))

    if not cleaned:
        raise ValueError("Não há textos válidos para vetorização.")

    task_type_map = {
        "similarity": "SEMANTIC_SIMILARITY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
        "retrieval_query": "RETRIEVAL_QUERY",
        "retrieval_document": "RETRIEVAL_DOCUMENT",
    }
    config = build_embed_config(model_name, task_type_map.get(mode), output_dim)

    embeddings = []
    chunk_size = 20
    progress = st.progress(0, text=progress_text) if progress_text else None

    for start in range(0, len(cleaned), chunk_size):
        batch = cleaned[start : start + chunk_size]
        try:
            response = embed_batch(client, model_name, batch, config)
        except Exception as exc:
            if progress:
                progress.empty()
            raise RuntimeError(format_embedding_error(exc)) from exc
        batch_embeddings = [np.array(item.values, dtype=float) for item in response.embeddings]

        if len(batch_embeddings) != len(batch):
            batch_embeddings = []
            for item in batch:
                try:
                    single = embed_batch(client, model_name, [item], config)
                except Exception as exc:
                    if progress:
                        progress.empty()
                    raise RuntimeError(format_embedding_error(exc)) from exc
                batch_embeddings.append(np.array(single.embeddings[0].values, dtype=float))

        embeddings.extend(batch_embeddings)

        if progress:
            done = min(start + chunk_size, len(cleaned))
            progress.progress(done / len(cleaned), text=f"{progress_text} ({done}/{len(cleaned)})")

    if progress:
        progress.empty()

    matrix = np.vstack(embeddings)
    return normalize_rows(matrix)


def embed_multimodal_content(
    client,
    model_name: str,
    contents: list,
    output_dim: int,
) -> np.ndarray:
    config = build_embed_config(model_name, None, output_dim)
    try:
        response = embed_batch(client, model_name, contents, config)
    except Exception as exc:
        raise RuntimeError(format_embedding_error(exc)) from exc
    matrix = np.vstack([np.array(item.values, dtype=float) for item in response.embeddings])
    return normalize_rows(matrix)


def parse_label_descriptions(raw_text: str) -> dict[str, str]:
    labels = {}
    for line in raw_text.splitlines():
        if ":" not in line:
            continue
        label, description = line.split(":", 1)
        label = label.strip()
        description = description.strip()
        if label and description:
            labels[label] = description
    return labels


def pair_summary(texts: list[str], similarity_matrix: np.ndarray) -> tuple[str, str]:
    best_pair = None
    worst_pair = None
    best_score = -2.0
    worst_score = 2.0

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = float(similarity_matrix[i, j])
            if score > best_score:
                best_score = score
                best_pair = f"{texts[i]}  <->  {texts[j]}  ({score:.3f})"
            if score < worst_score:
                worst_score = score
                worst_pair = f"{texts[i]}  <->  {texts[j]}  ({score:.3f})"

    return best_pair or "-", worst_pair or "-"


def confusion_dataframe(labels: list[str], matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for true_idx, true_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            rows.append(
                {
                    "Real": true_label,
                    "Previsto": pred_label,
                    "Quantidade": int(matrix[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def load_script_catalog() -> list[dict[str, str]]:
    catalog = []
    if not SCRIPTS_DIR.exists():
        return catalog

    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        try:
            module = ast.parse(text)
            docstring = ast.get_docstring(module) or ""
        except SyntaxError:
            docstring = ""
        metadata = SCRIPT_METADATA.get(path.name, {})
        catalog.append(
            {
                "name": path.name,
                "path": str(path),
                "code": text,
                "label": metadata.get("label", path.stem),
                "summary": metadata.get("summary", docstring.splitlines()[0] if docstring else ""),
                "output_hint": metadata.get("output_hint", ""),
            }
        )
    return catalog


def render_header():
    st.markdown(
        """
        <div class="hero">
            <h1>Embeddings com Gemini</h1>
            <p>Uma aula aplicada sobre representação vetorial, similaridade semântica, busca,
            classificação zero-shot e multimodalidade.</p>
            <div class="pill-row">
                <span class="pill">Espaço vetorial</span>
                <span class="pill">Similaridade do cosseno</span>
                <span class="pill">Busca semântica</span>
                <span class="pill">Zero-shot</span>
                <span class="pill">Multimodalidade</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theory(model_name: str, output_dim: int):
    render_header()

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(
            """
            <div class="card">
                <h3>A ideia central da aula</h3>
                <p>Embeddings são vetores que transformam conteúdos em coordenadas numéricas.
                A consequência prática disso é poderosa: textos, imagens e outros objetos passam a
                poder ser comparados por significado, e não apenas por palavras idênticas.</p>
                <p>Quando dois conteúdos ocupam regiões próximas do espaço vetorial, dizemos que
                eles são semanticamente parecidos. É essa geometria que sustenta busca vetorial,
                recomendação, agrupamento e classificação sem treinamento específico.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
                <h3>Como pensar embeddings em sala</h3>
                <p>1. O modelo lê o conteúdo e o projeta em um espaço de alta dimensão.</p>
                <p>2. A posição do vetor captura relações de sentido e contexto.</p>
                <p>3. A comparação entre vetores permite decidir o que é mais próximo, mais distante
                ou mais alinhado com uma tarefa.</p>
                <p>4. A mesma lógica serve para consulta-documento, texto-classe e texto-imagem.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
                <h3>Três perguntas que organizam a aula</h3>
                <p>Como representar significado numericamente?</p>
                <p>Como medir proximidade entre vetores?</p>
                <p>Como transformar essa proximidade em uma decisão útil?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
                <h3>Aplicações que vamos observar</h3>
                <p>Busca semântica em títulos de trabalhos.</p>
                <p>Classificação zero-shot de reviews.</p>
                <p>Comparação multimodal entre texto e imagem.</p>
                <p>Exercício aplicado com comentários de e-commerce.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Similaridade do Cosseno</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.markdown(
            """
            A similaridade do cosseno mede o alinhamento entre dois vetores.
            Ela é especialmente útil em embeddings porque privilegia direção semântica,
            e não apenas magnitude. Em termos intuitivos:

            - valor próximo de `1`: conteúdos muito parecidos;
            - valor próximo de `0`: pouca relação semântica;
            - valor negativo: direções opostas no espaço vetorial.
            """
        )
        st.latex(r"\cos(\theta) = \frac{\vec a \cdot \vec b}{\|\vec a\| \|\vec b\|}")
    with c2:
        cosine_df = pd.DataFrame(
            {
                "Faixa": ["0.85 a 1.00", "0.55 a 0.84", "0.20 a 0.54", "-1.00 a 0.19"],
                "Leitura didática": [
                    "Alta proximidade conceitual",
                    "Relação parcial ou contextual",
                    "Semelhança fraca",
                    "Baixa proximidade ou oposição",
                ],
                "Uso em aula": [
                    "buscar o top-1 ou top-k",
                    "comparar rótulos concorrentes",
                    "observar fronteiras de decisão",
                    "entender casos-limite",
                ],
            }
        )
        st.dataframe(cosine_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Modelos e decisões de projeto</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="card">
                <h3><code>gemini-embedding-001</code></h3>
                <p>Modelo estável para texto. Permite informar <code>task_type</code> diretamente
                na chamada e funciona muito bem para similaridade, classificação e recuperação textual.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="card">
                <h3><code>gemini-embedding-2-preview</code></h3>
                <p>Modelo multimodal. Trabalha com texto, imagem, áudio, vídeo e PDF no mesmo espaço vetorial.
                Para tarefas textuais, a orientação da documentação é colocar a tarefa no próprio prompt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="card">
                <h3>Dimensão do vetor</h3>
                <p>As dimensões mais úteis em aula costumam ser 768, 1536 e 3072.
                Em dimensões menores que 3072, normalizamos os vetores antes de comparar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Os espaços vetoriais de `gemini-embedding-001` e `gemini-embedding-2-preview` são incompatíveis. "
        "Não compare vetores produzidos por modelos diferentes."
    )

    st.markdown('<div class="section-title">Código-base de referência</div>', unsafe_allow_html=True)
    st.code(
        """from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["O filme foi excelente.", "O filme foi decepcionante."],
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
    ),
)""",
        language="python",
    )

    st.code(
        """def preparar_consulta(texto):
    return f"task: search result | query: {texto}"

def preparar_documento(texto, titulo="none"):
    return f"title: {titulo} | text: {texto}"

result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[preparar_consulta("trabalhos sobre IA aplicada")],
    config=types.EmbedContentConfig(output_dimensionality=768),
)""",
        language="python",
    )

    box1, box2, box3 = st.columns(3)
    box1.metric("Modelo ativo na barra lateral", model_name)
    box2.metric("Dimensão selecionada", output_dim)
    box3.metric("Métrica central da aula", "cosseno")


def render_similarity_lab(client, model_name: str, output_dim: int):
    st.subheader("Similaridade semântica e mapa vetorial")
    st.write(
        "Esta aba junta duas leituras da mesma ideia: primeiro, comparamos pares de frases; "
        "depois, projetamos vários vetores em 2D para enxergar a geometria do espaço semântico."
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Distância e similaridade do cosseno")
        st.write(
            "O cálculo central desta aba é a similaridade do cosseno. Ela mede o quanto dois vetores "
            "apontam na mesma direção. Quanto mais alinhados estiverem, maior tende a ser a proximidade semântica."
        )
        st.latex(r"\cos(\theta) = \frac{\vec a \cdot \vec b}{\|\vec a\| \|\vec b\|}")
        st.caption(
            "Se quisermos falar em distância do cosseno, uma forma comum é usar `1 - cos(θ)`, "
            "transformando alinhamento em afastamento."
        )
    with right:
        metric_df = pd.DataFrame(
            {
                "Faixa": ["próximo de 1", "próximo de 0", "próximo de -1"],
                "Leitura": [
                    "vetores muito alinhados; forte proximidade semântica",
                    "pouco alinhamento; relação fraca ou neutra",
                    "direções opostas; contraste forte",
                ],
            }
        )
        st.dataframe(metric_df, width="stretch", hide_index=True)
        st.info(
            "A faixa teórica da similaridade do cosseno continua sendo de `-1` a `1`. "
            "Na documentação oficial de embeddings do Gemini, essa continua sendo a interpretação da métrica."
        )

    st.warning(
        "Observação importante: em muitos usos modernos de embeddings, especialmente com vetores normalizados "
        "e tarefas de recuperação semântica, é comum ver na prática scores concentrados entre `0` e `1`. "
        "Isso não significa que a métrica tenha mudado de definição; significa apenas que, nesses dados, os vetores "
        "costumam aparecer mais alinhados do que opostos."
    )

    st.markdown("### Mapa semântico com PCA")
    default_corpus = """O gato dormiu no sofá.
O cachorro correu atrás da bola no parque.
A inteligência artificial está mudando a medicina.
Redes neurais profundas aprendem representações complexas.
Uma boa receita de risoto pede caldo quente.
Bolo fofinho costuma usar essência de baunilha.
Marte segue como candidato forte para colonização humana.
Buracos negros supermassivos habitam o centro de galáxias."""
    raw = st.text_area("Frases da atividade", value=default_corpus, height=200, key="similarity_corpus")

    if st.button("Gerar mapa semântico", key="btn_similarity_map", type="primary"):
        texts = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(texts) < 3:
            st.warning("Use pelo menos 3 frases para o mapa ficar interessante.")
            return

        try:
            embeddings = embed_texts(
                client,
                model_name,
                texts,
                output_dim=output_dim,
                mode="similarity",
                progress_text="Calculando embeddings das frases",
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return

        reduced = PCA(n_components=2).fit_transform(embeddings)
        sim = cosine_similarity(embeddings)
        best_pair, worst_pair = pair_summary(texts, sim)

        m1, m2, m3 = st.columns(3)
        m1.metric("Frases no experimento", len(texts))
        m2.metric("Par mais próximo", best_pair)
        m3.metric("Par mais distante", worst_pair)

        df_points = pd.DataFrame({"Dim 1": reduced[:, 0], "Dim 2": reduced[:, 1], "Frase": texts})
        df_sim = pd.DataFrame(sim, index=texts, columns=texts)

        chart = (
            alt.Chart(df_points)
            .mark_circle(size=250, color="#0f766e", opacity=0.88)
            .encode(
                x=alt.X("Dim 1", axis=alt.Axis(grid=False)),
                y=alt.Y("Dim 2", axis=alt.Axis(grid=False)),
                tooltip=["Frase"],
            )
            .properties(height=430)
        )
        labels = (
            alt.Chart(df_points)
            .mark_text(align="left", dx=12, dy=-8, color="#334155", fontSize=12)
            .encode(x="Dim 1", y="Dim 2", text="Frase")
        )
        st.altair_chart(chart + labels, width="stretch")
        st.dataframe(
            df_sim.style.background_gradient(cmap="YlGnBu", axis=None).format("{:.3f}"),
            width="stretch",
        )


def render_search_lab(client, model_name: str, output_dim: int):
    st.subheader("Busca semântica didática")
    st.write(
        "Aqui a ideia é mostrar o mecanismo da busca vetorial com um conjunto pequeno e legível. "
        "Os experimentos maiores com bases online ficaram reservados para os scripts da aba final."
    )

    demo_df = build_search_demo_df()
    st.dataframe(demo_df, width="stretch", hide_index=True)

    control_1, control_2 = st.columns([1.8, 0.8])
    with control_1:
        query = st.text_input(
            "Consulta semântica",
            value="Projetos relacionados a inteligência artificial, análise de dados e reconhecimento de padrões.",
        )
    with control_2:
        top_k = st.slider("Top-k", min_value=3, max_value=8, value=5, key="search_top_k")

    if st.button("Buscar documentos mais próximos", key="btn_congress_search", type="primary"):
        df = demo_df.copy()
        titles = df["Título"].fillna("").astype(str).tolist()
        documents = (df["Título"] + " | " + df["Resumo"]).tolist()
        try:
            doc_embeddings = embed_texts(
                client,
                model_name,
                documents,
                output_dim=output_dim,
                mode="retrieval_document",
                titles=titles,
                progress_text="Vetorizando documentos da base didática",
            )
            query_embedding = embed_texts(
                client,
                model_name,
                [query],
                output_dim=output_dim,
                mode="retrieval_query",
                progress_text="Vetorizando a consulta",
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return

        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        result = df.copy()
        result["Similaridade"] = scores
        result = result.sort_values("Similaridade", ascending=False).reset_index(drop=True)
        result["Ranking"] = result.index + 1

        best_title = result.loc[0, "Título"]
        m1, m2 = st.columns([2.2, 1])
        with m1:
            st.markdown("**Documento mais próximo da consulta**")
            st.write(best_title)
        with m2:
            st.metric("Score do top-1", f"{result.loc[0, 'Similaridade']:.3f}")

        chart_df = result.head(top_k)[["Título", "Similaridade"]].iloc[::-1]
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#1d4ed8")
            .encode(
                x=alt.X("Similaridade:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Título:N", sort=None),
                tooltip=["Título", alt.Tooltip("Similaridade:Q", format=".3f")],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, width="stretch")
        st.dataframe(result.head(top_k), width="stretch", hide_index=True)

    st.info(
        "Em recuperação assimétrica, a consulta e o documento não têm o mesmo papel. "
        "Por isso, no `gemini-embedding-2-preview`, usamos prefixos diferentes para consulta e documento."
    )


def render_zero_shot_case(client, model_name: str, output_dim: int):
    st.subheader("Classificação zero-shot de reclamações")
    st.write(
        "Aqui o problema é categorizar reclamações de clientes em um cenário de e-commerce. "
        "A classificação continua sendo zero-shot: a decisão é feita pela proximidade entre a reclamação e as descrições das classes."
    )

    dataset = build_complaints_demo_df()
    c1, c2 = st.columns([1, 1])
    with c1:
        text_column = st.selectbox("Texto a classificar", options=["Comentário", "Título"], index=0, key="zero_shot_text_column")
    with c2:
        st.markdown(
            '<div class="metric-strip">O objetivo agora não é medir satisfação geral, mas identificar a natureza da reclamação. '
            "Isso é útil para triagem, roteamento de atendimento e análise de recorrência de problemas.</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(dataset, width="stretch", hide_index=True)

    default_labels = """Entrega: reclamação sobre atraso, transporte, prazo, embalagem ou atualização de envio.
Cobrança: reclamação sobre preço, pagamento, cobrança duplicada, estorno ou divergência de valor.
Defeito no produto: reclamação sobre item quebrado, dano físico, mau funcionamento ou baixa durabilidade.
Atendimento: reclamação sobre demora, ausência de resposta, suporte ruim ou dificuldade no contato.
Pedido incorreto: reclamação sobre item trocado, peça errada, erro na separação ou produto diferente do anúncio."""
    label_text = st.text_area(
        "Descrições das classes (formato Classe: descrição)",
        value=default_labels,
        height=150,
        key="zero_shot_label_text",
    )

    if st.button("Classificar reclamações", key="btn_classify_reviews", type="primary"):
        label_map = parse_label_descriptions(label_text)
        expected_labels = [
            "Entrega",
            "Cobrança",
            "Defeito no produto",
            "Atendimento",
            "Pedido incorreto",
        ]
        if set(label_map.keys()) != set(expected_labels):
            st.warning("Use exatamente as classes Entrega, Cobrança, Defeito no produto, Atendimento e Pedido incorreto.")
            return

        sample = dataset.copy()
        texts = sample[text_column].fillna("").astype(str).tolist()

        try:
            text_embeddings = embed_texts(
                client,
                model_name,
                texts,
                output_dim=output_dim,
                mode="classification",
                progress_text="Vetorizando reclamações",
            )
            class_embeddings = embed_texts(
                client,
                model_name,
                [label_map[label] for label in expected_labels],
                output_dim=output_dim,
                mode="classification",
                progress_text="Vetorizando descrições das classes",
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return

        scores = cosine_similarity(text_embeddings, class_embeddings)
        top_idx = np.argmax(scores, axis=1)
        sample["Classificação"] = [expected_labels[idx] for idx in top_idx]
        sample["Confiança"] = scores.max(axis=1)
        sorted_scores = np.sort(scores, axis=1)
        sample["Margem"] = sorted_scores[:, -1] - sorted_scores[:, -2]

        m1, m2, m3 = st.columns(3)
        m1.metric("Reclamações analisadas", len(sample))
        m2.metric("Classes possíveis", len(expected_labels))
        m3.metric("Texto usado", text_column)

        st.markdown("**Resultado da categorização**")
        result_df = sample[["Pedido", "Título", "Comentário", "Classificação", "Confiança", "Margem"]].copy()
        st.dataframe(result_df, width="stretch", hide_index=True)

        distribution = (
            result_df["Classificação"]
            .value_counts()
            .rename_axis("Classe")
            .reset_index(name="Quantidade")
        )
        chart = (
            alt.Chart(distribution)
            .mark_bar(color="#0f766e", cornerRadiusTopRight=5, cornerRadiusTopLeft=5)
            .encode(
                x=alt.X("Classe:N", sort="-y"),
                y="Quantidade:Q",
                tooltip=["Classe", "Quantidade"],
            )
            .properties(height=280)
        )
        st.markdown("**Distribuição das categorias previstas**")
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Neste exemplo, a leitura principal é qualitativa: observar se a categoria prevista faz sentido "
            "para cada reclamação e como a descrição das classes influencia a triagem."
        )


def render_multimodal_lab(client, model_name: str, output_dim: int):
    st.subheader("Embeddings multimodais")
    st.write(
        "No `gemini-embedding-2-preview`, texto e imagem podem ocupar o mesmo espaço vetorial. "
        "Isso permite comparar uma foto com rótulos textuais, criar busca cruzada entre modalidades "
        "e até agregar texto + imagem em um único embedding."
    )

    gallery_columns = st.columns(3)
    for column, (label, info) in zip(gallery_columns, SAMPLE_IMAGES.items()):
        with column:
            st.image(info["url"], caption=label, width="stretch")
            st.caption(info["caption"])

    source_mode = st.radio("Fonte da imagem", options=["Escolher exemplo", "Enviar arquivo"], horizontal=True)
    selected_name = None
    image_bytes = None
    mime_type = None

    if source_mode == "Escolher exemplo":
        selected_name = st.selectbox("Imagem de exemplo", options=list(SAMPLE_IMAGES.keys()), key="multimodal_selected_image")
        selected = SAMPLE_IMAGES[selected_name]
        image_bytes = fetch_remote_image(selected["url"])
        mime_type = selected["mime_type"]
        st.markdown(f'<p class="gallery-note">Imagem atual: {selected_name}</p>', unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"], key="multimodal_upload")
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            mime_type = uploaded.type
            st.image(uploaded, width="stretch")

    caption_text = st.text_input(
        "Legenda opcional para agregar com a imagem",
        value="",
        help="Se preenchida, o embedding será calculado com texto + imagem no mesmo conteúdo.",
        key="multimodal_caption_text",
    )
    labels_input = st.text_area(
        "Rótulos textuais para comparar com a imagem",
        value=(
            "Um cachorro ou outro animal doméstico\n"
            "Uma paisagem aberta ou cena de viagem\n"
            "Objetos ou cena cotidiana\n"
            "Cena urbana ou arquitetura\n"
            "Documento ou material textual"
        ),
        height=140,
        key="multimodal_labels_input",
    )

    if st.button("Comparar imagem com rótulos", key="btn_multimodal_compare", type="primary"):
        if model_name != "gemini-embedding-2-preview":
            st.error("Selecione `gemini-embedding-2-preview` na barra lateral para usar multimodalidade.")
            return
        if not image_bytes:
            st.warning("Escolha uma imagem de exemplo ou envie um arquivo.")
            return

        labels = [line.strip() for line in labels_input.splitlines() if line.strip()]
        if not labels:
            st.warning("Informe pelo menos um rótulo textual.")
            return

        contents = []
        if caption_text.strip():
            contents.append(caption_text.strip())
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        try:
            image_embedding = embed_multimodal_content(client, model_name, contents, output_dim=output_dim)
            label_embeddings = embed_texts(
                client,
                model_name,
                labels,
                output_dim=output_dim,
                mode="classification",
                progress_text="Vetorizando rótulos textuais",
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return
        scores = cosine_similarity(image_embedding, label_embeddings)[0]
        result_df = pd.DataFrame({"Rótulo": labels, "Score": scores}).sort_values("Score", ascending=False)

        st.success(f"Rótulo mais próximo: {result_df.iloc[0]['Rótulo']}")
        chart = (
            alt.Chart(result_df)
            .mark_bar(color="#0f766e", cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                x=alt.X("Score:Q"),
                y=alt.Y("Rótulo:N", sort="-x"),
                tooltip=["Rótulo", alt.Tooltip("Score:Q", format=".3f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
        st.dataframe(result_df.style.format({"Score": "{:.3f}"}), width="stretch")

    st.markdown("### Outras modalidades para explorar")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(
            """
            <div class="card">
                <h3>Áudio</h3>
                <p>Podemos embutir trechos de fala, música ou ruído ambiente e comparar com descrições textuais.
                Um bom exercício é distinguir notícia, aula, propaganda e conversa espontânea.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            """
            <div class="card">
                <h3>Vídeo</h3>
                <p>Vídeos curtos podem ser convertidos para o mesmo espaço vetorial. Isso abre espaço para busca por cenas,
                agrupamento por conteúdo visual e comparação entre descrição textual e clipe.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            """
            <div class="card">
                <h3>PDF e documentos</h3>
                <p>É possível embutir páginas de documentos e recuperar materiais por tema, estilo e evidência visual.
                Isso é útil em RAG, análise documental e triagem de acervos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_example_page(client, model_name: str, output_dim: int):
    st.subheader("Exemplo completo: reviews do Homem-Aranha")
    st.write(
        "Esta seção retoma o exemplo maior da aula antiga: classificação de sentimentos em reviews do filme "
        "*Spider-Man: No Way Home* usando embeddings e zero-shot learning."
    )

    df = load_imdb_reviews()

    st.markdown("### 1. Base de dados carregada")
    st.write(
        "A base já está pré-processada e contém avaliações, título do review, texto do review e o filme. "
        "A primeira leitura didática é simplesmente observar a tabela e entender o tipo de dado com que vamos trabalhar."
    )
    st.dataframe(df, width="stretch")

    st.code(
        """import pandas as pd

url = "https://raw.githubusercontent.com/ricardorocha86/Datasets/refs/heads/master/reviews-homem-aranha.csv"
dados = pd.read_csv(url)
dados.head()""",
        language="python",
    )

    st.markdown("### 2. Qual é o interesse analítico?")
    st.markdown(
        """
        - Queremos classificar reviews sem treinar um classificador supervisionado do zero.
        - Vamos usar embeddings para representar semanticamente cada review.
        - Depois, vamos comparar esses reviews com descrições textuais das classes.
        - O enquadramento escolhido na aula é o NPS: `Promotor`, `Neutro` e `Detrator`.
        """
    )

    st.markdown("### 3. Construção da variável NPS")
    st.write(
        "Seguindo a linha da aula antiga, convertemos a nota numérica em uma categoria interpretável. "
        "Isso organiza o problema e nos dá um referencial claro para avaliar a classificação."
    )

    nps_preview = df[["Rating", "Title", "NPS"]].head(12).copy()
    st.dataframe(nps_preview, width="stretch", hide_index=True)

    st.code(
        """def NPS(rating):
    if rating >= 9:
        return "Promotor"
    elif rating >= 7:
        return "Neutro"
    else:
        return "Detrator"

dados["NPS"] = dados["Rating"].apply(NPS)""",
        language="python",
    )

    st.markdown("### 4. Descrições das classes")
    st.write(
        "No zero-shot, a qualidade das descrições influencia muito o resultado. "
        "A ideia é escrever sentenças que representem bem cada classe."
    )

    default_labels = """Promotor: espectador muito satisfeito, nota entre 9 e 10, gostou muito do filme e tende a recomendá-lo.
Neutro: espectador moderadamente satisfeito, nota entre 7 e 8, reconhece qualidades, mas sem entusiasmo forte.
Detrator: espectador insatisfeito, nota entre 1 e 6, não gostou do filme e tende a desencorajar outras pessoas."""
    label_text = st.text_area(
        "Descrições das classes (formato Classe: descrição)",
        value=default_labels,
        height=150,
        key="example_spiderman_labels",
    )

    st.code(
        """promotores = "São promotores, avaliaram o filme com nota entre 9 e 10. Gostaram muito do filme."
detratores = "São detratores, avaliaram o filme com nota entre 1 e 6. Não gostaram do filme."
neutros = "São neutros, avaliaram o filme com nota entre 7 e 8. Gostaram mais ou menos do filme." """,
        language="python",
    )

    st.markdown("### 5. Rodando a classificação por embeddings")
    c1, c2 = st.columns([1, 1])
    with c1:
        sample_size = st.slider(
            "Tamanho da amostra do exemplo",
            min_value=10,
            max_value=min(100, len(df)),
            value=40,
            step=10,
            key="example_spiderman_sample_size",
        )
    with c2:
        text_column = st.selectbox(
            "Texto usado no exemplo",
            options=["Review", "Title"],
            index=0,
            key="example_spiderman_text_column",
        )

    if st.button("Executar exemplo completo", key="btn_example_spiderman", type="primary"):
        label_map = parse_label_descriptions(label_text)
        expected_labels = ["Promotor", "Neutro", "Detrator"]
        if set(label_map.keys()) != set(expected_labels):
            st.warning("Use exatamente as classes Promotor, Neutro e Detrator.")
            return

        sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
        texts = sample[text_column].fillna("").astype(str).tolist()

        try:
            text_embeddings = embed_texts(
                client,
                model_name,
                texts,
                output_dim=output_dim,
                mode="classification",
                progress_text="Vetorizando reviews do exemplo",
            )
            class_embeddings = embed_texts(
                client,
                model_name,
                [label_map[label] for label in expected_labels],
                output_dim=output_dim,
                mode="classification",
                progress_text="Vetorizando descrições das classes",
            )
        except RuntimeError as exc:
            st.error(str(exc))
            return

        scores = cosine_similarity(text_embeddings, class_embeddings)
        top_idx = np.argmax(scores, axis=1)
        sample["Classificação"] = [expected_labels[idx] for idx in top_idx]
        sample["Confiança"] = scores.max(axis=1)
        sorted_scores = np.sort(scores, axis=1)
        sample["Margem"] = sorted_scores[:, -1] - sorted_scores[:, -2]

        accuracy = accuracy_score(sample["NPS"], sample["Classificação"])
        cm = confusion_matrix(sample["NPS"], sample["Classificação"], labels=expected_labels)
        report = pd.DataFrame(
            classification_report(
                sample["NPS"],
                sample["Classificação"],
                labels=expected_labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()

        st.code(
            """scores = cosine_similarity(text_embeddings, class_embeddings)
sample["Classificação"] = [labels[idx] for idx in np.argmax(scores, axis=1)]
taxa_acerto = (sample["NPS"] == sample["Classificação"]).mean()""",
            language="python",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Taxa de acerto", f"{accuracy:.1%}")
        m2.metric("Amostra", len(sample))
        m3.metric("Texto usado", text_column)

        st.markdown("### 6. Leitura dos resultados")
        cm_df = confusion_dataframe(expected_labels, cm)
        heatmap = (
            alt.Chart(cm_df)
            .mark_rect()
            .encode(
                x="Previsto:N",
                y="Real:N",
                color=alt.Color("Quantidade:Q", scale=alt.Scale(scheme="teals")),
                tooltip=["Real", "Previsto", "Quantidade"],
            )
            .properties(height=280)
        )
        labels = (
            alt.Chart(cm_df)
            .mark_text(color="#0f172a", fontSize=13)
            .encode(x="Previsto:N", y="Real:N", text="Quantidade:Q")
        )

        st.altair_chart(heatmap + labels, width="stretch")

        metrics_df = report.loc[expected_labels, ["precision", "recall", "f1-score"]].rename(
            columns={"precision": "Precisão", "recall": "Recall", "f1-score": "F1"}
        )
        st.dataframe(metrics_df.style.format("{:.3f}"), width="stretch")

        st.markdown("### 7. Casos em que o modelo erra")
        errors = sample[sample["NPS"] != sample["Classificação"]].copy()
        if errors.empty:
            st.success("Nenhum erro apareceu nesta amostra específica.")
        else:
            st.dataframe(
                errors[["Rating", "NPS", "Classificação", "Confiança", "Margem", text_column]].head(15),
                width="stretch",
            )

        st.markdown("### 8. Interpretação didática")
        st.markdown(
            """
            - O método é zero-shot porque não treinamos um classificador supervisionado novo.
            - O que decide a classe é a proximidade entre o review e a descrição de cada rótulo.
            - A qualidade da descrição das classes muda o comportamento do sistema.
            - A análise dos erros é tão importante quanto a taxa de acerto.
            """
        )


def main(): 
    with st.sidebar:
        st.markdown("## Configurações")
        
        # --- INÍCIO DA MUDANÇA: Campo para a chave ---
        user_api_key = st.text_input(
            "Sua Google API Key",
            type="password",
            help="Obtenha sua chave gratuita em https://aistudio.google.com/app/apikey"
        )
        # --- FIM DA MUDANÇA ---

        model_name = st.selectbox(
            "Modelo de embedding",
            options=["gemini-embedding-001", "gemini-embedding-2-preview"],
            help=(
                "`gemini-embedding-001` é voltado para texto. "
                "`gemini-embedding-2-preview` também permite comparar imagem, áudio, vídeo e PDF."
            ),
        )
        output_dim = st.select_slider(
            "Dimensão do vetor",
            options=[128, 256, 512, 768, 1536, 2048, 3072],
            value=768,
        )
        st.caption(
            "Valores didaticamente úteis: 768, 1536 ou 3072. "
            "Para dimensões menores que 3072, o app normaliza os vetores."
        )
        st.write("---")

    # --- INÍCIO DA MUDANÇA: Validação e Cliente ---
    if not user_api_key:
        st.warning("👈 Por favor, insira sua Google API Key na barra lateral para liberar o laboratório.")
        st.stop()

    client = get_client(user_api_key)
    # --- FIM DA MUDANÇA ---

    tabs = st.tabs(
        [
            "Referencial Teórico",
            "Similaridade",
            "Busca Semântica",
            "Classificação Zero-Shot",
            "Multimodal",
            "Exemplo",
        ]
    )

    with tabs[0]:
        render_theory(model_name, output_dim)
    with tabs[1]:
        render_similarity_lab(client, model_name, output_dim)
    with tabs[2]:
        render_search_lab(client, model_name, output_dim)
    with tabs[3]:
        render_zero_shot_case(client, model_name, output_dim)
    with tabs[4]:
        render_multimodal_lab(client, model_name, output_dim)
    with tabs[5]:
        render_example_page(client, model_name, output_dim)


if __name__ == "__main__":
    main()
