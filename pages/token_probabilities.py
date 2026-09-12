"""Laboratório didático de logprobs da Chat Completions API."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from ai_helpers import get_secret
from token_logprobs import parse_chat_logprobs, parse_response_logprobs


DOC_CHAT = "https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create"
DOC_RESPONSES = "https://developers.openai.com/api/reference/resources/responses/methods/create"
DOC_LOGPROBS = "https://cookbook.openai.com/examples/using_logprobs"


def request_logprobs(api_key: str, endpoint: str, model: str, prompt: str, top_k: int, max_tokens: int):
    with OpenAI(api_key=api_key, timeout=90.0) as client:
        if endpoint == "Responses API":
            response = client.responses.create(
                model=model,
                instructions="Responda diretamente, em uma única frase curta.",
                input=prompt,
                max_output_tokens=max_tokens,
                top_logprobs=top_k,
                store=False,
            )
            return response, parse_response_logprobs(response)
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Responda diretamente, em uma única frase curta."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_k,
        ), None


def replay_html(steps: list[dict], speed_ms: int) -> str:
    payload = json.dumps(steps, ensure_ascii=False).replace("</", "<\\/")
    return f"""
<div class="lab">
  <div class="controls"><button id="play">▶ Reproduzir</button><button id="prev">←</button><button id="next">→</button><span id="counter"></span></div>
  <div id="sentence" aria-live="polite"></div>
  <p class="hint">Alternativas antes da escolha do próximo token</p><div id="bars"></div>
</div>
<style>
body {{ margin:0; font-family:ui-sans-serif,system-ui; color:#e8edf3; background:transparent }}
.lab {{ padding:18px; border:1px solid #324155; border-radius:16px; background:linear-gradient(145deg,#101824,#172232) }}
.controls {{ display:flex; gap:8px; align-items:center }} button {{ color:#eef7ff;background:#24344a;border:1px solid #49617e;border-radius:8px;padding:8px 13px;cursor:pointer }}
#counter,.hint {{ color:#aab8ca;font-size:13px }} #sentence {{ min-height:66px;margin:18px 0 12px;font-size:25px;line-height:1.7 }}
.token {{ padding:3px 2px;border-radius:5px }} .active {{ background:#19c37d;color:#07170f }}
.row {{ display:grid;grid-template-columns:minmax(75px,160px) 1fr 72px;gap:10px;align-items:center;margin:8px 0 }}
.label {{ overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace }} .track {{ background:#26364b;border-radius:5px;height:23px;overflow:hidden }}
.bar {{ height:100%;min-width:2px;background:#748aa5;transition:width .35s }} .chosen .bar {{ background:#19c37d }} .pct {{ text-align:right;font-variant-numeric:tabular-nums }}
</style>
<script>
const steps={payload}; let current=0, timer=null; const speed={speed_ms};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function draw(i) {{ current=Math.max(0,Math.min(i,steps.length-1)); const step=steps[current];
 document.querySelector('#counter').textContent=`Token ${{current+1}} de ${{steps.length}}`;
 document.querySelector('#sentence').innerHTML=steps.slice(0,current+1).map((x,j)=>`<span class="token ${{j===current?'active':''}}" title="p=${{(x.probability*100).toFixed(2)}}%">${{esc(x.token)}}</span>`).join('');
 document.querySelector('#bars').innerHTML=step.alternatives.map(x=>`<div class="row ${{x.selected?'chosen':''}}"><span class="label" title="${{esc(x.token)}}">${{esc(x.label)}}</span><div class="track"><div class="bar" style="width:${{Math.max(.4,x.probability*100)}}%"></div></div><span class="pct">${{(x.probability*100).toFixed(2)}}%</span></div>`).join('');
}}
function play() {{ clearInterval(timer); current=-1; draw(0); timer=setInterval(()=>{{ if(current>=steps.length-1) clearInterval(timer); else draw(current+1); }},speed); }}
document.querySelector('#play').onclick=play; document.querySelector('#prev').onclick=()=>{{clearInterval(timer);draw(current-1)}}; document.querySelector('#next').onclick=()=>{{clearInterval(timer);draw(current+1)}}; draw(0);
</script>"""


st.title("Probabilidades do próximo token")
st.write("Veja uma geração já concluída ser reproduzida token a token — e o que o modelo poderia ter escolhido em cada passo.")

with st.expander("O que a API oferece hoje?", expanded=True):
    st.markdown(
        f"""
- **Sim, ainda existe — inclusive na API moderna:** a **Responses API** aceita `top_logprobs` e inclui os logprobs nos blocos de texto de saída. A **Chat Completions API** continua oferecendo `logprobs=true` + `top_logprobs`.
- **Não é uma divisão entre modelo “novo” e “antigo”:** o suporte depende da combinação **modelo + endpoint**. Nem todo modelo implementa todo parâmetro; um erro da API é a verificação definitiva para o modelo escolhido.
- As duas APIs aceitam até **20 alternativas** por posição, embora possam devolver menos. Este laboratório permite comparar os dois formatos e os normaliza na mesma visualização.
- `exp(logprob)` dá a probabilidade do token. As barras abaixo são probabilidades sobre o vocabulário inteiro; as alternativas visíveis podem somar menos de 100%.

Fontes oficiais: [referência de Responses]({DOC_RESPONSES}) · [referência de Chat Completions]({DOC_CHAT}) · [exemplo de logprobs]({DOC_LOGPROBS})
"""
    )

api_key = get_secret("OPENAI_API_KEY")
with st.form("token_form"):
    prompt = st.text_area("Digite uma frase ou pergunta", "Complete de modo criativo: No meio do caminho havia", height=100)
    col1, col2 = st.columns(2)
    endpoint = col1.selectbox("API", ["Responses API", "Chat Completions"])
    model = col2.text_input("Modelo", "gpt-4.1-mini")
    col3, col4 = st.columns(2)
    top_k = col3.slider("Alternativas por token", 1, 20, 5)
    max_tokens = col4.slider("Máximo de tokens", 4, 40, 20)
    submitted = st.form_submit_button("Gerar e analisar", type="primary", use_container_width=True)

if not api_key:
    st.info("Configure `OPENAI_API_KEY` nos secrets do Streamlit ou no ambiente para executar a demonstração.")

if submitted:
    if not api_key:
        st.error("A chave `OPENAI_API_KEY` não está configurada.")
    elif not prompt.strip() or not model.strip():
        st.warning("Preencha o texto e o identificador do modelo.")
    else:
        try:
            with st.spinner("Gerando e coletando logprobs..."):
                response, parsed = request_logprobs(api_key, endpoint, model.strip(), prompt.strip(), top_k, max_tokens)
                st.session_state["token_probability_run"] = {
                    "steps": parsed or parse_chat_logprobs(response),
                    "model": model.strip(),
                    "endpoint": endpoint,
                }
        except Exception as exc:
            st.error("Não foi possível obter logprobs. Confirme se o modelo selecionado aceita esse recurso na API escolhida.")
            st.code(str(exc))

run = st.session_state.get("token_probability_run")
if run:
    st.subheader("Replay da geração")
    speed = st.slider("Intervalo da animação (ms)", 250, 2000, 800, 50)
    components.html(replay_html(run["steps"], speed), height=490, scrolling=True)
    st.caption(f"Modelo usado: `{run['model']}` via {run['endpoint']}. O verde marca o token efetivamente escolhido.")

    with st.expander("Inspecionar os dados recebidos"):
        rows = [
            {"posição": s["index"] + 1, "token": s["label"], "probabilidade": s["probability"], "logprob": s["logprob"]}
            for s in run["steps"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={"probabilidade": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.download_button("Baixar JSON", json.dumps(run, ensure_ascii=False, indent=2), "token-logprobs.json", "application/json")

st.divider()
st.markdown("### Como a animação funciona")
st.write("A chamada não precisa ser transmitida ao navegador. O app recebe uma única resposta com o texto, o token escolhido, seu `logprob` e as melhores alternativas de cada posição; depois o JavaScript apenas reproduz esses dados no tempo. Isso torna a aula repetível, permite pausar e não faz novas chamadas durante o replay.")
