"""Visualização token a token usando Chat Completions da OpenAI."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from ai_helpers import get_secret
from token_logprobs import parse_chat_logprobs


MODEL_PRESETS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4o"]


def request_logprobs(
    api_key: str, model: str, prompt: str, top_k: int, max_tokens: int, temperature: float
):
    with OpenAI(api_key=api_key, timeout=90.0) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Responda diretamente, em uma única frase curta."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_k,
            temperature=temperature,
        )
        choice = response.choices[0]
        return parse_chat_logprobs(response), {
            "finish_reason": getattr(choice, "finish_reason", None) or "desconhecido",
            "output_text": getattr(choice.message, "content", "") or "",
            "raw_response": response.model_dump(mode="json"),
        }


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

st.markdown(
    "**Tokens e geração.** Tokens são unidades de texto: podem representar uma palavra, "
    "parte dela, pontuação ou espaços. Os tokens produzidos na resposta são chamados de "
    "*completion tokens*. A cada passo, o modelo atribui pontuações (*logits*) aos possíveis "
    "próximos tokens, considerando o prompt e tudo o que já gerou. Essas pontuações são "
    "convertidas em probabilidades; na amostragem, um token é sorteado segundo essa distribuição "
    "e incorporado ao contexto para calcular o próximo passo. Assim, um token com 60% de "
    "probabilidade tem mais chance de sair, mas não é uma escolha obrigatória. "
    "[Sobre tokens](https://help.openai.com/en/articles/4936856)."
)
st.markdown(
    "**Temperatura.** Para uma temperatura $T>0$, dividimos cada logit $z_i$ por $T$ "
    "antes de aplicar a função softmax, conforme a equação abaixo, em que $V$ é o "
    "vocabulário e $p_i(T)$ é a probabilidade do token $i$. Com $T=1$, temos a distribuição "
    "original; com $0<T<1$, ela fica mais concentrada nos tokens mais prováveis; com $T>1$, "
    "fica mais uniforme, dando mais chance às alternativas menos prováveis. A temperatura "
    "preserva a ordem dos logits, mas altera as chances relativas dos tokens. "
    "[Referência sobre temperatura](https://huggingface.co/docs/transformers/v4.40.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper)."
)
st.latex(r"p_i(T)=\frac{\exp(z_i/T)}{\sum_{j\in V}\exp(z_j/T)},\qquad T>0")
st.markdown(
    "**Da distribuição à escolha.** Pense em uma roleta com uma faixa para cada token, "
    "proporcional à sua probabilidade: a temperatura muda o tamanho dessas faixas antes "
    "do sorteio. No limite $T\\to0^+$, a distribuição se concentra nos maiores logits; "
    "o ajuste $T=0$ é tratado separadamente para favorecer a escolha do token mais provável, "
    "sem dividir por zero. Temperaturas maiores podem trazer mais variedade, mas não "
    "garantem qualidade ou correção. No replay, o verde indica o token escolhido e as "
    "porcentagens são calculadas por $p=\\exp(\\mathrm{logprob})$ a partir dos valores "
    "retornados pela API. “Alternativas por token” controla quantas alternativas são "
    "exibidas, não restringe o sorteio a elas; por isso, as barras podem somar menos de 100%."
)

api_key = get_secret("OPENAI_API_KEY")
with st.form("token_form"):
    prompt = st.text_area("Digite uma frase ou pergunta", "Complete de modo criativo: No meio do caminho havia", height=100)
    model = st.selectbox("Modelo", MODEL_PRESETS)
    col3, col4, col5 = st.columns(3)
    top_k = col3.slider("Alternativas por token", 1, 20, 5)
    max_tokens = col4.slider("Máximo de tokens", 4, 40, 20)
    temperature = col5.slider(
        "Temperatura",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Valores menores concentram a distribuição; valores maiores aumentam a variedade.",
    )
    submitted = st.form_submit_button("Gerar e analisar", type="primary", use_container_width=True)

if not api_key:
    st.info("Configure `OPENAI_API_KEY` nos secrets do Streamlit ou no ambiente para executar a demonstração.")

if submitted:
    if not api_key:
        st.error("A chave `OPENAI_API_KEY` não está configurada.")
    elif not prompt.strip():
        st.warning("Preencha o texto.")
    else:
        try:
            with st.spinner("Gerando e coletando logprobs..."):
                steps, metadata = request_logprobs(
                    api_key, model, prompt.strip(), top_k, max_tokens, temperature
                )
                st.session_state["token_probability_run"] = {
                    "steps": steps,
                    "model": model,
                    "temperature": temperature,
                    **metadata,
                }
        except Exception as exc:
            st.error("Não foi possível obter logprobs com o modelo selecionado.")
            st.code(str(exc))

run = st.session_state.get("token_probability_run")
if run:
    st.subheader("Replay da geração")
    speed = st.slider("Intervalo da animação (ms)", 250, 2000, 800, 50)
    components.html(replay_html(run["steps"], speed), height=490, scrolling=True)
    st.caption(
        f"Modelo: `{run['model']}` · API: Chat Completions · temperatura: {run.get('temperature', 1.0):.1f} · "
        f"término: `{run.get('finish_reason', 'não registrado')}`. O verde marca o token efetivamente escolhido."
    )
    if run.get("finish_reason") in {"length", "max_output_tokens"}:
        st.warning("A resposta atingiu o limite de tokens e pode ter sido cortada. Aumente “Máximo de tokens”.")
    else:
        st.success(f"Todos os {len(run['steps'])} tokens textuais retornados pela API aparecem no replay.")

    with st.expander("JSON completo da resposta"):
        st.json(run["raw_response"], expanded=False)
        st.download_button(
            "Baixar JSON",
            json.dumps(run["raw_response"], ensure_ascii=False, indent=2),
            "chat-completion-logprobs.json",
            "application/json",
        )
