import streamlit as st

# Página: Sobre os Modelos
st.markdown("""
# ◾◾◾ Guia dos Melhores Modelos de IA ◾◾◾

Em 2025, a inteligência artificial evoluiu muito. As empresas agora criam modelos de IA não só mais poderosos, mas também mais eficientes e especializados. Temos desde modelos pagos das grandes empresas até versões gratuitas que qualquer pessoa pode usar. Escolher o modelo certo hoje depende do que você precisa: velocidade, qualidade ou custo.

Uma novidade interessante são os modelos "misteriosos" como o **Sonoma Sky Alpha** e **Sonoma Dusk Alpha**. Estes modelos são lançados sem revelar quem os criou ou como funcionam. A ideia é simples: as empresas querem feedback honesto dos usuários antes do lançamento oficial. Em troca de uso gratuito, elas coletam dados sobre como o modelo se comporta no mundo real.

---
""")

# Modelos por Empresa

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "◾ OpenAI", "◾ Google", "◾ X.AI", "◾ Anthropic", "◾ DeepSeek", 
    "◾ Z.AI", "◾ OpenRouter", "◾ Qwen", "◾ Moonshot AI", "◾ Mistral AI", "◾ Meta"
])

with tab1:
    st.markdown("## ◾ OpenAI")
    
    st.markdown("""
    ### ◽ GPT-5 Nano
    **Data de Lançamento:** 7 de agosto de 2025

    **Descrição:** O GPT-5 Nano é uma variante da família GPT-5, projetada para latência ultrabaixa e eficiência. Focado em experiências em tempo real e aplicações incorporadas, o Nano oferece capacidades de raciocínio ricas para tarefas de pergunta e resposta (Q&A) rápidas. Como parte do sistema unificado GPT-5, beneficia-se do router de decisão em tempo real da OpenAI, que aloca dinamicamente a complexidade computacional com base na consulta do utilizador. Embora otimizado para velocidade, mantém um forte desempenho em tarefas que não exigem o raciocínio profundo do seu homólogo maior, tornando-o ideal para agentes que necessitam de respostas rápidas e contextualmente conscientes.

    **Referência Oficial:** [https://openai.com/gpt-5](https://openai.com/gpt-5)

    ### ◽ GPT-5 Mini
    **Data de Lançamento:** 7 de agosto de 2025

    **Descrição:** O GPT-5 Mini estabelece um equilíbrio entre velocidade, custo e capacidade de raciocínio dentro da família GPT-5. Projetado para alimentar experiências em tempo real para aplicações e agentes, o Mini oferece um raciocínio sólido e capacidades de chamada de ferramentas para resolver problemas do utilizador de forma eficiente. É uma opção de custo mais baixo em comparação com o modelo GPT-5 completo, mas ainda assim representa um avanço significativo em relação às gerações anteriores. Os utilizadores do plano gratuito do ChatGPT são transicionados para o GPT-5 Mini quando atingem os seus limites de utilização do GPT-5, garantindo uma experiência de alta capacidade contínua.

    **Referência Oficial:** [https://openai.com/gpt-5](https://openai.com/gpt-5)

    ### ◽ GPT-5
    **Data de Lançamento:** 7 de agosto de 2025

    **Descrição:** O GPT-5 é o modelo de linguagem multimodal de ponta da OpenAI, unificando capacidades de raciocínio de nível de doutoramento, uma taxa de alucinação significativamente reduzida e uma vasta janela de contexto de 272.000 tokens. Projetado para tarefas "agênticas" complexas de ponta a ponta, integra todas as ferramentas da OpenAI numa única interface. Um router inteligente seleciona automaticamente o nível de raciocínio ideal, equilibrando profundidade e velocidade. O GPT-5 melhora drasticamente as competências de escrita, matemática e codificação, sendo capaz de construir aplicações completas a partir de um único prompt. Está disponível para todos os utilizadores do ChatGPT, incluindo o plano gratuito.

    **Referência Oficial:** [https://openai.com/gpt-5](https://openai.com/gpt-5)

    ### ◽ GPT-OSS 120B
    **Data de Lançamento:** 5 de agosto de 2025

    **Descrição:** O GPT-OSS 120B é um modelo de linguagem de peso aberto de última geração da OpenAI, lançado sob a licença Apache 2.0. Concebido para casos de uso de produção e de alto raciocínio, atinge uma paridade próxima com o modelo proprietário o4-mini da OpenAI em benchmarks de raciocínio principais. O modelo está otimizado para uma implementação eficiente, cabendo numa única GPU de 80 GB, como a NVIDIA H100. Demonstra fortes capacidades de uso de ferramentas e é compatível com a API Responses da OpenAI, tornando-o uma base poderosa para fluxos de trabalho agênticos personalizáveis em infraestruturas próprias.

    **Referência Oficial:** [https://openai.com/index/introducing-gpt-oss/](https://openai.com/index/introducing-gpt-oss/)

    ### ◽ GPT-OSS 20B
    **Data de Lançamento:** 5 de agosto de 2025

    **Descrição:** O GPT-OSS 20B é a variante mais pequena da série de peso aberto da OpenAI, otimizada para baixa latência e implementação em hardware de consumo ou de extremidade. Utiliza uma arquitetura de Mistura de Especialistas (MoE) com 21 mil milhões de parâmetros totais e 3,6 mil milhões de parâmetros ativos, permitindo-lhe funcionar em dispositivos com apenas 16 GB de memória. O seu desempenho é comparável ao do o3-mini da OpenAI em benchmarks comuns. Lançado sob a licença Apache 2.0, é ideal para inferência local, iteração rápida e casos de uso no dispositivo que requerem capacidades agênticas como chamada de funções e uso de ferramentas.

    **Referência Oficial:** [https://openai.com/index/introducing-gpt-oss/](https://openai.com/index/introducing-gpt-oss/)
    """)

with tab2:
    st.markdown("## ◾ Google")
    
    st.markdown("""
    ### ◽ Gemini 2.5 Flash Lite
    **Data de Lançamento:** 22 de julho de 2025

    **Descrição:** O Gemini 2.5 Flash-Lite é o modelo de raciocínio mais leve e económico da família Gemini 2.5, otimizado para latência ultrabaixa e eficiência de custos. Ideal para tarefas de alto volume e sensíveis à latência, como tradução e classificação, oferece um desempenho superior em comparação com os modelos Flash anteriores. Mantém características-chave da família Gemini 2.5, incluindo uma janela de contexto de 1 milhão de tokens, suporte multimodal e integração com ferramentas como a Pesquisa Google. Por defeito, o modo "pensante" está desativado para priorizar a velocidade, mas os programadores podem ativá-lo seletivamente para tarefas que exijam maior inteligência.

    **Referência Oficial:** [https://deepmind.google/models/gemini/flash-lite/](https://deepmind.google/models/gemini/flash-lite/)

    ### ◽ Gemini 2.5 Flash
    **Data de Lançamento:** 17 de junho de 2025

    **Descrição:** O Gemini 2.5 Flash é o modelo de trabalho da Google, projetado para equilibrar velocidade e baixo custo para tarefas diárias de alto volume, como sumarização, aplicações de chat e extração de dados. A sua variante de imagem, lançada a 26 de agosto de 2025 e apelidada de "nano-banana", introduz capacidades de geração e edição de imagens de última geração. Permite a fusão de múltiplas imagens, mantém a consistência de personagens e estilo entre gerações e suporta edição conversacional com linguagem natural. O modelo integra-se com plataformas como Adobe Firefly e Express, oferecendo um controlo criativo poderoso.

    **Referência Oficial:** [https://deepmind.google/models/gemini/flash/](https://deepmind.google/models/gemini/flash/)

    ### ◽ Gemini 2.5 Pro
    **Data de Lançamento:** 17 de junho de 2025

    **Descrição:** O Gemini 2.5 Pro é o modelo mais poderoso e capaz da Google, projetado para tarefas altamente complexas e de codificação. É um modelo de "pensamento" que raciocina antes de responder, resultando num desempenho de ponta em benchmarks de matemática, ciência e codificação. Nativamente multimodal, processa entradas de texto, áudio, imagens e vídeo e possui uma vasta janela de contexto de 1 milhão de tokens para analisar grandes conjuntos de dados. O Gemini 2.5 Pro alimenta os níveis pagos do Gemini (AI Pro e AI Ultra), oferecendo limites de utilização significativamente mais elevados do que o nível gratuito e servindo como a espinha dorsal para as aplicações mais exigentes da Google.

    **Referência Oficial:** [https://deepmind.google/models/gemini/pro/](https://deepmind.google/models/gemini/pro/)
    """)

with tab3:
    st.markdown("## ◾ X.AI")

    
    st.markdown("""
    ### ◽ Grok 3 Mini
    **Data de Lançamento:** 17 de fevereiro de 2025

    **Descrição:** O Grok 3 Mini foi lançado juntamente com o seu homólogo maior, o Grok 3, como uma alternativa mais leve e rápida para utilizadores que priorizam a velocidade em detrimento de alguma precisão. Apesar do seu tamanho reduzido, o Grok 3 Mini inclui as capacidades de raciocínio avançado do Grok 3, ativadas através do modo "Think". Este modo permite-lhe abordar problemas complexos que requerem um pensamento mais profundo. O modelo demonstra um desempenho notável em benchmarks de matemática e codificação, oferecendo uma solução económica para tarefas STEM que não necessitam de um conhecimento mundial tão vasto como o modelo principal.

    **Referência Oficial:** [https://x.ai/news/grok-3](https://x.ai/news/grok-3)

    ### ◽ Grok Code Fast
    **Data de Lançamento:** 28 de agosto de 2025

    **Descrição:** O Grok Code Fast 1 é um modelo de raciocínio rápido e económico da xAI, que se destaca na codificação agêntica. Construído de raiz com uma nova arquitetura, foi pré-treinado num corpus rico em programação e pós-treinado em tarefas de codificação do mundo real. O modelo domina ferramentas de desenvolvimento comuns como `grep`, terminal e edição de ficheiros, integrando-se perfeitamente em IDEs. Com uma janela de contexto de 256k tokens, é versátil em toda a pilha de desenvolvimento de software, especialmente em TypeScript, Python, Java, Rust, C++ e Go, sendo capaz de realizar tarefas com supervisão mínima.

    **Referência Oficial:** [https://x.ai/news/grok-code-fast-1](https://x.ai/news/grok-code-fast-1)

    ### ◽ Grok 4
    **Data de Lançamento:** 9 de julho de 2025

    **Descrição:** O Grok 4 é o modelo mais inteligente da xAI, projetado com uso nativo de ferramentas, integração de pesquisa em tempo real e capacidades multimodais (texto e imagem). Possui uma janela de contexto de 256.000 tokens, permitindo a análise de documentos longos e conversas de múltiplos turnos. O Grok 4 foi treinado em grande escala com aprendizagem por reforço no supercomputador Colossus da xAI para refinar as suas capacidades de raciocínio. Está disponível em duas versões: o Grok 4 padrão e o Grok 4 Heavy, uma arquitetura multi-agente para raciocínio colaborativo complexo, posicionando-o para competir na vanguarda da IA.

    **Referência Oficial:** [https://x.ai/news/grok-4](https://x.ai/news/grok-4)
    """)

with tab4:
    st.markdown("## ◾ Anthropic")
    
    st.markdown("""
    ### ◽ Claude 3.5 Haiku
    **Data de Lançamento:** 22 de outubro de 2024

    **Descrição:** O Claude 3.5 Haiku é o modelo mais rápido e económico da Anthropic, otimizado para casos de uso onde a velocidade e a acessibilidade são cruciais. Supera o seu antecessor em todas as competências e ultrapassa o Claude 3 Opus em muitos benchmarks de inteligência. A sua velocidade, combinada com um melhor seguimento de instruções, torna-o ideal para tarefas de sub-agentes especializados, chatbots interativos virados para o utilizador e processamento de grandes volumes de dados. Casos de uso populares incluem conclusões de código, extração e rotulagem de dados, e moderação de conteúdo em tempo real, oferecendo um desempenho robusto a um preço acessível.

    **Referência Oficial:** [https://www.anthropic.com/claude/haiku](https://www.anthropic.com/claude/haiku)

    ### ◽ Claude Sonnet 4
    **Data de Lançamento:** 22 de maio de 2025

    **Descrição:** O Claude Sonnet 4 é um modelo híbrido da Anthropic que equilibra desempenho e custo, representando uma atualização significativa em relação ao Sonnet 3.7. Oferece um desempenho de codificação superior e um raciocínio aprimorado, respondendo com maior precisão às instruções. O modelo possui dois modos: respostas quase instantâneas para tarefas rápidas e "pensamento estendido" para um raciocínio mais profundo. O Sonnet 4 destaca-se em tarefas de codificação como revisões de código e correções de bugs, e alimenta o novo agente de codificação no GitHub Copilot. É ideal para aplicações de grande volume, assistentes de IA e geração de conteúdo.

    **Referência Oficial:** [https://www.anthropic.com/news/claude-4](https://www.anthropic.com/news/claude-4)
    """)

with tab5:
    st.markdown("## ◾ DeepSeek")
    
    st.markdown("""
    ### ◽ DeepSeek V3.1 Chat
    **Data de Lançamento:** 21 de agosto de 2025

    **Descrição:** O DeepSeek V3.1 Chat representa o modo "não-pensante" do modelo híbrido V3.1. Este modo foi projetado para tarefas mais simples que requerem respostas rápidas e diretas, priorizando a velocidade e a eficiência. Embora partilhe a mesma arquitetura subjacente e a janela de contexto de 128k tokens que o seu homólogo "Reasoner", o modo Chat é ideal para aplicações sensíveis à latência, como chatbots de conversação e sumarização rápida. O modelo suporta chamadas de ferramentas neste modo, permitindo a integração com fluxos de trabalho agênticos que necessitam de interações rápidas com ferramentas externas.

    **Referência Oficial:** [https://api-docs.deepseek.com/news/news250821](https://api-docs.deepseek.com/news/news250821)

    ### ◽ DeepSeek V3.1 Reasoner
    **Data de Lançamento:** 21 de agosto de 2025

    **Descrição:** O DeepSeek V3.1 Reasoner é o modo "pensante" do modelo híbrido V3.1, otimizado para raciocínio de múltiplos passos e uso de ferramentas complexas. Este modo é ajustado para tarefas que beneficiam de uma deliberação mais profunda, alcançando um desempenho de ponta em benchmarks de codificação, matemática e lógica. Embora seja mais intensivo em termos computacionais do que o modo Chat, o Reasoner atinge a qualidade de resposta do modelo R1 anterior, mas com maior rapidez. Esta eficiência torna-o uma escolha poderosa para aplicações agênticas avançadas que exigem planeamento e execução de tarefas complexas, mantendo um rácio custo-desempenho competitivo.

    **Referência Oficial:** [https://api-docs.deepseek.com/news/news250821](https://api-docs.deepseek.com/news/news250821)
    """)

with tab6:
    st.markdown("## ◾ Z.AI")
    
    st.markdown("""
    ### ◽ GLM 4.5 Air
    **Data de Lançamento:** 28 de julho de 2025

    **Descrição:** O GLM-4.5-Air é um modelo de Mistura de Especialistas (MoE) de peso aberto da Z.ai, com 106 mil milhões de parâmetros totais. Concebido como uma alternativa mais ágil ao seu homólogo maior, oferece um desempenho excecional na sua categoria de parâmetros. O modelo possui um design "Nativo de Agente", integrando intrinsecamente capacidades de raciocínio e ação, o que lhe permite planear autonomamente tarefas de múltiplos passos. Suporta modos de raciocínio híbridos (pensante e não-pensante) e é otimizado para velocidade, com versões de alta velocidade que excedem 100 tokens/segundo. O seu licenciamento aberto e preço acessível posicionam-no como uma opção forte para empresas.

    **Referência Oficial:** [https://z.ai/blog/glm-4.5](https://z.ai/blog/glm-4.5)
    """)

with tab7:
    st.markdown("## ◾ OpenRouter")
    
    st.markdown("""
    ### ◽ Sonoma Sky Alpha
**Data de Lançamento:** N/A (Modelo Alfa Ativo)

**Descrição:** O Sonoma Sky Alpha é um modelo de fronteira camuflado, de propósito geral e com inteligência máxima, disponibilizado pela OpenRouter para recolha de feedback da comunidade. As suas características de destaque incluem uma enorme janela de contexto de 2 milhões de tokens, suporte para entradas de imagem (multimodal) e chamada de ferramentas em paralelo. Durante o seu período de teste, o modelo é de utilização gratuita. No entanto, os utilizadores devem estar cientes de que todos os prompts e conclusões são registados pelo criador do modelo para fins de feedback e treino, indicando uma fase de avaliação em larga escala no mundo real.

**Referência Oficial:** [https://openrouter.ai/openrouter/sonoma-sky-alpha](https://openrouter.ai/openrouter/sonoma-sky-alpha)

### ◽ Sonoma Dusk Alpha
**Data de Lançamento:** N/A (Modelo Alfa Ativo)

**Descrição:** O Sonoma Dusk Alpha é um modelo de fronteira camuflado, rápido e inteligente, oferecido pela OpenRouter para avaliação da comunidade. Tal como o seu homólogo "Sky", possui uma janela de contexto de 2 milhões de tokens, aceita entradas de imagem e suporta chamadas de ferramentas em paralelo, tornando-o altamente versátil para tarefas complexas. Foi concebido para ser uma opção de alto desempenho que equilibra velocidade e inteligência. A sua utilização é gratuita durante a fase de teste, com a condição de que os prompts e as respostas sejam registados para ajudar a treinar e a melhorar o modelo subjacente.

**Referência Oficial:** [https://openrouter.ai/openrouter/sonoma-dusk-alpha](https://openrouter.ai/openrouter/sonoma-dusk-alpha)
    """)

with tab8:
    st.markdown("## ◾ Qwen")
    
    st.markdown("""
    ### ◽ Qwen 3 32B
**Data de Lançamento:** 29 de abril de 2025

**Descrição:** O Qwen3-32B é um modelo denso de 32 mil milhões de parâmetros da série Qwen 3 da Alibaba, lançado sob a licença Apache 2.0. Este modelo representa um avanço significativo em desempenho, superando modelos maiores da geração anterior (Qwen2.5) em tarefas de STEM, codificação e raciocínio. Suporta modos de "Pensamento" e "Não-Pensamento", permitindo aos utilizadores equilibrar a profundidade do raciocínio com a velocidade da resposta. Com uma janela de contexto de 32k tokens, o Qwen3-32B foi concebido para ser uma ferramenta poderosa e flexível para programadores e investigadores, competindo fortemente com outros modelos de peso aberto de topo.

**Referência Oficial:** [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)
    """)

with tab9:
    st.markdown("## ◾ Moonshot AI")
    
    st.markdown("""
    ### ◽ Kimi K2
**Data de Lançamento:** N/A (Ativo em julho de 2025)

**Descrição:** O Kimi K2 é um modelo de Mistura de Especialistas (MoE) de peso aberto da Moonshot AI, com um trilião de parâmetros totais e 32 mil milhões de parâmetros ativos. Foi concebido para a geração de código e resolução de problemas agênticos, sendo capaz de decidir autonomamente quais as funções a chamar e ajustar o seu plano. Apesar da sua escala massiva, a arquitetura MoE garante eficiência, oferecendo um desempenho de ponta em benchmarks de codificação e matemática a um custo significativamente inferior ao de outros modelos de fronteira. Com uma janela de contexto de 128.000 tokens, o Kimi K2 é disponibilizado nas versões Base e Instruct.

**Referência Oficial:** [https://moonshotai.github.io/Kimi-K2/](https://moonshotai.github.io/Kimi-K2/)
    """)

with tab10:
    st.markdown("## ◾ Mistral AI")
    
    st.markdown("""
    ### ◽ Mistral Small 3.2
**Data de Lançamento:** 25 de junho de 2025

**Descrição:** O Mistral Small 3.2 é uma atualização do modelo de código aberto de 24 mil milhões de parâmetros da Mistral AI, lançado sob a licença Apache 2.0. Este lançamento foca-se em refinamentos direcionados em vez de mudanças arquitetónicas, melhorando significativamente o seguimento de instruções, reduzindo a repetição de resultados e fortalecendo a chamada de funções. Estas melhorias resultam numa ferramenta mais fiável para os programadores, especialmente em cenários de uso de ferramentas. O modelo demonstra um desempenho aprimorado em benchmarks de codificação e STEM, oferecendo uma experiência de utilizador mais limpa para tarefas que exigem precisão e fiabilidade.

**Referência Oficial:**(https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)

### ◽ Mistral Nemo
**Data de Lançamento:** 18 de julho de 2024

**Descrição:** O Mistral Nemo é um modelo de 12 mil milhões de parâmetros de última geração, construído em colaboração com a NVIDIA e lançado sob a licença Apache 2.0. Oferece uma grande janela de contexto de 128.000 tokens e um desempenho de ponta na sua categoria de tamanho em raciocínio, conhecimento mundial e precisão de codificação. O Nemo utiliza um novo tokenizador, o "Tekken", que é mais eficiente na compressão de texto e código em várias línguas. Sendo um substituto direto do Mistral 7B, é fácil de integrar em sistemas existentes, tornando a IA de fronteira mais acessível para aplicações globais e multilingues.

**Referência Oficial:** [https://mistral.ai/news/mistral-nemo](https://mistral.ai/news/mistral-nemo)
    """)

with tab11:
    st.markdown("## ◾ Meta")
    
    st.markdown("""
    ### ◽ Llama 4 Maverick
**Data de Lançamento:** 5 de abril de 2025

**Descrição:** O Llama 4 Maverick é um modelo de linguagem multimodal de alta capacidade da Meta, construído sobre uma arquitetura de Mistura de Especialistas (MoE). Possui 400 mil milhões de parâmetros totais, com 17 mil milhões de parâmetros ativos e 128 especialistas por passagem. Nativamente multimodal, suporta entradas de texto e imagem e produz texto e código em 12 línguas. O Maverick está otimizado para tarefas de visão-linguagem e comportamento de assistente, apresentando uma janela de contexto de 1 milhão de tokens. Lançado sob a Llama 4 Community License, é adequado para aplicações comerciais e de investigação que exijam uma compreensão multimodal avançada.

**Referência Oficial:** [https://www.llama.com/models/llama-4/](https://www.llama.com/models/llama-4/)

### ◽ Llama 4 Scout
**Data de Lançamento:** 5 de abril de 2025

**Descrição:** O Llama 4 Scout é um modelo de Mistura de Especialistas (MoE) da Meta, com 109 mil milhões de parâmetros totais e 17 mil milhões ativos, utilizando 16 especialistas. A sua característica de destaque é uma janela de contexto de 10 milhões de tokens, a maior disponível na indústria, desbloqueando new cases de uso em memória, personalização e análise de documentos longos. Tal como o Maverick, é nativamente multimodal, permitindo uma inteligência visual e de texto superior. O Scout foi concebido para uma operação eficiente numa pequena pegada de GPU, tornando-o uma escolha de primeira classe para implementações locais ou comerciais que necessitem de capacidades de contexto longo.

**Referência Oficial:** [https://www.llama.com/models/llama-4/](https://www.llama.com/models/llama-4/)
    """)
st.markdown("""
---


## Matriz de Características e Arquitetura de Modelos de IA de Fronteira

| Empresa | Modelo | Data de Lançamento | Tipo de Arquitetura | Parâmetros (Total / Ativos) | Janela de Contexto Máx. (Tokens) | Modalidades Suportadas | Licença / Modelo de Acesso | Diferenciador Chave |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OpenAI | GPT-5 Nano | 7 Ago 2025 | Densa | N/A | 272k | Texto, Código | Proprietário | Latência ultrabaixa para Q&A |
| OpenAI | GPT-5 Mini | 7 Ago 2025 | Densa | N/A | 272k | Texto, Código, Ferramentas | Proprietário | Equilíbrio entre velocidade e raciocínio |
| OpenAI | GPT-5 | 7 Ago 2025 | Densa | N/A | 272k | Texto, Código, Imagem, Ferramentas | Proprietário | Raciocínio de nível de doutoramento |
| Google | Gemini 2.5 Flash Lite | 22 Jul 2025 | Densa/Híbrida | N/A | 1M | Texto, Código, Imagem, Áudio, Vídeo | Proprietário | Custo-eficiência e latência ultrabaixa |
| Google | Gemini 2.5 Flash | 17 Jun 2025 | Densa | N/A | 1M | Texto, Código, Imagem, Áudio, Vídeo | Proprietário | Geração e edição de imagem SOTA |
| Google | Gemini 2.5 Pro | 17 Jun 2025 | Densa/Híbrida | N/A | 1M | Texto, Código, Imagem, Áudio, Vídeo | Proprietário | Desempenho máximo para tarefas complexas |
| X.AI | Grok 3 Mini | 17 Fev 2025 | Densa/Híbrida | N/A | 131k | Texto, Código | Proprietário | Alternativa rápida com modo "Think" |
| X.AI | Grok Code Fast | 28 Ago 2025 | MoE | 314B / N/A | 256k | Código, Ferramentas | Proprietário | Velocidade e economia para codificação agêntica |
| X.AI | Grok 4 | 9 Jul 2025 | Densa | 1.7T / 1.7T | 256k | Texto, Código, Imagem, Ferramentas | Proprietário | Pesquisa em tempo real e uso nativo de ferramentas |
| Anthropic | Claude 3.5 Haiku | 22 Out 2024 | Densa | N/A | 200k | Texto, Código, Imagem | Proprietário | Modelo mais rápido e económico |
| Anthropic | Claude Sonnet 4 | 22 Mai 2025 | Híbrida | N/A | 200k | Texto, Código, Imagem, Ferramentas | Proprietário | Raciocínio híbrido para codificação |
| DeepSeek | DeepSeek V3.1 Chat | 21 Ago 2025 | MoE/Híbrida | 671B / 37B | 128k | Texto, Código, Ferramentas | MIT License | Modo não-pensante para velocidade |
| DeepSeek | DeepSeek V3.1 Reasoner | 21 Ago 2025 | MoE/Híbrida | 671B / 37B | 128k | Texto, Código, Ferramentas | MIT License | Modo pensante para raciocínio profundo |
| Z.AI | GLM 4.5 Air | 28 Jul 2025 | MoE/Híbrida | 106B / 12B | N/A | Texto, Código, Ferramentas | Open Source | Design "Nativo de Agente" |
| OpenRouter | Sonoma Sky Alpha | N/A | Desconhecido | N/A | 2M | Texto, Imagem, Ferramentas | Camuflado (Alpha) | Contexto massivo e inteligência máxima |
| OpenRouter | Sonoma Dusk Alpha | N/A | Desconhecido | N/A | 2M | Texto, Imagem, Ferramentas | Camuflado (Alpha) | Contexto massivo, rápido e inteligente |
| Qwen | Qwen 3 32B | 29 Abr 2025 | Densa/Híbrida | 32B / 32B | 32k | Texto, Código | Apache 2.0 | Desempenho de topo em modelo denso |
| Moonshot AI | Kimi K2 | N/A | MoE | 1T / 32B | 128k | Texto, Código, Ferramentas | Open Weight | Codificação agêntica em escala de triliões de parâmetros |
| Mistral AI | Mistral Small 3.2 | 25 Jun 2025 | Densa | 24B / 24B | N/A | Texto, Código, Ferramentas | Apache 2.0 | Melhoria no seguimento de instruções |
| Mistral AI | Mistral Nemo | 18 Jul 2024 | Densa | 12B / 12B | 128k | Texto, Código, Ferramentas | Apache 2.0 | Desempenho SOTA para o seu tamanho |
| Meta | Llama 4 Maverick | 5 Abr 2025 | MoE | 400B / 17B | 1M | Texto, Imagem, Código | Llama 4 Community License | Multimodalidade nativa e 128 especialistas |
| Meta | Llama 4 Scout | 5 Abr 2025 | MoE | 109B / 17B | 10M | Texto, Imagem, Código | Llama 4 Community License | Janela de contexto de 10M tokens |
| OpenAI | GPT-OSS 120B | 5 Ago 2025 | MoE | 117B / 5.1B | 131k | Texto, Código, Ferramentas | Apache 2.0 | Raciocínio de peso aberto de produção |
| OpenAI | GPT-OSS 20B | 5 Ago 2025 | MoE | 21B / 3.6B | 131k | Texto, Código, Ferramentas | Apache 2.0 | MoE eficiente para dispositivos de extremidade |
""")