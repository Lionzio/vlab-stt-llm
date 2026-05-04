# Análise Experimental e Qualitativa de STT e Extração NLP no Domínio Clínico

**Projeto:** `vlab-stt-llm`
**Foco:** Avaliação do pipeline de Inteligência Artificial para Recomendação de Parâmetros e Interface de Voz.

---

## 1. Resumo da Estratégia Adotada (Abordagem Híbrida e Fail-Fast)

O desenvolvimento deste componente experimental baseou-se em uma arquitetura de duas etapas probabilísticas (STT $\rightarrow$ Extração NLP) seguidas por uma etapa determinística de consolidação rigorosa (Schema Enforcement). 

Para a extração semântica, adotamos testes A/B comparando duas estratégias:
*   **Variante V1 (Direct Schema):** Utiliza *Zero-Shot Prompting* em conjunto com a *Structured Outputs API* nativa do Gemini. Foca em baixa latência, característica fundamental para aplicações de comando de voz.
*   **Variante V2 (Chain-of-Thought):** Utiliza um prompt estruturado que obriga o modelo a gerar um bloco de raciocínio `<reasoning>` antes de emitir o JSON.

**O Diferencial — Postura Fail-Fast (Hard Rules):** 
Modelos fundacionais (LLMs) são estatísticos e propensos a alucinações (ex: inferir valores vitais fora do escopo clínico). Portanto, a solução proposta **não confia cegamente na IA**. Utilizamos o **Pydantic** (`@model_validator`) para atuar de forma determinística:
1.  **Injeção de Unidade:** Preenche unidades omitidas com valores canônicos do domínio (ex: PEEP $\rightarrow$ `cmH2O`).
2.  **Validação de Limites (Safety Bounds):** Se a IA extrai uma fração absurda ou comete um erro de escala (ex: FiO2 de 200%), o algoritmo bloqueia a ação e transita o status imediatamente para `OUT_OF_BOUNDS`, impedindo uma falha letal na ponta (equipamento).

---

## 2. Dificuldades e Nuances do Domínio Médico

A captura de voz em Unidades de Terapia Intensiva (UTIs) impõe severos desafios lexicais e semânticos:

1.  **Omissão de Unidades (Implicidade):** Médicos frequentemente dizem *"Coloca a PEEP em cinco"*. O sistema deve deduzir que a unidade natural é `cmH2O`, não `mmHg` ou `%`.
2.  **Ambiguidade Numérica:** Expressões como *"PA doze por oito"* são compreendidas como $120 \times 80 \text{ mmHg}$, mas algoritmicamente são frações matemáticas ($12/8$). A IA deve ser orientada a recusar a simplificação para um único `float` e acionar o status `REQUIRES_CLARIFICATION`.
3.  **Variabilidade Fonética e Homófonos:** Acrônimos como `PEEP` podem ser interpretados pelo STT como *"pipe"*. A camada semântica precisa de alta resiliência para consolidar essas entidades ambíguas em um único formato canônico.

---

## 3. Análise Orientada a STT (Comportamento do Modelo e Métricas)

Durante a avaliação automatizada no Mock Dataset de 10 casos, observamos o comportamento da transcrição e seu impacto direto na extração estruturada.

### 3.1 Padrões de Erro Frequentes Identificados
*   **Formatação Oscilante de Números:** O STT alterna imprevisivelmente entre numerais inteiros ("600") e numerais por extenso ("seiscentos" ou até "meia dúzia").
*   **Interpretação Fonética Errática:** Siglas técnicas desconhecidas pelo léxico padrão do modelo frequentemente geram erros de substituição acústica (ex: `FiO2` transcrito como `efiio dois`).
*   **Ruído Paralinguístico:** Tosses, hesitações ou ruídos de fundo (alarmes) são transcritos literalmente ("coff coff") ou via marcadores textuais (`[ruído]`).

### 3.2 Impacto na Extração Estruturada (Exact Match vs WER)
Neste projeto, substituímos métricas acadêmicas abstratas (Recall/F1) por métricas pragmáticas de engenharia: **Exact Match Rate** e **Parse Error Rate**.

Observou-se que um aumento isolado no *Word Error Rate (WER)* da transcrição **não degrada proporcionalmente o Exact Match** do pipeline. O impacto dos erros de STT é mitigado de forma agressiva pela alta capacidade de inferência contextual do LLM. 
Por exemplo, mesmo quando o STT gerou *"pipe"* no lugar de *"peep"*, a taxa de Parse Error permaneceu em 0% e o Exact Match foi mantido em 100%, pois o LLM inferiu corretamente o parâmetro através da semântica da frase inteira (*"coloca a pipe em cinco"*).

A perda de extração torna-se irreversível apenas em cenários de *syllable dropout* crítico (onde a palavra-chave ou o valor numérico somem fisicamente da onda sonora gravada).

---

## 4. Mitigações Aplicadas e Recomendadas

Para elevar o desempenho do pipeline e contornar os desafios supracitados, as seguintes mitigações estruturais foram implementadas:

1.  **Prompt Engineering Defensivo (Few-Shot):** O *System Instruction* das variantes V1 e V2 explicita o mapeamento de jargões, gírias clínicas e homófonos comuns para seus respectivos parâmetros canônicos.
2.  **Arquitetura Fail-Fast com Pydantic:** A injeção de regras atua como o "adulto na sala". Se o LLM alucinar no parseamento matemático, o sistema reverte a transação de forma determinística caso o dado infrinja os *Safety Bounds*, sem depender de checagens baseadas em prompt.
3.  **Normalização Lexical:** Para o cálculo das métricas acústicas base de STT (WER/CER), recomenda-se a aplicação contínua de pipelines de limpeza textual (ex: via `jiwer` transforms) para remover vírgulas e pontuações injetadas aleatoriamente pelos modelos de Whisper/Gemini, evitando falsas punições na avaliação.