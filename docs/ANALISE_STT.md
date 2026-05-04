# Análise Experimental e Qualitativa de STT e Extração NLP no Domínio Clínico

**Projeto:** `vlab-stt-llm`
**Foco:** Avaliação do pipeline de Inteligência Artificial para Recomendação de Parâmetros e Interface de Voz.

---

## 1. Resumo da Estratégia Adotada (Abordagem Híbrida A/B)

O desenvolvimento deste componente experimental baseou-se em uma arquitetura de duas etapas probabilísticas (STT $\rightarrow$ Extração NLP) seguidas por uma etapa determinística de consolidação (Schema Enforcement). 

Para a extração semântica, adotamos testes A/B comparando duas estratégias:
*   **Variante V1 (Direct Schema):** Utiliza *Zero-Shot Prompting* em conjunto com a *Structured Outputs API* nativa do Gemini (via `response_schema`). Foca em baixa latência e aderência rígida à estrutura.
*   **Variante V2 (Chain-of-Thought):** Utiliza um prompt estruturado que obriga o modelo a gerar um bloco de raciocínio `<reasoning>` antes de emitir o JSON. Otimiza o *Recall* em contextos complexos ao dar "espaço" para o modelo raciocinar, ao custo de maior latência e consumo de tokens.

**O Diferencial — Abordagem Híbrida:** 
Independente da variante (V1 ou V2), modelos de base fundacional (LLMs) são estatísticos e propensos a alucinações (ex: inferir valores fora de limites vitais). Portanto, a solução proposta não confia cegamente na IA. Utilizamos o **Pydantic** (`@model_validator`) como uma camada de software tradicional (Hard Rules) para atuar de forma determinística:
1.  **Injeção de Unidade:** Preenche unidades omitidas com valores canônicos do domínio (ex: PEEP $\rightarrow$ `cmH2O`).
2.  **Validação de Limites (Safety Bounds):** Se a IA extrai uma fração absurda (ex: FiO2 de 200%), o algoritmo bloqueia e transita o status para `OUT_OF_BOUNDS`.

---

## 2. Dificuldades e Nuances do Domínio Médico

A captura de voz em Unidades de Terapia Intensiva (UTIs) impõe severos desafios lexicais e semânticos:

1.  **Omissão de Unidades (Implicidade):** Médicos frequentemente dizem *"Coloca a PEEP em cinco"*. O sistema deve ser capaz de deduzir que a unidade natural é `cmH2O`, não `mmHg` ou `%`.
2.  **Ambiguidade Numérica:** Expressões como *"PA doze por oito"* são clinicamente compreendidas como $120 \times 80 \text{ mmHg}$, mas algoritmicamente são frações ($12/8$). A IA deve ser orientada a recusar a simplificação para um único `float` e acionar o status `REQUIRES_CLARIFICATION`.
3.  **Variabilidade Fonética de Siglas:** Acrônimos como `FiO2` podem ser ditos como *"éfi i ó dois"*, *"fi dois"* ou *"efe i o dois"*. A etapa semântica precisa de alta robustez de normalização para consolidar as entidades em um único formato canônico.

---

## 3. Análise Orientada a STT (Comportamento do Modelo)

Durante a avaliação automatizada (`scripts/evaluate_pipeline.py`), observamos o comportamento do modelo *Gemini 2.0 Flash* atuando como STT sobre áudios sintéticos pt-BR (gerados via `gTTS`).

### 3.1 Padrões de Erro Frequentes
*   **Formatação Aleatória de Números:** Ocasionalmente o STT oscila entre numerais inteiros ("600") e por extenso ("seiscentos").
*   **Interpretação Fonética Errática:** Siglas técnicas desconhecidas pelo modelo de linguagem base frequentemente geram erros de substituição. Exemplo detectado: `FiO2` foi transcrito como `efiio dois`.
*   **Inclusão de Ruído Paralinguístico:** Tosses, hesitações ou ruídos de fundo muitas vezes são transcritos de forma literal (ex: "coff coff") ou através de marcadores descritivos (`[ruído]`), o que pode sujar a entrada da próxima camada.

### 3.2 Impacto na Extração Estruturada
O impacto dos erros do STT na extração NLP é mitigado pela capacidade de raciocínio semântico dos LLMs atuais. Observou-se que um aumento no *Word Error Rate (WER)* nem sempre destrói o *F1-Score* da extração de parâmetros. Por exemplo, mesmo quando o STT gerou *"efiio dois"*, as variantes de extração (V1 e V2) mapearam com sucesso o `parameter` para `fio2` devido ao entendimento contextual da frase inteira.

Apenas em casos de perda severa de decodificação acústica (onde o numeral desaparece da transcrição) a extração sofre perda irreversível.

---

## 4. Mitigações Aplicadas

Para elevar o desempenho do pipeline e contornar os desafios supracitados, as seguintes mitigações foram implementadas:

1.  **Normalização Lexical para Avaliação (jiwer):** Foi desenvolvido um pipeline no script `metrics.py` (via `Compose`, `RemovePunctuation`, `ToLowerCase`) para limpar as strings de referência e as hipóteses do STT antes de calcular o WER/CER. Isso evita que vírgulas sintáticas adicionadas pela IA punam erroneamente as métricas acústicas.
2.  **Prompt Engineering Direcionado:** O *System Instruction* tanto do STT quanto da extração foi enriquecido com *Few-Shot Prompting*, explicitando o mapeamento de jargões técnicos para seus parâmetros canônicos.
3.  **Arquitetura Fail-Fast com Pydantic:** A injeção pós-extração garante que o backend do equipamento médico receba sempre dados "limpos". Se a IA vacilar no mapeamento de unidades, o Pydantic entra em cena para aplicar o valor *default* seguro, ou reverte a transação se o dado for perigoso.