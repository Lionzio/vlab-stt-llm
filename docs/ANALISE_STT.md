# Análise experimental de STT e extração NLP no domínio clínico

**Projeto:** `vlab-stt-llm`

## 1. Estratégia adotada

A arquitetura do pipeline combina duas etapas probabilísticas (STT → Extração NLP) com uma etapa determinística final (Schema Enforcement). O dado nunca chega ao sistema consumidor sem passar pela validação rígida do Pydantic — independentemente de ter vindo da IA ou do extrator de fallback.

Para a extração semântica, comparamos duas estratégias via teste A/B:

**V1 (Direct Schema):** Zero-Shot Prompting com `response_schema` nativo do Gemini. O modelo retorna JSON estruturado diretamente, sem etapas intermediárias. Foco em baixa latência, essencial para aplicações de comando de voz em tempo real.

**V2 (Chain-of-Thought):** O prompt exige que o modelo produza um bloco `<reasoning>` antes de preencher o JSON. O modelo explica em texto livre cada decisão (identificação da intenção, mapeamento do parâmetro, inferência da unidade, verificação de limites) antes de commitar a resposta estruturada. Isso aumenta a interpretabilidade e a robustez em cenários ambíguos, mas eleva latência e consumo de tokens.

### O diferencial determinístico

LLMs são modelos estatísticos — eles inferem o que é provável, não o que é seguro. Em domínios críticos, isso é inaceitável sem uma rede de segurança. O `@model_validator` do Pydantic atua como essa rede:

- **Injeção de unidade:** se o modelo retornar PEEP sem unidade, o validator injeta `cmH2O` pela tabela canônica do domínio e sinaliza `OK_INFERRED_UNIT_BY_RULE`.
- **Safety bounds:** se o modelo aceitar FiO2 de 200%, o validator força `OUT_OF_BOUNDS` — sem depender de instrução de prompt que o modelo pode ignorar.

---

## 2. Desafios do domínio clínico

A captura de voz em UTI impõe problemas que não existem em contextos genéricos:

**Omissão de unidades:** médicos dizem "coloca a PEEP em cinco" assumindo que o sistema sabe que a unidade é `cmH2O`. O sistema precisa inferir isso pelo contexto, não exigir que o usuário seja explícito.

**Ambiguidade em frações:** "PA doze por oito" é pressão arterial 120/80 mmHg. Matematicamente é a fração 12/8. O sistema deve se recusar a resolver a ambiguidade como um único `float` e acionar `REQUIRES_CLARIFICATION`.

**Homófonos e siglas:** `PEEP` pode ser transcrito como "pipe" pelo STT — um erro fonético que ocorreu no TC-002 dos testes. A camada semântica precisa mapear ambos para o mesmo parâmetro canônico.

---

## 3. Comportamento do STT e métricas

### Padrões de erro identificados

O STT (Gemini Flash rodando sobre áudios sintéticos do gTTS) apresentou três padrões de erro consistentes:

- **Formatação de números:** alternância imprevisível entre numeral ("600") e extenso ("seiscentos"). O TC-006 recebeu "600" onde o gabarito esperava "seiscentos".
- **Siglas técnicas:** `FiO2` foi transcrito como "efiio dois" no TC-004 — erro de substituição acústica em sigla fora do vocabulário padrão do modelo.
- **Ruído paralinguístico:** tosses e ruídos de fundo aparecem literalmente ("coff coff") ou como marcadores textuais.

### WER alto não implica falha de extração

O resultado mais relevante do benchmark é que WER e taxa de extração correta não se correlacionam linearmente. No TC-002, o STT transcreveu "pipe" no lugar de "peep" — WER de 20% — mas o LLM extraiu corretamente `parameter=peep, unit=cmH2O, status=OK_INFERRED_UNIT`. No TC-004, o STT gerou "efiio dois" para FiO2 — WER de 33% — mas o LLM mapeou para `fio2` e o Pydantic bloqueou o valor 200% como `OUT_OF_BOUNDS`.

A extração só falha de forma irreversível quando a palavra-chave ou o valor numérico desaparecem completamente da onda sonora — o chamado *syllable dropout* crítico.

---

## 4. Mitigações implementadas

**Prompt defensivo com few-shot:** o system instruction das variantes V1 e V2 inclui exemplos explícitos de mapeamento de jargões clínicos e homófonos comuns (`pipe → peep`, `f i o dois → fio2`). Isso reduz erros de substituição semântica sem depender do vocabulário padrão do modelo.

**Validação determinística via Pydantic:** a validação roda localmente, sem chamada de rede. Se o LLM inferir uma frequência respiratória de 150 irpm, o Pydantic intercepta e força `OUT_OF_BOUNDS` — a decisão não passa por probabilidade.

**Normalização textual para métricas:** o cálculo de WER/CER usa o pipeline `jiwer` com `RemovePunctuation` e `ToLowerCase` antes da comparação, evitando que pontuação injetada aleatoriamente pelos modelos penalize artificialmente os scores.