# Evaluation Report — vlab-stt-llm Pipeline (A/B Testing)

**Gerado em:** 2026-05-03 18:24 UTC  
**Tempo total de execução:** 178.2s  
**Casos avaliados:** 6  

## Análise Comparativa A/B: V1 (Direct) vs V2 (Chain-of-Thought)

### Metodologia
| Aspecto | V1 — Direct Schema | V2 — Chain-of-Thought |
|---------|--------------------|-----------------------|
| Técnica | Zero-Shot + `response_schema` nativo | CoT estruturado com `<reasoning>` |
| Vantagem principal | Velocidade e previsibilidade | Robustez em casos ambíguos |

### Métricas de Transcrição STT (Compartilhadas)

| ID | Cenário | WER | CER | Cache STT |
|----|---------|:---:|:---:|:---------:|
| TC-001 | `ideal` | N/A | N/A | — |
| TC-002 | `unidade_omitida` | N/A | N/A | — |
| TC-003 | `ambiguidade_terminologica` | N/A | N/A | — |
| TC-004 | `fora_do_padrao_limites` | 83.3% | 38.5% | — |
| TC-005 | `comando_incompleto` | N/A | N/A | — |
| TC-006 | `ruido_simulado` | N/A | N/A | — |

**WER médio:** `83.3%` | **CER médio:** `38.5%`

### Avaliação de Extração (Precision, Recall, F1-Score)

As métricas abaixo avaliam a precisão dos modelos em extrair Entidades e Intenções em relação ao Gabarito (Ground Truth).

| Entidade | V1 Precision | V1 Recall | **V1 F1-Score** | V2 Precision | V2 Recall | **V2 F1-Score** |
|----------|:----------:|:---------:|:-------------:|:----------:|:---------:|:-------------:|
| **Intent** | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | **0.0%** |
| **Parameter** | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | **0.0%** |

## Análise Detalhada por Caso (Comparativo)

### TC-001 — `ideal`

**Transcrição Obtida:** ``

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `frequencia_respiratoria`| `N/A`| ❌ | `N/A`| ❌ |
| status | `OK`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** Cenário de caminho feliz. Espera-se extração perfeita sem inferências.

> ⚠️ **Erro V1:** `STTError: Cota da API Gemini excedida.`
> ⚠️ **Erro V2:** `STTError: Cota da API Gemini excedida.`

### TC-002 — `unidade_omitida`

**Transcrição Obtida:** ``

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `peep`| `N/A`| ❌ | `N/A`| ❌ |
| status | `OK_INFERRED_UNIT`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** O LLM deve inferir a unidade canônica. Unidade obtida: `N/A`.

> ⚠️ **Erro V1:** `STTError: Cota da API Gemini excedida.`
> ⚠️ **Erro V2:** `STTError: Cota da API Gemini excedida.`

### TC-003 — `ambiguidade_terminologica`

**Transcrição Obtida:** ``

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `pressao_arterial`| `N/A`| ❌ | `N/A`| ❌ |
| status | `REQUIRES_CLARIFICATION`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** Cenário de sigla ambígua ('PA'). O LLM deve mapear corretamente para pressao_arterial e pedir clarificação (12 por 8).

> ⚠️ **Erro V1:** `STTError: Timeout de 60s excedido durante a geração.`
> ⚠️ **Erro V2:** `STTError: Cota da API Gemini excedida.`

### TC-004 — `fora_do_padrao_limites`

**Transcrição Obtida:** `configurar f i u dois para 200 por cento`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `fio2`| `N/A`| ❌ | `N/A`| ❌ |
| status | `OUT_OF_BOUNDS`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** Valor inválido intencionalmente (FiO2=200%). O Pydantic deve bloquear. Status obtido: `N/A`.

> ⚠️ **Erro V1:** `Extractor retornou None.`
> ⚠️ **Erro V2:** `Extractor retornou None.`

### TC-005 — `comando_incompleto`

**Transcrição Obtida:** ``

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `iniciar_terapia` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `modo_ventilatorio`| `N/A`| ❌ | `N/A`| ❌ |
| status | `MISSING_VALUE`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** Frase interrompida. O LLM não deve alucinar parâmetros inexistentes.

> ⚠️ **Erro V1:** `STTError: Cota da API Gemini excedida.`
> ⚠️ **Erro V2:** `STTError: Cota da API Gemini excedida.`

### TC-006 — `ruido_simulado`

**Transcrição Obtida:** ``

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param  | `volume_corrente`| `N/A`| ❌ | `N/A`| ❌ |
| status | `OK`   | `N/A`   | ❌ | `N/A`   | ❌ |

> **Análise:** Artefato de ruído inserido. O LLM deve ignorar tokens espúrios e extrair o valor corretamente.

> ⚠️ **Erro V1:** `STTError: Cota da API Gemini excedida.`
> ⚠️ **Erro V2:** `STTError: Cota da API Gemini excedida.`

### Conclusão Comparativa
A abordagem **V1** é recomendada por menor latência para produção direta. A **V2** traz ganhos interpretativos para ambientes de testes e homologação rigorosa de hardware médico.