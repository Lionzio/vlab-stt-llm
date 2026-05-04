# Evaluation Report — vlab-stt-llm Pipeline (A/B Testing)

**Gerado em:** 2026-05-04 12:32 UTC  
**Tempo total de execução:** 5.4s  
**Casos avaliados:** 6  
**Semaphore limit:** 2 req paralelas  

## Análise Comparativa A/B: V1 (Direct) vs V2 (Chain-of-Thought)

### Metodologia

| Aspecto | V1 — Direct Schema | V2 — Chain-of-Thought |
|---------|--------------------|-----------------------|
| Técnica | Zero-Shot + `response_schema` nativo | CoT estruturado com `<reasoning>` |
| Vantagem principal | Velocidade e previsibilidade | Robustez em casos ambíguos |

### Métricas de Transcrição STT (Compartilhadas)

| ID | Cenário | WER | CER | Cache STT |
|----|---------|:---:|:---:|:---------:|
| TC-001 | `ideal` | 0.0% | 0.0% | ✓ |
| TC-002 | `unidade_omitida` | 20.0% | 9.1% | ✓ |
| TC-003 | `ambiguidade_terminologica` | 0.0% | 0.0% | ✓ |
| TC-004 | `fora_do_padrao_limites` | 33.3% | 17.9% | ✓ |
| TC-005 | `comando_incompleto` | 0.0% | 0.0% | ✓ |
| TC-006 | `ruido_simulado` | 50.0% | 26.8% | ✓ |

**WER médio:** `17.2%` | **CER médio:** `9.0%`

### Avaliação de Extração (Precision, Recall, F1-Score)

As métricas abaixo avaliam a precisão dos modelos em extrair Entidades e Intenções em relação ao Gabarito (Ground Truth).

| Entidade | V1 Precision | V1 Recall | **V1 F1-Score** | V2 Precision | V2 Recall | **V2 F1-Score** |
|----------|:----------:|:---------:|:-------------:|:----------:|:---------:|:-------------:|
| **Intent** | 100.0% | 50.0% | **66.7%** | 100.0% | 50.0% | **66.7%** |
| **Parameter** | 100.0% | 50.0% | **66.7%** | 100.0% | 50.0% | **66.7%** |

## Análise Detalhada por Caso (Comparativo)

### TC-001 — `ideal`

**Transcrição Obtida:** `ajustar a frequência respiratória para quinze incursões por minuto`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✅ | `ajustar_parametro` | ✅ |
| param | `frequencia_respiratoria` | `frequencia_respiratoria` | ✅ | `frequencia_respiratoria` | ✅ |
| status | `OK` | `OK` | ✅ | `OK` | ✅ |

> **Análise:** Cenário de caminho feliz. Espera-se extração perfeita sem inferências.

### TC-002 — `unidade_omitida`

**Transcrição Obtida:** `coloca peep em cinco`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param | `peep` | `N/A` | ❌ | `N/A` | ❌ |
| status | `OK_INFERRED_UNIT` | `N/A` | ❌ | `N/A` | ❌ |

> **Análise:** O LLM deve inferir a unidade canônica. Unidade obtida: `N/A`.

> ⚠️ **Erro V1:** `Extractor retornou None.`
> ⚠️ **Erro V2:** `Extractor retornou None.`

### TC-003 — `ambiguidade_terminologica`

**Transcrição Obtida:** `mudar a pa para doze por oito`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✅ | `ajustar_parametro` | ✅ |
| param | `pressao_arterial` | `pressao_arterial` | ✅ | `pressao_arterial` | ✅ |
| status | `REQUIRES_CLARIFICATION` | `REQUIRES_CLARIFICATION` | ✅ | `REQUIRES_CLARIFICATION` | ✅ |

> **Análise:** Cenário de sigla ambígua ('PA'). O LLM deve mapear corretamente para pressao_arterial e pedir clarificação (12 por 8).

### TC-004 — `fora_do_padrao_limites`

**Transcrição Obtida:** `configurar efiio dois para duzentos por cento`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param | `fio2` | `N/A` | ❌ | `N/A` | ❌ |
| status | `OUT_OF_BOUNDS` | `N/A` | ❌ | `N/A` | ❌ |

> **Análise:** Valor inválido intencionalmente (FiO2=200%). O Pydantic deve bloquear. Status obtido: `N/A`.

> ⚠️ **Erro V1:** `Extractor retornou None.`
> ⚠️ **Erro V2:** `Extractor retornou None.`

### TC-005 — `comando_incompleto`

**Transcrição Obtida:** `inicia o modo de ventilação`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `iniciar_terapia` | `iniciar_terapia` | ✅ | `iniciar_terapia` | ✅ |
| param | `modo_ventilatorio` | `modo_ventilatorio` | ✅ | `modo_ventilatorio` | ✅ |
| status | `MISSING_VALUE` | `MISSING_VALUE` | ✅ | `MISSING_VALUE` | ✅ |

> **Análise:** Frase interrompida. O LLM não deve alucinar parâmetros inexistentes.

### TC-006 — `ruido_simulado`

**Transcrição Obtida:** `ajusta o volume corrente para 600 coff coff mililitros`

| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |
|-------|----------|-------------|----------|-------------|----------|
| intent | `ajustar_parametro` | `N/A` | ❌ | `N/A` | ❌ |
| param | `volume_corrente` | `N/A` | ❌ | `N/A` | ❌ |
| status | `OK` | `N/A` | ❌ | `N/A` | ❌ |

> **Análise:** Artefato de ruído inserido. O LLM deve ignorar tokens espúrios e extrair o valor corretamente.

> ⚠️ **Erro V1:** `Extractor retornou None.`
> ⚠️ **Erro V2:** `Extractor retornou None.`

### Conclusão Comparativa
A abordagem **V1** é recomendada por menor latência para produção direta. A **V2** traz ganhos interpretativos para ambientes de testes e homologação rigorosa de hardware médico.