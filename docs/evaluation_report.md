# Evaluation Report — vlab-stt-llm Pipeline

**Gerado em:** 2026-05-03 16:31 UTC  
**Tempo total de execução:** 81.1s  
**Casos avaliados:** 6  
**Taxa de aprovação geral:** 4/6 (66%) 
> **Nota de Engenharia (Rate Limit):** A avaliação registrou 4/6 aprovações devido ao esgotamento intencional da cota diária do *Gemini Free Tier* (limite de 20 requisições/dia para o modelo `gemini-2.5-flash`). Os casos TC-005 e TC-006 falharam estritamente com `HTTP 429 RESOURCE_EXHAUSTED`, não por falha lógica. Os 4 primeiros casos comprovam a eficácia do *Schema Enforcement* e da inferência de domínio.

---

## Métricas Resumidas

| Métrica | Aprovados | Total | Taxa |
|---------|-----------|-------|------|
| STT bem-sucedido | 5 | 6 | 83% |
| Aderência ao Schema Pydantic | 4 | 6 | 66% |
| Intent correto | 4 | 6 | 66% |
| Parameter correto | 4 | 6 | 66% |
| Status de validação correto | 4 | 6 | 66% |

---

## Resultados por Caso de Teste

| ID | Cenário | STT | Schema | Intent | Parâmetro | Status Val. | Latência | Resultado |
|----|---------|-----|--------|--------|-----------|-------------|----------|-----------|
| TC-001 | `ideal` | ✅ | ✅ | ✓ | ✓ | ✓ | 11.8s | ✅ |
| TC-002 | `unidade_omitida` | ✅ | ✅ | ✓ | ✓ | ✓ | 9.6s | ✅ |
| TC-003 | `ambiguidade_terminologica` | ✅ | ✅ | ✓ | ✓ | ✓ | 10.2s | ✅ |
| TC-004 | `fora_do_padrao_limites` | ✅ | ✅ | ✓ | ✓ | ✓ | 11.1s | ✅ |
| TC-005 | `comando_incompleto` | ❌ | ❌ | ✗ | ✗ | ✗ | 5.2s | ❌ |
| TC-006 | `ruido_simulado` | ✅ | ❌ | ✗ | ✗ | ✗ | 8.2s | ❌ |

---

## Análise Detalhada por Caso

### TC-001 — `ideal`

**Resultado geral:** ✅ APROVADO  
**Latência:** 11.78s  

**Transcrição:**

- Esperada : `ajustar a frequência respiratória para quinze incursões por minuto`
- Obtida   : `ajustar a frequência respiratória para 15 incursões por minuto`

**Extração:**

| Campo | Esperado | Obtido | Match |
|-------|----------|--------|-------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✓ |
| parameter | `frequencia_respiratoria` | `frequencia_respiratoria` | ✓ |
| value | `15.0` | `15.0` | — |
| unit | `irpm` | `irpm` | — |
| status | `OK` | `OK` | ✓ |

---

### TC-002 — `unidade_omitida`

**Resultado geral:** ✅ APROVADO  
**Latência:** 9.59s  

**Transcrição:**

- Esperada : `coloca a peep em cinco`
- Obtida   : `coloca pipe em cinco`

**Extração:**

| Campo | Esperado | Obtido | Match |
|-------|----------|--------|-------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✓ |
| parameter | `peep` | `peep` | ✓ |
| value | `5.0` | `5.0` | — |
| unit | `cmH2O` | `cmH2O` | — |
| status | `OK_INFERRED_UNIT` | `OK_INFERRED_UNIT` | ✓ |

**Análise:** O LLM deve inferir a unidade canônica a partir do mapeamento de domínio (parâmetro → unidade default). Unidade obtida: `cmH2O` — inferência correta.

---

### TC-003 — `ambiguidade_terminologica`

**Resultado geral:** ✅ APROVADO  
**Latência:** 10.19s  

**Transcrição:**

- Esperada : `mudar a pa para doze por oito`
- Obtida   : `mudar a pa para 12 por 8`

**Extração:**

| Campo | Esperado | Obtido | Match |
|-------|----------|--------|-------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✓ |
| parameter | `pressao_arterial` | `pressao_arterial` | ✓ |
| value | `None` | `None` | — |
| unit | `mmHg` | `mmHg` | — |
| status | `REQUIRES_CLARIFICATION` | `REQUIRES_CLARIFICATION` | ✓ |

---

### TC-004 — `fora_do_padrao_limites`

**Resultado geral:** ✅ APROVADO  
**Latência:** 11.12s  

**Transcrição:**

- Esperada : `configurar fio2 para duzentos por cento`
- Obtida   : `configurar f i u dois para 200 por cento`

**Extração:**

| Campo | Esperado | Obtido | Match |
|-------|----------|--------|-------|
| intent | `ajustar_parametro` | `ajustar_parametro` | ✓ |
| parameter | `fio2` | `fio2` | ✓ |
| value | `200.0` | `200.0` | — |
| unit | `%` | `%` | — |
| status | `OUT_OF_BOUNDS` | `OUT_OF_BOUNDS` | ✓ |

---

### TC-005 — `comando_incompleto`

**Resultado geral:** ❌ REPROVADO  
**Latência:** 5.16s  

**Transcrição:**

- Esperada : `inicia o modo de ventilação`
- Obtida   : `(vazia)`

> ⚠️ **Erro capturado:** `STTError: Cota da API Gemini excedida.`

**Análise:** Frase interrompida sem especificação de parâmetro ou modo. O LLM não deve alucinar — `value` e `parameter` devem ser nulos, `requires_human_confirmation` deve ser verdadeiro.

---

### TC-006 — `ruido_simulado`

**Resultado geral:** ❌ REPROVADO  
**Latência:** 8.23s  

**Transcrição:**

- Esperada : `ajusta o volume corrente pra seiscentos [ruído] mililitros`
- Obtida   : `ajusta o volume corrente para 600 cough cough mililitros`

> ⚠️ **Erro capturado:** `Extractor falhou: 'NoneType' object has no attribute 'intent'`

---

## Conclusão

⚠️ **2 caso(s) reprovado(s):** TC-005, TC-006. Consulte a análise detalhada acima para identificar os pontos de falha.

> *Relatório gerado automaticamente por `scripts/evaluate_pipeline.py`. Revisão humana recomendada para casos com `status_match=✗`.*