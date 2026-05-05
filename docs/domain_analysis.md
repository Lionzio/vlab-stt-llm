# Análise de Domínio: vlab-stt-llm

**Versão:** 1.0.0 | **Status:** Baseline de avaliação | **Classificação:** Uso interno

---

## 1. Visão geral e estratégia

### 1.1 Contexto e motivação

Profissionais de saúde em ambientes como UTIs e centros cirúrgicos interagem com equipamentos complexos que exigem atenção visual e motora. A interface por voz surge como uma alternativa para reduzir a carga cognitiva, desde que implementada com mecanismos estritos de segurança, pois qualquer erro na interpretação de parâmetros (como em um ventilador pulmonar) possui impacto direto na segurança do paciente.

O projeto `vlab-stt-llm` valida a viabilidade de capturar comandos de voz em português clínico, transcrevê-los e extrair os parâmetros médicos estruturados. A solução utiliza uma arquitetura híbrida: combina a flexibilidade dos Modelos de Linguagem (LLMs) com regras determinísticas locais para garantir a conformidade dos dados.

### 1.2 Arquitetura do pipeline

A arquitetura foi projetada para prever falhas de rede e indisponibilidade de APIs. Caso o serviço de IA retorne erros (como limite de taxa/HTTP 429), ferramentas baseadas em heurísticas offline assumem a extração.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE vlab-stt-llm                            │
│                                                                             │
│  ┌──────────┐  Áudio   ┌───────────────┐    Texto    ┌──────────────────┐   │
│  │ Entrada  │─────────▶│ STT Engine    ├────────────▶│ Extractor Engine │   │
│  │ de Voz   │          │ (IA ou Mock)  │             │ (IA ou Heurística│   │
│  └──────────┘          └───────────────┘             └────────┬─────────┘   │
│                                                               │             │
│                                                   ┌───────────▼───────────┐ │
│                                                   │ Validador Pydantic    │ │
│                                                   │ (Regras Determinísticas │
│                                                   └───────────┬───────────┘ │
│                                                               │             │
│                                                   ┌───────────▼───────────┐ │
│                                                   │   JSON Estruturado    │ │
│                                                   │   (Output Final)      │ │
│                                                   └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Etapa 1 — STT (Speech-to-Text):** Transforma o áudio em texto. Utiliza a API Gemini como via principal. Em contingência, aciona um serviço de simulação local.
* **Etapa 2 — Motor de Extração:** Interpreta a intenção do comando. O LLM mapeia o jargão clínico. A via de contingência utiliza Expressões Regulares (Regex).
* **Etapa 3 — Validação:** O Pydantic aplica regras matemáticas e clínicas de validação. Ele infere unidades ausentes e bloqueia valores fora das margens de segurança antes de liberar o payload.

---

## 2. Catálogo de comandos e regras de negócio

### 2.1 Justificativa do escopo base

O escopo do protótipo focou em três parâmetros associados à Ventilação Mecânica Invasiva (VMI), escolhidos para validar diferentes desafios do sistema:

1. **PEEP (Positive End-Expiratory Pressure):** Avalia a robustez a jargões, visto que a equipe médica frequentemente omite a unidade e utiliza adaptações fonéticas (como "pipe").
2. **FiO2 (Fração Inspirada de Oxigênio):** Avalia os limites de segurança. O sistema deve impedir que erros de transcrição resultem em valores fisicamente impossíveis (como 200%).
3. **FR (Frequência Respiratória):** Avalia a ambiguidade de unidades, que podem ser ditas de forma abreviada ("irpm") ou por extenso ("incursões por minuto").

### 2.2 Limites clínicos de segurança

As regras abaixo estão codificadas na camada do Pydantic (`PARAMETER_BOUNDS`). Valores fora desses limites são classificados como inválidos pelo sistema.

| Parâmetro Canônico | Alias Clínico Comum | Unidade Padrão | Intervalo Seguro | Risco Potencial em Caso de Erro |
|---|---|---|---|---|
| `peep` | PEEP, pipe | cmH2O | 0.0 a 25.0 | Barotrauma / Colapso Alveolar |
| `fio2` | FiO2, fi de o2 | % | 21.0 a 100.0 | Toxicidade por Oxigênio |
| `frequencia_respiratoria` | FR, respiração | irpm | 4.0 a 60.0 | Alcalose / Acidose Respiratória |
| `volume_corrente` | VC, Vt | ml | 200.0 a 800.0 | Volutrauma |
| `pressao_arterial` | PA, pressão | mmHg | Sistólica/Diastólica | Exige interpretação manual |

---

## 3. Contrato de interface (Schema)

A saída do pipeline deve seguir rigorosamente o schema abaixo (`MedicalParameterExtraction`). Ele representa o JSON consumido por um sistema de hardware (ex: CLP).

```json
{
  "intent": "ajustar_parametro",
  "parameter": "peep",
  "value": 5.0,
  "unit": "cmH2O",
  "status": "OK_INFERRED_UNIT_BY_RULE",
  "notes": "Unidade 'cmH2O' injetada por regra determinística local."
}
```

### Tratamento de status

O campo `status` guia a lógica do sistema consumidor sobre o que fazer com a instrução:

| Status | Condição | Ação Esperada |
|---|---|---|
| `OK` | Extração validada. | Encaminhar comando. |
| `OK_INFERRED_UNIT` | A IA deduziu a unidade corretamentamente. | Encaminhar comando. |
| `OK_INFERRED_UNIT_BY_RULE` | A unidade foi injetada pela validação local. | Encaminhar comando. |
| `OUT_OF_BOUNDS` | O valor extraído excede o intervalo seguro. | Bloquear operação. |
| `MISSING_VALUE` | Comando incompleto (sem número detectável). | Solicitar repetição. |
| `REQUIRES_CLARIFICATION` | Ambiguidade identificada (ex: fração de PA). | Solicitar contexto ao usuário. |

---

## 4. Engenharia e resiliência

Erros de inferência e instabilidade de rede são riscos centrais no uso de LLMs em sistemas críticos. O pipeline os mitiga através das seguintes abordagens:

1. **Limitação no prompt:** As instruções exigem que o modelo retorne `null` e o status `MISSING_VALUE` caso a frase não possua dígitos, impedindo a geração de números inexistentes para preencher lacunas.
2. **Validação estrita:** Se a IA interpretar um valor inválido, o validador Pydantic altera o status para `OUT_OF_BOUNDS`, barrando a requisição antes da resposta HTTP final.
3. **Contingência de API:** Se as tentativas de rede falharem por esgotamento de cota, o sistema direciona o texto transcrito para o extrator baseado em Regex (`HeuristicParameterExtractor`), assegurando a entrega do JSON sem tempo de inatividade para o usuário.