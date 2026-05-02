```markdown
# Domain Analysis: vlab-stt-llm
**Versão:** 0.1.0 | **Status:** Documento Vivo | **Classificação:** Uso Interno — Protótipo Experimental

---

## 1. Visão Geral e Estratégia

### 1.1 Contexto e Motivação

Profissionais de saúde em ambientes de alta pressão — UTIs, centros cirúrgicos, emergências — interagem constantemente com equipamentos clínicos complexos via interfaces físicas que exigem atenção visual e motora. A interação por voz representa uma oportunidade de reduzir a carga cognitiva e aumentar a velocidade de resposta em cenários críticos, desde que implementada com rigor clínico e de engenharia.

O `vlab-stt-llm` é um pipeline experimental que valida a viabilidade técnica de capturar **comandos de voz em português clínico**, transcrevê-los e extrair parâmetros médicos estruturados com semântica validada, prontos para serem consumidos por sistemas de controle de equipamentos.

### 1.2 Arquitetura do Pipeline em Duas Etapas

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE vlab-stt-llm                             │
│                                                                            │
│  ┌──────────┐    Audio     ┌─────────────┐  Transcrição  ┌─────────────┐  │
│  │  Entrada │─────────────▶│  STT Engine │──────────────▶│ LLM + Few-  │  │
│  │  de Voz  │   (PCM/WAV)  │ (Whisper/   │   (texto bruto│ Shot Prompt │  │
│  └──────────┘              │  cloud API) │    + confiança│ + Schema    │  │
│                            └─────────────┘               │ Enforcement │  │
│                                                           └──────┬──────┘  │
│                                                                  │         │
│                                                     ┌────────────▼───────┐ │
│                                                     │  Pydantic Validator│ │
│                                                     │  (Schema Estrito)  │ │
│                                                     └────────────┬───────┘ │
│                                                                  │         │
│                                                     ┌────────────▼───────┐ │
│                                                     │  JSON Estruturado  │ │
│                                                     │  (Output Final)    │ │
│                                                     └────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

**Etapa 1 — STT (Speech-to-Text):** Responsável pela transcrição do áudio bruto em texto. Não possui conhecimento de domínio. Entrega texto com possíveis erros fonéticos, ruído e coloquialismos clínicos. O score de confiança da transcrição é propagado como metadado para a etapa seguinte.

**Etapa 2 — LLM com Few-Shot Prompting e Schema Enforcement:** O LLM atua como o motor de compreensão semântica de domínio. Recebe o texto transcrito e um prompt cuidadosamente engenheirado com exemplos (few-shot) de pares `transcrição → JSON`. A saída é forçada a conformar com um schema Pydantic rigoroso, rejeitando ou sinalizando qualquer output que viole as regras de domínio (intervalos válidos, unidades canônicas, intenções reconhecidas).

### 1.3 Premissas do Protótipo

- Idioma primário: **Português Brasileiro** (registro clínico, incluindo termos técnicos e coloquialismos de UTI)
- Escopo inicial de equipamentos: **ventiladores mecânicos** e **monitores multiparamétricos**
- Não há tomada de ação autônoma nos equipamentos nesta fase — o pipeline produz um payload estruturado para revisão e confirmação humana
- Toda saída com `confidence_score < threshold` configurável é escalada para revisão manual

---

## 2. Catálogo de Comandos e Artefatos (Dataset Base)

### 2.1 Taxonomia de Intenções

| ID | Intenção (`intent`) | Descrição Funcional | Equipamento Alvo |
|----|---------------------|---------------------|------------------|
| I-01 | `ajustar_parametro` | Modificar o valor de um parâmetro clínico em equipamento ativo | Ventilador, Monitor |
| I-02 | `iniciar_terapia` | Iniciar um protocolo ou modo de terapia respiratória/hemodinâmica | Ventilador |
| I-03 | `silenciar_alarme` | Silenciar ou pausar temporariamente um alarme em equipamento | Ventilador, Monitor, Bomba |
| I-04 | `consultar_status` | Consultar o valor corrente de um parâmetro ou estado do equipamento | Ventilador, Monitor |
| I-05 | `definir_limite_alarme` | Configurar os limites superior/inferior de disparo de alarme | Monitor, Ventilador |
| I-06 | `pausar_ventilacao` | Solicitar uma pausa inspiratória ou expiratória para manobra | Ventilador |
| I-07 | `registrar_observacao` | Ditar uma nota clínica associada ao estado do paciente | Sistema de registro |

### 2.2 Catálogo de Parâmetros por Intenção

#### I-01 — `ajustar_parametro`

| Parâmetro | Sigla/Alias | Valor Típico (Adulto) | Unidade Canônica | Intervalo Válido | Equipamento |
|-----------|-------------|----------------------|------------------|------------------|-------------|
| Pressão Expiratória Final Positiva | PEEP | 5 – 10 | `cmH2O` | 0 – 25 | Ventilador |
| Fração Inspirada de Oxigênio | FiO2 | 40 – 60 | `%` | 21 – 100 | Ventilador |
| Pressão Inspiratória | Pi / P-insp | 15 – 25 | `cmH2O` | 5 – 50 | Ventilador |
| Volume Corrente | VC / Vt | 400 – 600 | `mL` | 200 – 800 | Ventilador |
| Frequência Respiratória | FR | 12 – 20 | `rpm` | 4 – 60 | Ventilador |
| Pressão Arterial Média | PAM | 70 – 100 | `mmHg` | 40 – 160 | Monitor |
| Taxa de Infusão | — | 5 – 50 | `mL/h` | 0 – 999 | Bomba |
| Temperatura Alvo (hipotermia) | T-alvo | 33 – 36 | `°C` | 30 – 40 | Monitor |

#### I-02 — `iniciar_terapia`

| Protocolo/Modo | Alias Clínico Comum | Parâmetros Associados Obrigatórios |
|----------------|---------------------|------------------------------------|
| Ventilação Protetora | — | VC ≤ 6 mL/kg de peso predito, PEEP, FiO2 |
| Modo Pressão Controlada | PCV, P-controlada | Pi, FR, PEEP, FiO2 |
| Modo Volume Controlado | VCV, V-controlada | VC, FR, PEEP, FiO2 |
| Modo Espontâneo com Suporte | PSV, pressão-suporte | Pressão de suporte, PEEP |
| Manobra de Recrutamento | — | Pressão de plateau, duração |

#### I-03 — `silenciar_alarme`

| Parâmetro | Descrição | Duração Padrão | Unidade |
|-----------|-----------|----------------|---------|
| `duration` | Tempo de silenciamento | 120 | `s` |
| `alarm_type` | Tipo de alarme (opcional, inferível) | — | `enum` |

#### I-04 — `consultar_status`

| Parâmetro Consultável | Unidade Retornada |
|-----------------------|-------------------|
| PEEP atual | `cmH2O` |
| FiO2 atual | `%` |
| SpO2 | `%` |
| Frequência Cardíaca | `bpm` |
| Pressão Arterial | `mmHg` (sistólica/diastólica/média) |
| Capnografia (EtCO2) | `mmHg` |

#### I-05 — `definir_limite_alarme`

| Parâmetro | Campo Obrigatório | Unidade |
|-----------|-------------------|---------|
| Limite superior FC | `high` | `bpm` |
| Limite inferior SpO2 | `low` | `%` |
| Limite superior FR | `high` | `rpm` |
| Limite PAM | `high` / `low` | `mmHg` |

---

## 3. Schema de Saída Esperado

O payload abaixo representa o contrato de interface entre o pipeline de IA e os sistemas consumidores. Toda saída do LLM é validada contra este schema via Pydantic antes de ser publicada.

```json
{
  "schema_version": "1.0",
  "pipeline_run_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "timestamp_utc": "2025-06-10T14:32:00.123Z",

  "stt_output": {
    "raw_transcript": "bota a PEEP em cinco",
    "confidence_score": 0.94,
    "language": "pt-BR",
    "audio_duration_s": 2.1,
    "flagged_segments": []
  },

  "llm_extraction": {
    "intent": "ajustar_parametro",
    "parameter": "PEEP",
    "parameter_canonical": "pressao_expiratoria_final_positiva",
    "value": 5,
    "unit": "cmH2O",
    "unit_was_inferred": true,
    "target_device": "ventilador_mecanico",
    "raw_value_expression": "cinco"
  },

  "validation": {
    "status": "valid",
    "within_clinical_range": true,
    "range_reference": { "min": 0, "max": 25, "unit": "cmH2O" },
    "violations": [],
    "requires_human_confirmation": false
  },

  "metadata": {
    "model_id": "gpt-4o",
    "prompt_version": "few_shot_v2.1",
    "extraction_latency_ms": 312,
    "overall_confidence": 0.91,
    "escalated_for_review": false
  }
}
```

### 3.1 Casos de Status de Validação

| `validation.status` | Significado | Ação do Sistema |
|--------------------|-------------|-----------------|
| `valid` | Extração bem-sucedida, dentro dos limites clínicos | Encaminhar ao consumidor |
| `out_of_range` | Valor extraído fora do intervalo clinicamente seguro | Bloquear + alertar operador |
| `unit_inference_failed` | Unidade não pôde ser inferida com confiança ≥ threshold | Solicitar confirmação |
| `intent_unrecognized` | Intenção não mapeada no catálogo | Descartar + logar |
| `low_confidence` | Score de confiança composto abaixo do threshold | Escalar para revisão humana |
| `schema_violation` | Output do LLM não conformou ao schema Pydantic | Rejeitar + re-tentar (max 2x) |
| `conflicting_parameters` | Combinação de parâmetros mutuamente exclusiva detectada | Bloquear + alertar |

---

## 4. Análise de Dificuldades e Casos de Borda

Esta seção constitui o núcleo da análise de risco de engenharia do protótipo. A combinação de áudio de ambiente clínico com terminologia médica em português representa um dos cenários mais desafiadores para sistemas STT + LLM comerciais.

### 4.1 Ambiguidades Fonéticas e de Transcrição (Camada STT)

O STT opera sem conhecimento de domínio. Ele transcreve sons, não significados. O léxico clínico em português é particularmente suscetível a:

| Categoria | Exemplo de Erro STT | Transcrição Incorreta | Parâmetro Afetado | Risco Clínico |
|-----------|---------------------|-----------------------|-------------------|---------------|
| Confusão de numerais | "setenta" → "sessenta" | FiO2 de 70% transcrito como 60% | FiO2, FR, FC | **ALTO** — desvio de 10% na FiO2 é clinicamente relevante |
| Confusão de unidades faladas | "cinco cmH2O" → "cinco centímetros" | PEEP com unidade ambígua | PEEP | **MÉDIO** — unidade inferível por contexto |
| Homofonia de siglas | "PA" → "pá" | Parâmetro não reconhecido | Pressão Arterial | **ALTO** — sigla não mapeada descarta o comando |
| Numerais compostos | "doze e meio" → "12,5" vs "12 e meio" | Ambiguidade de valor fracionário | VC, doses | **MÉDIO** |
| Abreviações verbais | "bota a fi de dois quarenta" | FiO2 = 240%? | FiO2 | **CRÍTICO** — valor impossível não rejeitado pelo STT |
| Ruído mascarando sílabas | "P[ruído]EP em oito" | "PEEP em oito" parcialmente corrido | PEEP | **ALTO** — syllable dropout altera o parâmetro reconhecido |

**Mitigação no nível STT:** Uso de vocabulário customizado (_custom vocabulary_) nas APIs de STT (ex: AWS Transcribe Medical, Google Speech-to-Text Healthcare) com termos clínicos, siglas e unidades prioritizadas. Onde disponível, utilizar modelos de linguagem de domínio médico (ex: Whisper fine-tuned em corpus clínico PT-BR).

### 4.2 Omissões Semânticas (Conhecimento Implícito de Domínio)

O profissional de saúde comunica-se com pressuposto de contexto compartilhado. Omissões são a norma, não a exceção:

| Tipo de Omissão | Frase Real | O que está Implícito | Estratégia de Inferência |
|-----------------|------------|----------------------|--------------------------|
| Omissão de unidade | "bota a PEEP em cinco" | cmH2O (unidade canônica de PEEP) | Mapeamento parâmetro → unidade default no prompt |
| Omissão de dispositivo | "sobe o fluxo" | Ventilador ativo no contexto de sessão | Contexto de equipamento ativo na sessão |
| Omissão de direção | "ajusta o volume" | Ambíguo — aumentar ou definir? | Necessita valor explícito; escalar se ausente |
| Omissão de paciente | "zera o alarme" | Paciente do leito ativo | Contexto de sessão / autenticação do operador |
| Abreviação de protocolo | "modo VC, seis por quilo" | Volume Corrente Protegido = 6 mL/kg peso predito | LLM com regra de negócio para calcular VC absoluto |

O LLM é o único ator do pipeline com capacidade de resolver estas omissões, via conhecimento de domínio embutido nos exemplos few-shot e nas instruções do system prompt.

### 4.3 Ruído de Ambiente Clínico (Impacto na WER)

A UTI é um dos ambientes acústicos mais adversos para sistemas de reconhecimento de fala:

| Fonte de Ruído | Frequência Típica | Impacto Estimado na WER | Mitigação |
|---------------|-------------------|-------------------------|-----------|
| Alarmes de monitor (bipes 60–100 dB) | Contínua | +8–15% WER em sobreposição direta | VAD (Voice Activity Detection) ajustado; beamforming se multi-mic |
| Fala sobreposta (médico + enfermeiro) | Frequente | +20–35% WER (diarização incorreta) | Speaker diarization; push-to-talk obrigatório no protótipo |
| Ventilador mecânico (ruído de fundo contínuo) | Contínua | +3–7% WER (ruído de banda larga) | Noise cancellation no pré-processamento de áudio |
| Intercomunicadores e rádios | Episódica | +5–10% WER em janelas de sobreposição | Supressão de ruído espectral |
| Movimentação e atrito de tecidos | Episódica | Baixo impacto isolado | Posicionamento correto do microfone |

**WER de referência:** Modelos STT de propósito geral em português médico em ambiente de UTI simulado apresentam WER de **12–22%** sem adaptação de domínio. Com vocabulário customizado e supressão de ruído, espera-se redução para **5–10%**.

### 4.4 Variações de Escrita e Sinônimos Clínicos

O vocabulário clínico brasileiro apresenta alta variabilidade entre profissionais, especialidades e regiões:

| Parâmetro Canônico | Variantes Observadas | Tipo de Variação |
|--------------------|----------------------|------------------|
| `pressao_arterial` | PA, pressão, PA sistólica, Pas, pressão do paciente, tensão | Sigla, truncamento, sinônimo |
| `frequencia_cardiaca` | FC, frequência, batimento, pulso, ritmo | Sinônimo, metonímia |
| `saturacao_oxigenio` | SpO2, sato2, sato, saturação, sat, pulso-ox | Sigla, truncamento, marca |
| `pressao_expiratoria_final_positiva` | PEEP, peep (minúscula falada), pressão final | Sigla, descrição parcial |
| `volume_corrente` | VC, Vt, volume, volume tidal | Sigla, anglicismo |
| `fracao_inspirada_oxigenio` | FiO2, fi dois, fi de o2, fio, fração de oxigênio | Sigla, pronúncia expandida |
| `frequencia_respiratoria` | FR, frequência respiratória, respiração, ciclos | Sigla, sinônimo, metonímia |

**Impacto no pipeline:** O STT pode transcrever corretamente a forma falada ("sato dois"), mas o LLM precisa mapear para o parâmetro canônico (`saturacao_oxigenio`) sem ambiguidade. Isso é tratado por uma tabela de sinônimos no system prompt e nos exemplos few-shot.

### 4.5 Casos de Borda Críticos (Edge Cases Identificados)

| ID | Cenário | Comportamento Esperado do Pipeline | Risco se Mal Tratado |
|----|---------|-----------------------------------|----------------------|
| EC-01 | Valor fora do intervalo fisiológico ("FiO2 em duzentos") | `status: out_of_range`, bloqueio, alerta | Configuração impossível enviada ao equipamento |
| EC-02 | Parâmetros conflitantes (modo VCV + pressão controlada) | `status: conflicting_parameters` | Modo incoerente no ventilador |
| EC-03 | Comando em negação ("não muda o PEEP") | `intent: noop` ou descarte explícito | Interpretação invertida do comando |
| EC-04 | Múltiplos comandos em sequência ("PEEP cinco e FiO2 quarenta") | Dois payloads separados ou `multi_command: true` | Apenas um parâmetro processado |
| EC-05 | Valor relativo ("sobe dois a mais") | Necessita contexto do valor atual; escalar se indisponível | Valor absoluto incorreto calculado |
| EC-06 | Parâmetro ambíguo sem especificação de dispositivo em cenário multi-equipamento | Escalar para confirmação | Configuração no equipamento errado |
| EC-07 | Transcrição de altíssima incerteza (confidence < 0.5) | Descartar e solicitar repetição | Extração sobre texto sem sentido |
| EC-08 | Comando de voz durante alarme sonoro ativo | VAD pode rejeitar o áudio; priorizar canal de fala | Comando perdido em momento crítico |

---

## 5. Mitigações Propostas

### 5.1 Estratégia Anti-Alucinação do LLM

Alucinações em contexto médico — onde o LLM inventa parâmetros, valores ou unidades não presentes na transcrição — são inaceitáveis mesmo em fase de protótipo.

**Camada 1 — Prompt Engineering Defensivo:**
- O system prompt instrui explicitamente o LLM a retornar `intent: "unrecognized"` em vez de inventar uma intenção
- Campos opcionais são `null` por default; o LLM nunca deve preencher um campo com suposição não fundamentada na transcrição
- Instrução explícita: *"Se a transcrição não contiver valor numérico claro, o campo `value` deve ser `null` e `requires_human_confirmation` deve ser `true`"*

**Camada 2 — Schema Enforcement via Pydantic:**
```
Transcrição → LLM → JSON bruto → Pydantic Validator → JSON final ou exceção
```
- Tipos estritos (`float`, `int`, `Literal[...]` para enums de intenção e unidade)
- `@validator` e `@root_validator` customizados para regras de domínio (ex: FiO2 ∈ [21, 100])
- Exceções Pydantic são capturadas, logadas e resultam em `status: schema_violation`

**Camada 3 — Retry com Feedback:**
- Em caso de `schema_violation`, o pipeline realiza até 2 tentativas adicionais, incluindo o erro de validação no prompt como feedback corretivo para o LLM
- Após 3 falhas consecutivas: descarte definitivo + alerta ao operador

### 5.2 Validação de Intervalos Clínicos

```
┌─────────────────────────────────────────────────────────┐
│               GRADE DE VALIDAÇÃO DE VALORES              │
│                                                         │
│  Valor extraído pelo LLM                                │
│          │                                              │
│          ▼                                              │
│  ┌───────────────────┐                                  │
│  │ Dentro do intervalo│──SIM──▶ status: valid           │
│  │ fisiológico?       │                                 │
│  └────────┬──────────┘                                  │
│           │NÃO                                          │
│           ▼                                             │
│  ┌───────────────────┐                                  │
│  │ Dentro do intervalo│──SIM──▶ status: valid           │
│  │ de equipamento     │         requires_confirmation:  │
│  │ (limites técnicos)?│         true (aviso clínico)    │
│  └────────┬──────────┘                                  │
│           │NÃO                                          │
│           ▼                                             │
│     status: out_of_range                                │
│     Bloqueio + Alerta Obrigatório                       │
└─────────────────────────────────────────────────────────┘
```

A tabela de intervalos é mantida como configuração versionada e auditável, separada do código, permitindo atualização baseada em evidência clínica sem redeploy do pipeline.

### 5.3 Tratamento de Confiança Composta

O `overall_confidence` é calculado como:

```
overall_confidence = w_stt × stt_confidence + w_llm × llm_extraction_confidence
```

Onde os pesos `w_stt` e `w_llm` são configuráveis (default: 0.4 e 0.6, respectivamente, priorizando a qualidade da extração semântica sobre a transcrição bruta).

| Faixa de `overall_confidence` | Ação |
|-------------------------------|------|
| ≥ 0.90 | Processamento automático |
| 0.75 – 0.89 | Processamento com flag de baixa confiança no payload |
| 0.60 – 0.74 | Processamento bloqueado; confirmação humana obrigatória |
| < 0.60 | Descarte + solicitação de repetição do comando |

### 5.4 Governança de Dados e Auditabilidade

- **Imutabilidade do log:** Todo `pipeline_run_id` é persistido com o áudio original, transcrição bruta, output do LLM e status de validação — rastreabilidade completa para análise post-hoc
- **Sem ação autônoma em equipamentos:** O payload do protótipo é lido por um operador humano antes de qualquer configuração física — o pipeline é um _decision support tool_, não um sistema de controle autônomo
- **Anonimização:** Nenhum dado de paciente (nome, prontuário, data de nascimento) deve transitar pelo pipeline de IA; o contexto de sessão usa apenas identificadores internos desvinculados

---

## Apêndice A — Glossário de Siglas

| Sigla | Expansão |
|-------|----------|
| STT | Speech-to-Text (Reconhecimento de Fala) |
| LLM | Large Language Model |
| WER | Word Error Rate (Taxa de Erro de Palavras) |
| VAD | Voice Activity Detection |
| PEEP | Positive End-Expiratory Pressure |
| FiO2 | Fraction of Inspired Oxygen |
| VC / Vt | Volume Corrente / Tidal Volume |
| FR | Frequência Respiratória |
| FC | Frequência Cardíaca |
| PA / PAM | Pressão Arterial / Pressão Arterial Média |
| SpO2 | Saturação Periférica de Oxigênio |
| EtCO2 | End-Tidal Carbon Dioxide (Capnografia) |
| PCV | Pressure-Controlled Ventilation |
| VCV | Volume-Controlled Ventilation |
| PSV | Pressure Support Ventilation |
| UTI | Unidade de Terapia Intensiva |

---

*Documento mantido pela equipe de engenharia do `vlab-stt-llm`. Revisões clínicas devem ser validadas por profissional médico habilitado antes de qualquer uso em ambiente assistencial real.*
```