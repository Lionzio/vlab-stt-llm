# Domain Analysis: vlab-stt-llm
**Versão:** 1.0.0 | **Status:** Baseline de Avaliação (Sprint 4) | **Classificação:** Uso Interno — Desafio Técnico

---

## 1. Visão Geral e Estratégia

### 1.1 Contexto e Motivação

Profissionais de saúde em ambientes de alta pressão — UTIs, centros cirúrgicos, emergências — interagem constantemente com equipamentos clínicos complexos via interfaces físicas que exigem atenção visual e motora. A interação por voz representa uma oportunidade de reduzir a carga cognitiva e aumentar a velocidade de resposta em cenários críticos, desde que implementada com extremo rigor clínico e de engenharia de software.

O `vlab-stt-llm` é um pipeline experimental que valida a viabilidade técnica de capturar **comandos de voz em português clínico**, transcrevê-los e extrair parâmetros médicos estruturados com semântica validada. O sistema adota uma arquitetura híbrida: combina a flexibilidade da IA Generativa (LLMs) com a rigidez de regras determinísticas (Regex/Pydantic) para garantir resiliência contra indisponibilidade de rede e segurança contra alucinações (Fail-Fast).

### 1.2 Arquitetura do Pipeline Híbrido (Graceful Degradation)
O pipeline foi projetado prevendo falhas de infraestrutura. Se a Inteligência Artificial falhar (Rate Limit ou Network Timeout), ferramentas offline assumem o controle, garantindo que o contrato JSON seja sempre respeitado.
```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE vlab-stt-llm                            │
│                                                                             │
│  ┌──────────┐  Audio   ┌───────────────┐ Transcrição ┌──────────────────┐   │
│  │ Entrada  │─────────▶│ STT Engine    ├────────────▶│ Extractor Engine │   │
│  │ de Voz   │          │ (IA ou Mock)  │             │ (IA ou Fallback) │   │
│  └──────────┘          └───────────────┘             └────────┬─────────┘   │
│                                                               │             │
│                                                   ┌───────────▼───────────┐ │
│                                                   │ Pydantic Validator    │ │
│                                                   │ (Hard Safety Bounds)  │ │
│                                                   └───────────┬───────────┘ │
│                                                               │             │
│                                                   ┌───────────▼───────────┐ │
│                                                   │   JSON Estruturado    │ │
│                                                   │   (Output Final)      │ │
│                                                   └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
Etapa 1 — STT (Speech-to-Text): Responsável por transformar o áudio em texto. A via principal é o Gemini STT. Em caso de falha de cota, um Mock STT de bypass é acionado.Etapa 2 — Motor de Extração: Interpreta a intenção do usuário. A via principal usa LLM para lidar com o jargão. A Via de Fallback usa Regex e Dicionários de Expressões Regulares (offline).Etapa 3 — Validação Determinística: O Pydantic atua como um "firewall clínico". Independentemente se os dados vieram da IA ou do Fallback, ele impõe regras matemáticas clínicas, infere unidades ou aborta a operação (Safety Bounds).2. Catálogo de Comandos e Baseline Clínico2.1 Justificativa do Baseline (Por que PEEP, FiO2 e FR?)Para estruturar o escopo deste protótipo, definimos um baseline de três parâmetros críticos associados à Ventilação Mecânica Invasiva (VMI). A escolha não foi aleatória:PEEP (Positive End-Expiratory Pressure): Teste de robustez a jargões, visto que médicos frequentemente omitem a unidade e usam aportuguesamentos ("pipe").FiO2 (Fração Inspirada de Oxigênio): Teste de segurança rigorosa (Safety Bounds). Um erro de transcrição que eleve a FiO2 para 200% (impossível fisicamente) deve ser detectado e bloqueado pela arquitetura.FR (Frequência Respiratória): Teste de ambiguidade de unidades, pois transita entre siglas (irpm) e extensos ("incursões por minuto").2.2 Catálogo de Parâmetros e Limites Clínicos (Safety Bounds)As regras abaixo estão codificadas diretamente na camada de validação do Pydantic (PARAMETER_BOUNDS), impedindo que alucinações matemáticas da IA cheguem aos sistemas do hospital.Parâmetro CanônicoAlias Clínico ComumUnidade PadrãoIntervalo Seguro (Bounds)Risco de QuebrapeepPEEP, pipecmH2O0.0 a 25.0Barotrauma / Colapso Alveolarfio2FiO2, fi de o2%21.0 a 100.0Toxicidade por O2frequencia_respiratoriaFR, respiraçãoirpm4.0 a 60.0Alcalose / Acidose Respiratóriavolume_correnteVC, Vtml200.0 a 800.0Volutraumapressao_arterialPA, pressãommHgRequer Sistólica/DiastólicaHipotensão / Hipertensão3. Schema de Interface (Contract)O payload abaixo representa o contrato estrito (MedicalParameterExtraction) entre o pipeline de IA e o sistema consumidor (ex: CLP do ventilador pulmonar).JSON{
  "intent": "ajustar_parametro",
  "parameter": "peep",
  "value": 5.0,
  "unit": "cmH2O",
  "status": "OK_INFERRED_UNIT_BY_RULE",
  "notes": "Unidade 'cmH2O' injetada por regra determinística local."
}
3.1 Tratamento de Estados (State Machine)StatusSignificadoAção do Sistema ClínicoOKExtração perfeita sem interferências.Encaminhar comandoOK_INFERRED_UNITO LLM deduziu a unidade pelo contexto de forma autônoma.Encaminhar comandoOK_INFERRED_UNIT_BY_RULEO motor Pydantic injetou a unidade canônica por regra hard-coded (segurança).Encaminhar comandoOUT_OF_BOUNDSO valor extraído excedeu os limites clínicos permitidos.Bloquear Operação + Alerta VisualMISSING_VALUEComando incompleto sem valor numérico claro (ex: "ajusta a PEEP").Solicitar repetição ao usuárioREQUIRES_CLARIFICATIONAmbiguidade semântica/matemática (ex: frações "12 por 8").Solicitar contexto ao usuário4. Engenharia de Resiliência (Fail-Fast & Fallback)Alucinações matemáticas e quedas de nuvem são os maiores riscos em cenários críticos. O pipeline vlab implementa três camadas de proteção:Prompting Defensivo (IA): O sistema é instruído a preencher o value com null e adotar status de MISSING_VALUE caso a frase não possua dígitos (evitando que a IA "invente" um número para agradar o usuário).Barreira Determinística (Pydantic): Funciona como um "Juiz de Ouro" rodando localmente. Se o LLM alucinar uma Frequência Respiratória de 150 irpm, o Pydantic intercepta, altera o payload para OUT_OF_BOUNDS e barra a instrução.Graceful Degradation (Heurística Offline): Em caso de falha de API ou esgotamento de cotas (HTTP 429), o sistema intercepta o erro de nuvem silenciosamente e roteia a transcrição para um extrator local baseado em Expressões Regulares (HeuristicParameterExtractor), garantindo que a extração não falhe e entregue o JSON.