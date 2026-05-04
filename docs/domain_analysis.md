# Domain Analysis: vlab-stt-llm
**Versão:** 0.2.0 | **Status:** Documento Vivo | **Classificação:** Uso Interno — Protótipo Experimental

---

## 1. Visão Geral e Estratégia

### 1.1 Contexto e Motivação

Profissionais de saúde em ambientes de alta pressão — UTIs, centros cirúrgicos, emergências — interagem constantemente com equipamentos clínicos complexos via interfaces físicas que exigem atenção visual e motora. A interação por voz representa uma oportunidade de reduzir a carga cognitiva e aumentar a velocidade de resposta em cenários críticos, desde que implementada com extremo rigor clínico e de engenharia de IA.

O `vlab-stt-llm` é um pipeline experimental que valida a viabilidade técnica de capturar **comandos de voz em português clínico**, transcrevê-los e extrair parâmetros médicos estruturados com semântica validada, adotando uma arquitetura rigorosa de **Fail-Fast**.

### 1.2 Arquitetura do Pipeline em Duas Etapas
```text
┌────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE vlab-stt-llm                             │
│                                                                            │
│  ┌──────────┐   Audio      ┌─────────────┐  Transcrição  ┌─────────────┐   │
│  │  Entrada │─────────────▶│  STT Engine │──────────────▶│ LLM + Few-  │   │
│  │  de Voz  │   (PCM/WAV)  │   (Gemini)  │   (texto bruto│ Shot Prompt │   │
│  └──────────┘              └─────────────┘               │             │   │
│                                                          └──────┬──────┘   │
│                                                                 │          │
│                                                     ┌───────────▼────────┐ │
│                                                     │ Pydantic Validator │ │
│                                                     │ (Fail-Fast Rules)  │ │
│                                                     └───────────┬────────┘ │
│                                                                 │          │
│                                                     ┌───────────▼────────┐ │
│                                                     │  JSON Estruturado  │ │
│                                                     │  (Output Final)    │ │
│                                                     └────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
Etapa 1 — STT (Speech-to-Text): Responsável pela transcrição do áudio em texto. Não possui conhecimento formal de regras de negócio, entregando um texto fonético sujeito a ruídos e coloquialismos.Etapa 2 — Extração Semântica e Hard Rules: O LLM atua como motor de compreensão. Ele recebe a transcrição bruta e um prompt de domínio. Sua saída é interceptada pelo Pydantic, que atua como juiz determinístico: ele impõe as regras matemáticas clínicas, inferindo unidades ausentes ou abortando a operação em caso de valores inseguros (Safety Bounds).2. Catálogo de Comandos (Dataset Base)2.1 Taxonomia de Intenções (intent)IntençãoDescrição FuncionalEquipamento Alvoajustar_parametroModificar o valor de um parâmetro clínico em equipamento ativoVentilador, Monitoriniciar_terapiaIniciar um protocolo ou modo de terapia respiratória/hemodinâmicaVentiladorsilenciar_alarmeSilenciar ou pausar temporariamente um alarme em equipamentoVentilador, Monitorconsultar_statusConsultar o valor corrente de um parâmetro ou estadoVentilador, MonitordesconhecidaFuga de escopo ou comando não reconhecido-2.2 Catálogo de Parâmetros e Limites ClínicosAs regras abaixo estão codificadas diretamente na camada de validação do Pydantic (PARAMETER_BOUNDS).Parâmetro CanônicoAlias Clínico ComumUnidade PadrãoIntervalo SeguropeepPEEP, pipecmH2O0.0 a 25.0fio2FiO2, fi de o2%21.0 a 100.0frequencia_respiratoriaFR, respiraçãoirpm4.0 a 60.0volume_correnteVC, Vtml200.0 a 800.0frequencia_cardiacaFC, batimentobpm20.0 a 300.0pressao_arterialPA, pressãommHg(Requer clarificação sistólica/diastólica)3. Schema de Saída EsperadoO payload abaixo representa o contrato de interface (MedicalParameterExtraction) entre o pipeline de IA e o sistema consumidor do equipamento.JSON{
  "intent": "ajustar_parametro",
  "parameter": "peep",
  "value": 5.0,
  "unit": "cmH2O",
  "status": "OK_INFERRED_UNIT_BY_RULE",
  "notes": "Unidade 'cmH2O' injetada por regra determinística."
}
3.1 Casos de Status de Validação (status)StatusSignificadoAção do SistemaOKExtração perfeita sem interferências.Encaminhar comandoOK_INFERRED_UNITO LLM deduziu a unidade pelo contexto.Encaminhar comandoOK_INFERRED_UNIT_BY_RULEO Pydantic injetou a unidade canônica por regra dura.Encaminhar comandoOUT_OF_BOUNDSO valor extraído excedeu os limites seguros.Bloquear + AlertaMISSING_VALUEComando incompleto sem valor numérico claro.Solicitar repetiçãoREQUIRES_CLARIFICATIONAmbiguidade matemática (ex: frações "12 por 8").Solicitar contexto4. Análise de Dificuldades Clínicas e Ruído de STTA UTI é um ambiente acústico adverso e o jargão médico é semanticamente complexo.4.1 Ambiguidades FonéticasCategoriaExemplo de Erro STTTranscrição ObtidaParâmetro AfetadoRisco ClínicoConfusão de numerais"setenta" → "sessenta"FiO2 de 70% transcrito como 60%FiO2ALTO — desvio é clinicamente relevante.Homofonia"PEEP" → "pipe"Sigla ambíguaPEEPMÉDIO — Resolvido via Embeddings/LLM.Numerais literais"frequência em meia dúzia"Numeral por extenso em vez de dígitoFRMÉDIO — Resolvido pelo NLP interno.4.2 Omissões Semânticas (Conhecimento Implícito)Omissões são a norma. Frases como "bota a PEEP em cinco" pressupõem a unidade cmH2O. O LLM é a camada responsável por resolver essas omissões mapeando jargões ("f i o dois") para entidades canônicas ("fio2").5. Mitigações e Engenharia de Resiliência5.1 Postura Fail-Fast (Anti-Alucinação)Alucinações matemáticas de LLMs são o maior risco em cenários críticos. O pipeline vlab atua sob o princípio do Fail-Fast (Falhe Rápido e Seguro).Camada 1 — Prompting Defensivo:O sistema é instruído a preencher o value com null e adotar status de MISSING_VALUE caso a frase não possua dígitos ou referências diretas de escala (evitando inferências inventadas).Camada 2 — Barreira Determinística (Pydantic):PlaintextJSON bruto do LLM ──▶ Pydantic Validator ──▶ [Check_Limites_Vitais] ──▶ Saída ou Bloqueio
O framework Pydantic é executado localmente de forma agnóstica à nuvem. Se o LLM alucinar uma Frequência Respiratória de 150 irpm, o Pydantic altera o payload para OUT_OF_BOUNDS antes que a informação atinja a interface do ventilador mecânico.5.2 Avaliação ContínuaAs decisões de arquitetura e promtings são retroalimentadas pelos testes unitários e de avaliação em cima do mock_data, priorizando a métrica de Exact Match (garantia matemática da estrutura de dados) em oposição ao F1-Score puramente linguístico.Documento mantido pela equipe de engenharia do vlab-stt-llm. Revisões clínicas devem ser validadas por profissional médico habilitado antes de qualquer uso em ambiente assistencial real.