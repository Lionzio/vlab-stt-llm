```markdown
# vlab-stt-llm

> Pipeline experimental de Inteligência Artificial para transcrição e extração estruturada de parâmetros médicos a partir de comandos de voz em português clínico, com foco em equipamentos de suporte à vida (ventiladores mecânicos, monitores multiparamétricos).

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://docs.pydantic.dev/)
[![SDK](https://img.shields.io/badge/Google%20Gemini-google--genai-orange)](https://ai.google.dev/)
[![Pytest](https://img.shields.io/badge/Pytest-Su%C3%ADte_Ativa-brightgreen)](https://docs.pytest.org/)

---

## 🎯 Objetivo e Visão Geral

Este repositório atende ao Desafio Técnico para a vaga de **Desenvolvedor de Inteligência Artificial**. O foco exclusivo é a prototipagem experimental, avaliação rigorosa e documentação clara de um pipeline capaz de:
1.  Ingerir comandos de voz ambíguos ou incompletos (STT).
2.  Extrair intenções, parâmetros e valores (Extração Semântica).
3.  Impor segurança clínica através de Schema Enforcement e validações determinísticas (Inteligência Híbrida).

### A Arquitetura da Solução

O pipeline implementa o padrão **Service/Manager** para orquestração de cotas e roteamento transparente de modelos, seguido por uma validação matemática offline:

```text
Áudio (.mp3)
      │
      ▼
┌───────────────┐     transcrição     ┌───────────────────────┐
│  GeminiSTT    │ ───────────────────▶│ ParameterExtractor    │
│  (Flash Model)│                     │ (V1 Direct / V2 CoT)  │
└───────┬───────┘                     └──────────┬────────────┘
        │                                        │ JSON bruto
        │          ┌───────────────────┐         │
        └─────────▶│   GeminiManager   │◀────────┘
                   │ (Router & Fallback)│
                   └───────────────────┘
                                                 ▼
                                      ┌───────────────────────┐
                                      │ Pydantic Validator    │
                                      │ (Inteligência Híbrida)│
                                      └──────────┬────────────┘
                                                 │
                                                 ▼
                                      MedicalParameterExtraction
                                      {intent, parameter, value,
                                       unit, status, notes}
🔬 Abordagem Híbrida e Testes A/BA IA probabilística não deve atuar sem uma rede de segurança em domínios críticos. Este projeto implementa testes A/B entre duas estratégias de extração semântica:Variante V1 (Direct Schema): Utiliza Zero-Shot Prompting associado ao recurso Structured Outputs nativo do Gemini (via response_schema). Otimizada para baixa latência.Variante V2 (Chain-of-Thought): Utiliza a estratégia <reasoning> para obrigar o modelo (via gemini-2.5-pro) a externalizar sua lógica clínica antes de preencher a estrutura JSON. Otimizada para ambiguidade e complexidade contextual.O Diferencial Determinístico (Hard Rules):Independentemente da variante escolhida, o dado bruto do LLM passa pelo crivo do Pydantic Validator. Se o modelo omitir uma unidade óbvia, o Pydantic a injeta (OK_INFERRED_UNIT_BY_RULE). Se o modelo alucinar e aceitar uma $FiO_2$ de $200\%$, o Pydantic derruba a operação (OUT_OF_BOUNDS).📊 Avaliação e Métricas (Benchmark)O pipeline foi estressado contra 6 cenários clínicos críticos sintetizados em áudio. O script de avaliação (evaluate_pipeline.py) mede:Acurácia Acústica: WER (Word Error Rate) e CER (Character Error Rate) com normalização lexical prévia (via jiwer).Desempenho de NLP: Precision, Recall e F1-Score para Intenções e Parâmetros extraídos.Resiliência a Cotas (Free Tier): O avaliador conta com controle rigoroso de concorrência (asyncio.Semaphore(2)), roteamento multi-chaves automático (GeminiManager) e cache em disco local para blindar o ambiente contra erros de cota (HTTP 429) do Google AI Studio.Para uma análise profunda dos desafios fonéticos encontrados e impacto das falhas, consulte a Análise Qualitativa de Domínio.🚀 Reprodutibilidade e Setup1. Pré-requisitosPython 3.11+Poetry 2.x ou superior2. Instalação do AmbienteClone o repositório e instale as dependências usando Poetry:PowerShellgit clone https://github.com/Lionzio/vlab-stt-llm.git
cd vlab-stt-llm

# O arquivo poetry.toml força a criação da .venv dentro do próprio projeto.
python -m poetry install
3. Configuração de Chaves (API)Crie um arquivo .env na raiz do repositório. Para usufruir da rotação de cotas do GeminiManager, recomenda-se fornecer duas chaves distintas (de projetos Google Cloud distintos):Snippet de códigoGEMINI_API_KEY_PRIMARY=sua_chave_primaria_aqui
GEMINI_API_KEY_SECONDARY=sua_chave_secundaria_aqui  # Opcional, atua como fallback em caso de erro 429
🛠 Executando os Experimentos1. Suíte de Testes (Inteligência Híbrida)Garante matematicamente que as regras de segurança do Schema estão ativas operando offline:PowerShellpython -m poetry run pytest tests/test_pipeline.py -v
2. Benchmark de Modelos A/BRoda o pipeline completo contra os 6 áudios de teste, comparando V1 e V2. Gera uma tabela Rich no terminal e exporta o docs/evaluation_report.md:PowerShell# Se for a primeira vez e os áudios não existirem, gere-os:
python -m poetry run python scripts/generate_dataset.py

# Inicie a avaliação:
python -m poetry run python scripts/evaluate_pipeline.py
3. Endpoint FastAPI (Live Test)Suba o servidor para testar a injeção de áudios ao vivo via Swagger:PowerShellpython -m poetry run uvicorn src.main:app --reload --port 8000
Acesse o Swagger UI em: http://localhost:8000/docsA arquitetura está protegida: caso suas chaves esgotem, a API retornará graciosamente um HTTP 429 Too Many Requests.📚 Documentação Adicionaldocs/ANALISE_STT.md — O "coração científico" do projeto. Contém análise detalhada das decisões técnicas, dificuldades semânticas inerentes ao contexto médico, falhas frequentes de transcrição e a estratégia de mitigação.docs/evaluation_report.md — O relatório estatístico gerado pela última execução completa do pipeline.Aviso de Uso Clínico: Este projeto é um protótipo experimental desenvolvido exclusivamente para fins de avaliação técnica e prova de conceito arquitetural. Não está homologado para uso em ambiente assistencial real.