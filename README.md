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
3.  Impor segurança clínica através de Schema Enforcement e validações determinísticas (Inteligência Híbrida) adotando uma postura *Fail-Fast*.

### A Arquitetura da Solução

O pipeline implementa o padrão **Service/Manager** para rotação de chaves e roteamento transparente de modelos, seguido por uma validação matemática offline:
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
                   │(Rotation & Quota) │
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
🔬 Abordagem Híbrida e Postura Fail-Fast
A IA probabilística não deve atuar sem uma rede de segurança em domínios críticos. Este projeto implementa testes A/B entre duas estratégias de extração semântica:

Variante V1 (Direct Schema): Utiliza Zero-Shot Prompting associado ao recurso Structured Outputs nativo do Gemini (via response_schema). Otimizada para baixa latência.

Variante V2 (Chain-of-Thought): Utiliza a estratégia <reasoning> para obrigar o modelo a externalizar sua lógica clínica antes de preencher a estrutura JSON. Otimizada para ambiguidade e complexidade contextual.

O Diferencial Determinístico (Hard Rules & Fail-Fast):
Independentemente da variante escolhida, o dado bruto do LLM passa pelo crivo do Pydantic Validator. Se o modelo omitir uma unidade óbvia, o Pydantic a injeta (OK_INFERRED_UNIT_BY_RULE). Se o modelo alucinar e aceitar uma FiO2 de 200%, o Pydantic adota uma postura Fail-Fast, bloqueando a operação imediatamente e sinalizando o status como OUT_OF_BOUNDS.

📊 Avaliação e Métricas (Benchmark de Engenharia)
O pipeline foi estressado contra casos clínicos sintetizados (mock_data/ground_truth.json). O script de avaliação (evaluate_pipeline.py) abandona métricas acadêmicas isoladas e mede o desempenho pragmático de engenharia:

Taxa de Extração Exata (Exact Match): Porcentagem de áudios que geraram um JSON exatamente igual ao gabarito esperado.

Taxa de Falha de Parseamento (Parse Error Rate): Quantidade de vezes que a IA devolveu um JSON malformado que quebrou a validação do schema.

Latência (End-to-End): Tempo total (STT + LLM Extração) para validação de viabilidade em tempo real.

Resiliência a Cotas (Free Tier): O avaliador conta com controle rigoroso de concorrência (asyncio.Semaphore) e rotação multi-chaves automática (GeminiManager) para blindar o ambiente contra erros de limite de taxa (HTTP 429).

🚀 Reprodutibilidade e Setup
1. Pré-requisitos
Python 3.11+

Poetry 2.x ou superior

2. Instalação do Ambiente
Clone o repositório e instale as dependências usando Poetry:

PowerShell
git clone [https://github.com/Lionzio/vlab-stt-llm.git](https://github.com/Lionzio/vlab-stt-llm.git)
cd vlab-stt-llm

# O arquivo poetry.toml força a criação da .venv dentro do próprio projeto.
python -m poetry install
3. Configuração de Chaves (API)
Crie um arquivo .env na raiz do repositório. Para usufruir da rotação de cotas do GeminiManager, recomenda-se fortemente fornecer duas chaves distintas:

Snippet de código
GEMINI_API_KEY_PRIMARY=sua_chave_primaria_aqui
GEMINI_API_KEY_SECONDARY=sua_chave_secundaria_aqui  # Atua como fallback em caso de HTTP 429
🛠 Executando os Experimentos
1. Suíte de Testes (Inteligência Híbrida)
Garante matematicamente que as regras de segurança do Schema (Fail-Fast) estão ativas:

PowerShell
python -m poetry run pytest tests/test_pipeline.py -v
2. Benchmark de Modelos A/B
Roda o pipeline completo contra os áudios de teste, comparando V1 e V2 com as métricas pragmáticas.

PowerShell
# Inicie a avaliação automatizada:
python -m poetry run python scripts/evaluate_pipeline.py
3. Endpoint FastAPI (Live Test)
Suba o servidor para testar a injeção de áudios ao vivo via Swagger:

PowerShell
python -m poetry run uvicorn src.main:app --reload --port 8000
Acesse o Swagger UI em: http://localhost:8000/docs

📚 Documentação Adicional
mock_data/README.md — Mini-dicionário clínico e catálogo de parâmetros suportados.

docs/ANALISE_STT.md — O "coração científico" do projeto. Contém a análise das decisões, dificuldades semânticas, falhas frequentes e mitigações.

docs/domain_analysis.md — Detalhamento completo do catálogo de comandos, intenções e regras de negócio de UTI.

docs/evaluation_report.md — Relatório gerado pela execução do pipeline de avaliação.

Aviso de Uso Clínico: Este projeto é um protótipo experimental desenvolvido exclusivamente para avaliação técnica. Não está homologado para uso assistencial real.