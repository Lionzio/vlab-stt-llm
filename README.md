# 🫁 vlab-stt-llm: HMI Híbrida para Equipamentos Críticos

> **Prova de Conceito (PoC) — Desafio Técnico de Engenharia de IA**
> 
> Pipeline experimental de Inteligência Artificial para transcrição e extração estruturada de parâmetros médicos a partir de comandos de voz em português clínico, com foco em equipamentos de suporte à vida (ventiladores mecânicos, monitores multiparamétricos).

[![CI Pipeline](https://github.com/Lionzio/vlab-stt-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Lionzio/vlab-stt-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://docs.pydantic.dev/)
[![SDK](https://img.shields.io/badge/Google%20Gemini-google--genai-orange)](https://ai.google.dev/)

---

## 🎯 Visão Geral e Foco do Projeto

Este repositório atende ao Desafio Técnico para a vaga de **Desenvolvedor de Inteligência Artificial**. O desenvolvimento foi estritamente guiado por três pilares fundamentais:

1. **Prototipagem Experimental:** Criação de um pipeline inovador que combina IA Probabilística (LLMs) com regras determinísticas (*Fail-Fast*).
2. **Avaliação Técnica:** Bateria de testes rigorosa baseada em métricas pragmáticas de engenharia (WER, CER, F1-Score, Exact Match), validada por integração contínua (CI/CD).
3. **Clareza de Documentação:** Rastreabilidade completa de decisões arquiteturais, raciocínio clínico e instruções de reprodutibilidade.

---

## 🔬 Prototipagem Experimental: A Arquitetura da Solução

Profissionais de UTI lidam com sobrecarga cognitiva. A voz acelera processos, mas a IA pura sofre de alucinações matemáticas. Para resolver isso, implementamos uma **Abordagem Híbrida**:

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
O Diferencial Determinístico (Hard Rules & Fail-Fast)
Independentemente da variante de IA escolhida, o dado bruto passa pelo crivo do Pydantic Validator:

Omissão: Se o modelo omitir uma unidade óbvia, o Pydantic a injeta (OK_INFERRED_UNIT_BY_RULE).

Perigo Clínico: Se o modelo alucinar e aceitar uma FiO2 de 200%, o Pydantic adota uma postura Fail-Fast, bloqueando a operação e sinalizando o status como OUT_OF_BOUNDS.

📊 Avaliação Técnica e Benchmark
O pipeline foi estressado contra casos clínicos sintetizados (áudios simulando jargões, omissões e ruídos). O script de avaliação (evaluate_pipeline.py) executa um Teste A/B automatizado:

Variante V1 (Direct Schema): Zero-Shot Prompting via response_schema. (Foco em Baixa Latência).

Variante V2 (Chain-of-Thought): Prompt com <reasoning>. (Foco em Raciocínio Contextual).

Métricas de Engenharia Aferidas:
Taxa de Extração (Exact Match): O pipeline alcançou 100% de F1-Score em ambas as estratégias na extração semântica, corrigindo ativamente as falhas silábicas do motor STT.

Resiliência a Cotas (Free Tier): Controle de concorrência (asyncio.Semaphore), rotação multi-chaves (GeminiManager) e Fallback Offline (HeuristicParameterExtractor) blindam o ambiente contra o Erro HTTP 429.

🚀 Setup e Reprodutibilidade
Para garantir uma execução limpa e agnóstica a sistemas operacionais, o ambiente principal foi empacotado em Docker.

Opção 1: Via Docker (Recomendada 🥇)
1. Configure as chaves de API:
Crie um arquivo .env na raiz do projeto (use o .env.example como base):

Snippet de código
GEMINI_API_KEY_PRIMARY="sua_chave_primaria_aqui"
GEMINI_API_KEY_SECONDARY="sua_chave_secundaria_aqui" # Opcional (Fallback)
2. Faça o Build e Suba o Servidor:

Bash
docker build -t vlab-api .
docker run -p 8000:8000 --env-file .env --name vlab-server vlab-api
Acesse a interface interativa (Swagger) em: http://127.0.0.1:8000/docs

Opção 2: Via Local (Poetry)
Se preferir rodar nativamente (requer Python 3.11+):

Bash
python -m pip install poetry
poetry install
poetry run uvicorn src.main:app --reload --port 8000
🛠 Executando a Bateria de Testes
1. Suíte de Testes de Segurança (Inteligência Híbrida)
Garante que as regras do Schema (Fail-Fast) estão ativas:

Bash
poetry run pytest tests/test_pipeline.py -v
2. Benchmark Automatizado (A/B Test)
Roda a avaliação completa gerando as métricas de ML e STT.

Bash
poetry run python scripts/evaluate_pipeline.py
📚 Clareza de Documentação
O projeto conta com documentação aprofundada para embasar as decisões técnicas:

📄 docs/domain_analysis.md: Detalhamento da arquitetura Fail-Fast, catálogo de comandos e regras de negócio clínico de UTI.

📄 docs/ANALISE_STT.md: O "coração científico" da PoC. Análise de dificuldades fonéticas, ambiguidades e impacto do WER na extração estruturada.

📄 docs/evaluation_report.md: Relatório de performance gerado dinamicamente pelo script de testes A/B.

🗃️ mock_data/README.md: Mini-dicionário clínico e catálogo dos cenários de teste injetados.

⚠️ Aviso de Uso Clínico: Este projeto é um protótipo experimental desenvolvido exclusivamente para avaliação técnica. Não está homologado para uso assistencial real.