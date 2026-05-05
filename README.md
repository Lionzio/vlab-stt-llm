# vlab-stt-llm

Pipeline experimental para transcrição e extração estruturada de parâmetros médicos a partir de comandos de voz, com foco em equipamentos de suporte à vida (ventiladores mecânicos e monitores).

[![Live Demo](https://img.shields.io/badge/Live_Demo-Acessar_Frontend-000000?style=for-the-badge&logo=vercel)](https://vlab-stt-llm.vercel.app)
[![API Docs](https://img.shields.io/badge/Swagger_UI-Acessar_API-46E3B7?style=for-the-badge&logo=fastapi)](https://vlab-stt-llm.onrender.com/docs)
[![CI Pipeline](https://github.com/Lionzio/vlab-stt-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Lionzio/vlab-stt-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)

---

## 🚀 Teste em produção (Live demo)

A maneira mais rápida de avaliar esta Prova de Conceito é através do nosso ambiente Cloud (Frontend na Vercel + API no Render com Graceful Degradation):

1. Acesse o **[VLab Command Center (Frontend)](https://vlab-stt-llm.vercel.app)**.
2. Na pasta `data/audio_samples/` deste repositório, baixe um dos arquivos de áudio de teste (ex: `TC-001.mp3` para um caso de sucesso ou `TC-004.mp3` para forçar um bloqueio de segurança).
3. Faça o upload do áudio na interface e acompanhe a extração estruturada em tempo real.

> **Nota para desenvolvedores:** A documentação interativa da API de produção (Swagger) está disponível em **[vlab-stt-llm.onrender.com/docs](https://vlab-stt-llm.onrender.com/docs)**.

---

## Visão geral

Este repositório contém a entrega para o Desafio Técnico de Desenvolvedor de IA. O objetivo do projeto é construir um pipeline capaz de:

1. Converter comandos de voz ambíguos ou com ruído em texto (STT).
2. Extrair a intenção e os parâmetros clínicos usando IA (LLM).
3. Validar a saída através de regras de negócio locais, garantindo a segurança da operação antes de enviar qualquer dado a um equipamento médico.

## Arquitetura

O fluxo utiliza a API do Gemini para as etapas de IA, com um gerenciador central (`GeminiManager`) responsável por alternar as chaves de API caso o limite de requisições seja atingido (fallback) e capturar instabilidades da rede (Erro 503).

```text
Áudio (.mp3)
      │
      ▼
┌───────────────┐     transcrição     ┌───────────────────────┐
│  GeminiSTT    │ ───────────────────▶│ ParameterExtractor    │
│  (Flash Model)│                     │ (V1 Direct / V2 CoT)  │
└───────┬───────┘                     └──────────┬────────────┘
        │                                        │ JSON Bruto
        │          ┌───────────────────┐         │
        └─────────▶│   GeminiManager   │◀────────┘
                   │(Rotação de Cotas) │
                   └───────────────────┘
                                                 ▼
                                      ┌───────────────────────┐
                                      │ Pydantic Validator    │
                                      │ (Regras Locais)       │
                                      └──────────┬────────────┘
                                                 │
                                                 ▼
                                      MedicalParameterExtraction
                                      {intent, parameter, value,
                                       unit, status, notes}

Validação híbrida (IA + Regras locais)
Para evitar alucinações matemáticas comuns em LLMs, o dado bruto gerado pela IA é interceptado pelo Pydantic para validação determinística (Fail-Fast):

Correção de unidades: Se o modelo omitir a unidade de um parâmetro (ex: PEEP), o validador injeta a unidade padrão (cmH2O) e ajusta o status.

Limites de segurança: Se o modelo extrair um valor impossível (ex: FiO2 de 200%), a validação recusa a operação e altera o status para OUT_OF_BOUNDS.

Avaliação e testes A/B
O projeto inclui um script de avaliação (evaluate_pipeline.py) que testa o pipeline contra casos de uso sintetizados (omissões, ruídos, etc.) comparando duas abordagens de prompt:

Variante V1 (Direct Schema): Usa o response_schema nativo da API. Foco em menor latência.

Variante V2 (Chain-of-Thought): Exige que o modelo escreva o raciocínio em uma tag <reasoning> antes do JSON. Foco em lidar com ambiguidades.

O script mede o Word Error Rate (WER) do áudio e as taxas de precisão (F1-Score) da extração final.

Como executar localmente (Desenvolvimento)
O ambiente recomendado para execução local é via Docker, garantindo a reprodutibilidade da infraestrutura.

1. Configuração
Crie um arquivo .env na raiz do projeto contendo suas chaves de API:

Snippet de código
GEMINI_API_KEY_PRIMARY="sua_chave_aqui"
GEMINI_API_KEY_SECONDARY="sua_chave_reserva_aqui" # Opcional (Fallback)

2. Executando a API via Docker
Bash
docker build -t vlab-api .
docker run -p 8000:8000 --env-file .env vlab-api
A API ficará disponível em http://127.0.0.1:8000/docs.

3. Testes e Benchmark Automáticos
Requer Python 3.11+ e Poetry instalados:

Bash
poetry install
poetry run pytest tests/test_pipeline.py -v
poetry run python scripts/evaluate_pipeline.py

4. Executando o Frontend Local
Com a API rodando, abra um segundo terminal:

Bash
cd frontend
npm install
npm run dev
Acesse http://localhost:5173/ no seu navegador.

Documentação adicional

Detalhes sobre a construção do sistema podem ser encontrados na pasta docs/:

docs/domain_analysis.md: Catálogo de parâmetros, regras de negócio e estratégia de fallback.

docs/ANALISE_STT.md: Análise sobre as falhas frequentes de transcrição clínica e as mitigações adotadas.

docs/evaluation_report.md: Relatório do benchmark gerado automaticamente.

Aviso: Este projeto é um protótipo construído exclusivamente para avaliação técnica. Não deve ser utilizado em ambiente clínico real.