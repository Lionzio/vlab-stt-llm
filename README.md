```markdown
# vlab-stt-llm

> Pipeline experimental de IA para extração estruturada de parâmetros médicos a partir de comandos de voz em português clínico, focado em equipamentos de suporte à vida (ventiladores mecânicos, monitores multiparamétricos).

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-red)](https://docs.pydantic.dev/)
[![SDK](https://img.shields.io/badge/Google%20Gemini-google--genai-orange)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multistage-blue?logo=docker)](https://www.docker.com/)
[![Avaliação](https://img.shields.io/badge/Avalia%C3%A7%C3%A3o-4%2F6%20casos-yellow)](docs/evaluation_report.md)

---

## Visão Geral

O `vlab-stt-llm` valida a viabilidade de um pipeline em **duas etapas** para assistência clínica por voz:

```
Áudio (.mp3/.wav)
      │
      ▼
┌─────────────┐     transcrição bruta     ┌──────────────────────┐
│  GeminiSTT  │ ─────────────────────────▶│  ParameterExtractor  │
│  (STT)      │                           │  (LLM + Few-Shot)    │
└─────────────┘                           └──────────┬───────────┘
                                                     │ JSON bruto
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Pydantic Validator  │
                                          │  (Schema Enforcement)│
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          MedicalParameterExtraction
                                          {intent, parameter, value,
                                           unit, status, notes}
```

**Etapa 1 — STT:** `GeminiSTT` faz upload do áudio para o Gemini Files API, aguarda o estado `ACTIVE` via polling assíncrono e solicita a transcrição com `asyncio.wait_for` e timeout explícito de 60s.

**Etapa 2 — Extração:** `ParameterExtractor` envia a transcrição ao `gemini-2.5-flash` com `response_mime_type="application/json"` e `response_schema=MedicalParameterExtraction`, garantindo que o output do LLM seja diretamente validável via Pydantic sem parsing adicional.

---

## Resultados da Avaliação

O pipeline foi testado contra 6 cenários clínicos críticos. **4/6 casos aprovados** em execução real contra a API Gemini.

| ID | Cenário | STT | Schema | Intent | Resultado |
|----|---------|:---:|:------:|:------:|:---------:|
| TC-001 | `ideal` | ✅ | ✅ | ✓ | ✅ PASS |
| TC-002 | `unidade_omitida` | ✅ | ✅ | ✓ | ✅ PASS |
| TC-003 | `ambiguidade_terminologica` | ✅ | ✅ | ✓ | ✅ PASS |
| TC-004 | `fora_do_padrao_limites` | ✅ | ✅ | ✓ | ✅ PASS |
| TC-005 | `comando_incompleto` | ❌ | ❌ | ✗ | ⚠️ QUOTA |
| TC-006 | `ruido_simulado` | ✅ | ❌ | ✗ | ⚠️ QUOTA |

> **Nota sobre TC-005 e TC-006:** As falhas são exclusivamente causadas pelo esgotamento da cota diária do Gemini Free Tier (20 req/dia — `HTTP 429 RESOURCE_EXHAUSTED`), **não por falha lógica do pipeline**. TC-006 teve o STT concluído com sucesso (`"ajusta o volume corrente para 600 cough cough mililitros"`), e o LLM retornou `None` apenas porque a segunda chamada à API encontrou o limite de cota. O relatório completo está em [`docs/evaluation_report.md`](docs/evaluation_report.md).

**Destaques dos casos aprovados:**
- **TC-002** prova a inferência de unidade implícita: `"coloca pipe em cinco"` (STT transcreveu "PEEP" como "pipe") → LLM extraiu `parameter=peep, unit=cmH2O, status=OK_INFERRED_UNIT`
- **TC-003** prova o mapeamento de ambiguidade PA sistólica/diastólica → `status=REQUIRES_CLARIFICATION`
- **TC-004** prova o Schema Enforcement: `FiO2=200%` → `status=OUT_OF_BOUNDS`, bloqueio automático

---

## Estrutura do Repositório

```
vlab-stt-llm/
├── src/
│   ├── main.py                  # FastAPI app + endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── extraction.py        # MedicalParameterExtraction (Pydantic)
│   │   └── health.py
│   └── services/
│       ├── stt.py               # GeminiSTT (google.genai SDK moderno)
│       └── extractor.py         # ParameterExtractor (LLM + Schema Enforcement)
├── tests/
│   └── test_main.py             # Testes unitários FastAPI
├── scripts/
│   ├── generate_dataset.py      # Geração de áudios mock via gTTS
│   └── evaluate_pipeline.py    # Benchmarking automatizado com Rich
├── data/
│   ├── audio_samples/           # MP3 sintéticos (6 casos de teste)
│   └── ground_truth.json        # Gabarito de avaliação
├── docs/
│   ├── domain_analysis.md       # Análise de domínio e edge cases clínicos
│   └── evaluation_report.md     # Relatório gerado automaticamente
├── Dockerfile                   # Build multistage (builder + runtime)
├── pyproject.toml
└── poetry.toml                  # virtualenvs.in-project = true
```

---

## Pré-requisitos

- Python 3.11+
- [Poetry](https://python-poetry.org/) 2.x
- Conta Google AI Studio com `GEMINI_API_KEY` válida

---

## Setup

### 1. Clone e instale as dependências

```powershell
git clone https://github.com/Lionzio/vlab-stt-llm.git
cd vlab-stt-llm

# O poetry.toml já configura virtualenvs.in-project = true
# O .venv será criado na raiz do projeto (evita o limite MAX_PATH do Windows)
python -m poetry install
```

### 2. Configure as variáveis de ambiente

```powershell
# Crie o arquivo .env na raiz do projeto
Copy-Item .env.example .env   # se existir, ou crie manualmente

# Edite o .env e adicione sua chave:
# GEMINI_API_KEY=sua_chave_aqui
```

O conteúdo mínimo do `.env`:

```env
GEMINI_API_KEY=AIza...
```

### 3. Verifique a instalação

```powershell
python -m poetry run pytest tests/ -v
```

---

## Executando a API

### Desenvolvimento local

```powershell
python -m poetry run uvicorn src.main:app --reload --port 8000
```

A documentação interativa estará disponível em:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Via Docker

```bash
# Build da imagem multistage
docker build -t vlab-stt-llm:latest .

# Execução com a chave de API injetada
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=sua_chave_aqui \
  vlab-stt-llm:latest
```

> O Dockerfile utiliza build multistage (`builder` → `runtime`) com usuário non-root (`appuser`) para segurança em ambiente corporativo.

---

## Uso via cURL

### Health Check

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Resposta esperada:

```json
{
  "status": "ok",
  "service": "vlab-stt-llm",
  "version": "0.1.0",
  "message": "API operacional"
}
```

### Extração de parâmetros médicos a partir de áudio

```bash
curl -s -X POST http://localhost:8000/api/v1/extract-from-audio \
  -F "audio_file=@data/audio_samples/TC-001.mp3" | python -m json.tool
```

Resposta esperada:

```json
{
  "transcription": "ajustar a frequência respiratória para 15 incursões por minuto",
  "extraction": {
    "intent": "ajustar_parametro",
    "parameter": "frequencia_respiratoria",
    "value": 15.0,
    "unit": "irpm",
    "status": "OK",
    "notes": null
  }
}
```

Testando o cenário de valor fora do limite (`TC-004`):

```bash
curl -s -X POST http://localhost:8000/api/v1/extract-from-audio \
  -F "audio_file=@data/audio_samples/TC-004.mp3" | python -m json.tool
```

```json
{
  "transcription": "configurar f i u dois para 200 por cento",
  "extraction": {
    "intent": "ajustar_parametro",
    "parameter": "fio2",
    "value": 200.0,
    "unit": "%",
    "status": "OUT_OF_BOUNDS",
    "notes": "FiO2 máximo permitido é 100%. Valor rejeitado."
  }
}
```

---

## Scripts Utilitários

### Gerar o dataset de testes

Sintetiza os 6 áudios MP3 via gTTS e reconstrói o `ground_truth.json`:

```powershell
python -m poetry run python scripts/generate_dataset.py
```

### Executar a avaliação automatizada

Roda o pipeline completo contra todos os casos, exibe uma tabela Rich no terminal e gera `docs/evaluation_report.md`:

```powershell
python -m poetry run python scripts/evaluate_pipeline.py
```

O script inclui um throttle de 5s entre casos (`await asyncio.sleep(5)`) para mitigar o rate limit do Gemini Free Tier (20 req/dia). Em ambiente com tier pago, o parâmetro pode ser reduzido a 0.

---

## Decisões de Arquitetura

| Decisão | Justificativa |
|---------|---------------|
| `google.genai` (SDK moderno) | Migração obrigatória — `google.generativeai` foi descontinuado. O novo SDK expõe interface `async` nativa via `client.aio.*` |
| `response_schema=MedicalParameterExtraction` | Schema Enforcement nativo do Gemini: o modelo é forçado a retornar JSON compatível com o schema Pydantic, eliminando parsing frágil |
| Polling `_wait_for_file_active` | O upload de arquivos no Gemini Files API é assíncrono no servidor — submeter `generate_content` com arquivo em `PROCESSING` resulta em resposta vazia |
| `asyncio.wait_for(..., timeout=60)` | Evita que uma chamada travada bloqueie o event loop indefinidamente em produção |
| `finally: delete_file` com `try/except` isolado | Garante cleanup do arquivo remoto sem mascarar a exceção original em caso de falha no cleanup |
| `virtualenvs.in-project = true` | Resolve o limite `MAX_PATH` do Windows causado pela profundidade do caminho padrão do AppData ao usar Python da Microsoft Store |
| Docker multistage (builder + runtime) | A imagem final não carrega Poetry, pip cache ou artefatos de build — reduz superfície de ataque e tamanho da imagem |

---

## Documentação Adicional

- [`docs/domain_analysis.md`](docs/domain_analysis.md) — Análise de domínio: catálogo de parâmetros clínicos, edge cases, ambiguidades fonéticas e estratégia de mitigação
- [`docs/evaluation_report.md`](docs/evaluation_report.md) — Relatório de benchmarking gerado automaticamente pela última execução do pipeline

---

## Aviso de Uso Clínico

> Este projeto é um **protótipo experimental** desenvolvido para fins de avaliação técnica. Não está homologado para uso em ambiente assistencial real. Toda saída do pipeline deve ser revisada por profissional médico habilitado antes de qualquer aplicação clínica.
```