# ==============================================================================
# Stage 1 — Builder: instala dependências em ambiente isolado
# ==============================================================================
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install --upgrade pip && \
    pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copia somente os manifests primeiro — maximiza o cache de camadas do Docker
COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && \
    rm -rf ${POETRY_CACHE_DIR}

# ==============================================================================
# Stage 2 — Runtime: imagem final enxuta sem artefatos de build
# ==============================================================================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cria usuário non-root para execução segura
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

# Copia pacotes instalados pelo builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código-fonte da aplicação
COPY src/ ./src/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]