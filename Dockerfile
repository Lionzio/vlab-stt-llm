# ==============================================================================
# Imagem base
# ==============================================================================
FROM python:3.11-slim

# Impede a criação de arquivos .pyc e diz ao Poetry para NÃO criar .venv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Instala o Poetry globalmente no container
RUN pip install --no-cache-dir poetry

# Copia APENAS os manifests primeiro (Isso faz o build ser instantâneo nas próximas vezes)
COPY pyproject.toml poetry.lock ./

# Instala apenas as dependências de produção
# A flag --no-root é a mágica que evita o erro "No file/folder found"
RUN poetry install --only main --no-root

# Agora copia o resto do código da aplicação
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]