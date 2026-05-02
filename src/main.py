from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import HealthCheckResponse

APP_VERSION = "0.1.0"
SERVICE_NAME = "vlab-stt-llm"

app = FastAPI(
    title="VLab STT + LLM Pipeline",
    description=(
        "Pipeline de IA para transcrição de áudio (STT) e extração de dados "
        "médicos estruturados via LLM. Fornece endpoints para ingestão de áudio, "
        "processamento assíncrono e recuperação de resultados."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Verifica se a API está operacional.",
    tags=["Observability"],
)
async def health_check() -> HealthCheckResponse:
    """Retorna o status de saúde da aplicação.

    Returns:
        HealthCheckResponse: Payload com status, nome do serviço, versão e mensagem.
    """
    return HealthCheckResponse(
        status="ok",
        service=SERVICE_NAME,
        version=APP_VERSION,
        message="API operacional",
    )
