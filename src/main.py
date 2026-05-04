# src/main.py
"""Aplicação FastAPI — Pipeline STT + LLM para extração de parâmetros médicos.

Expõe dois endpoints:
    GET  /health                        — Verificação de saúde da aplicação.
    POST /api/v1/extract-from-audio     — Pipeline completo: áudio → transcrição → extração.

Utiliza Injeção de Dependência (DI) para compartilhar a mesma instância
do GeminiManager (estado, limites de cota e conexões HTTP) entre todas as requisições.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import RetryError

from src.schemas import HealthCheckResponse
from src.services.extractor import MedicalParameterExtraction, ParameterExtractor
from src.services.gemini_manager import (
    GeminiAuthError,
    GeminiManager,
    GeminiQuotaError,
)
from src.services.stt import GeminiSTT, STTAuthError, STTError, STTQuotaError

# Configurando o logger principal da aplicação para forçar a exibição do nível INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:\t  %(message)s",
    force=True,  # Sobrescreve as configurações de log do uvicorn
)
logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
SERVICE_NAME = "vlab-stt-llm"

# ---------------------------------------------------------------------------
# Schemas de I/O do pipeline
# ---------------------------------------------------------------------------


class PipelineResponse(BaseModel):
    """Resposta consolidada do pipeline STT + extração de parâmetros.

    Attributes:
        transcription: Texto transcrito pelo STT a partir do áudio enviado.
        extraction: Parâmetros médicos estruturados extraídos pelo LLM.
    """

    transcription: str = Field(
        ...,
        description="Texto transcrito pelo STT a partir do áudio.",
        examples=["ajustar a frequência respiratória para quinze incursões por minuto"],
    )
    extraction: MedicalParameterExtraction = Field(
        ...,
        description="Parâmetros médicos extraídos e validados pelo LLM.",
    )


# ---------------------------------------------------------------------------
# Lifespan — inicialização e teardown da aplicação
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Instancia o GeminiManager (Singleton virtual via App State) no startup,
    garantindo que as dependências de ambiente (chaves de API) estejam
    disponíveis antes da aplicação aceitar tráfego.

    Args:
        application: Instância da aplicação FastAPI.

    Yields:
        Controle para a aplicação em execução.

    Raises:
        RuntimeError: Se variáveis de ambiente obrigatórias estiverem ausentes.
    """
    try:
        # Inicializa o manager centralizado uma única vez
        application.state.gemini_manager = GeminiManager()
        logger.info("GeminiManager inicializado e anexado ao estado da aplicação.")
    except GeminiAuthError as exc:
        raise RuntimeError(
            f"Erro na configuração da API Gemini no startup: {exc}"
        ) from exc

    logger.info("Aplicação %s v%s iniciada.", SERVICE_NAME, APP_VERSION)
    yield
    logger.info("Aplicação %s encerrada.", SERVICE_NAME)


# ---------------------------------------------------------------------------
# Instância FastAPI
# ---------------------------------------------------------------------------

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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependências (DI)
# ---------------------------------------------------------------------------


def get_gemini_manager(request: Request) -> GeminiManager:
    """Recupera a instância global do GeminiManager anexada à aplicação.

    Permite que os endpoints consumam o manager de forma limpa e mockável.
    """
    return request.app.state.gemini_manager


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Verifica se a API está operacional.",
    tags=["Observability"],
)
async def health_check() -> HealthCheckResponse:
    """Retorna o status de saúde da aplicação."""
    return HealthCheckResponse(
        status="ok",
        service=SERVICE_NAME,
        version=APP_VERSION,
        message="API operacional",
    )


@app.post(
    "/api/v1/extract-from-audio",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Extração de parâmetros médicos a partir de áudio",
    description=(
        "Recebe um arquivo de áudio via multipart/form-data, transcreve o conteúdo "
        "via Gemini STT e extrai parâmetros médicos estruturados via LLM. "
        "Retorna a transcrição e a extração validada em JSON."
    ),
    tags=["Pipeline"],
    responses={
        400: {"description": "Transcrição vazia ou parâmetro de entrada inválido."},
        422: {"description": "Arquivo ausente ou tipo de conteúdo incorreto."},
        429: {"description": "Cota da API Gemini excedida em todas as chaves."},
        500: {"description": "Falha interna no STT ou no extractor."},
        503: {"description": "Falha de autenticação com a API Gemini."},
    },
)
async def extract_from_audio(
    audio_file: UploadFile,
    manager: GeminiManager = Depends(get_gemini_manager),
) -> PipelineResponse:
    """Processa um arquivo de áudio pelo pipeline STT → LLM.

    Fluxo:
        1. Persiste o áudio recebido em arquivo temporário no disco.
        2. Instancia STT e Extractor injetando o Manager compartilhado.
        3. Transcreve o áudio via GeminiSTT.
        4. Extrai parâmetros médicos do texto via ParameterExtractor.
        5. Remove o arquivo temporário no bloco finally (garantido).
    """
    temp_path: str | None = None

    try:
        # 1. Persistência do áudio em arquivo temporário
        suffix = _extract_suffix(audio_file.filename)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="vlab_audio_"
        ) as tmp:
            temp_path = tmp.name
            content = await audio_file.read()
            tmp.write(content)

        logger.info("Arquivo temporário criado: %s (%d bytes)", temp_path, len(content))

        # 2. Transcrição via GeminiSTT (com Manager Injetado)
        stt = GeminiSTT(manager=manager)
        transcription: str | None = await stt.transcribe(temp_path)

        if not transcription:
            logger.warning("STT retornou transcrição vazia para o arquivo enviado.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "A transcrição do áudio retornou resultado vazio. "
                    "Verifique se o arquivo contém fala audível e tente novamente."
                ),
            )

        logger.info("Transcrição concluída: %r", transcription)

        # 3. Extração de parâmetros médicos via ParameterExtractor (com Manager Injetado)
        extractor = ParameterExtractor(manager=manager)
        extraction: MedicalParameterExtraction | None = await extractor.extract(
            transcription
        )

        if extraction is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Falha ao extrair dados. O modelo retornou um resultado vazio ou inválido.",
            )

        logger.info(
            "Extração concluída: intent=%s, status=%s",
            extraction.intent,
            extraction.status,
        )

        return PipelineResponse(transcription=transcription, extraction=extraction)

    except HTTPException:
        # Re-propaga HTTPExceptions (400, 429, 503) geradas manualmente acima
        raise

    except RetryError as exc:
        # Desempacota a exceção original encapsulada pelo Tenacity.
        original = exc.last_attempt.exception()
        if isinstance(original, (STTQuotaError, GeminiQuotaError)):
            logger.warning(
                "RetryError desempacotado — cota esgotada após todos os retries: %s",
                original,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Todas as cotas do serviço excedidas. Tente novamente mais tarde.",
            ) from original

        logger.error(
            "RetryError desempacotado — falha inesperada após retries: %s",
            original,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha após múltiplas tentativas: {original}",
        ) from original

    except (STTAuthError, GeminiAuthError, ValueError) as exc:
        logger.error("Falha de autenticação com a API Gemini: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Falha de autenticação com o serviço de inteligência artificial. "
                "Verifique a configuração das chaves de API."
            ),
        ) from exc

    except (STTQuotaError, GeminiQuotaError) as exc:
        logger.warning("Cotas das APIs esgotadas: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Todas as cotas do serviço excedidas. Tente novamente mais tarde.",
        ) from exc

    except STTError as exc:
        logger.error("Falha no serviço STT: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha no serviço de transcrição: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Erro inesperado no pipeline de extração: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno inesperado. Consulte os logs do servidor.",
        ) from exc

    finally:
        # 4. Limpeza garantida do arquivo temporário
        if temp_path is not None:
            try:
                os.remove(temp_path)
                logger.debug("Arquivo temporário removido: %s", temp_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Não foi possível remover arquivo temporário %s: %s",
                    temp_path,
                    cleanup_exc,
                )


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _extract_suffix(filename: str | None) -> str:
    """Extrai a extensão de um nome de arquivo para uso no arquivo temporário.

    Args:
        filename: Nome original do arquivo enviado. Pode ser None.

    Returns:
        Extensão com ponto (ex: ".mp3") ou string vazia se não identificável.
    """
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()
