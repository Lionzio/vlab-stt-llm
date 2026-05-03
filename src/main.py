# src/main.py

"""Aplicação FastAPI — Pipeline STT + LLM para extração de parâmetros médicos.

Expõe dois endpoints:
    GET  /health                        — Verificação de saúde da aplicação.
    POST /api/v1/extract-from-audio     — Pipeline completo: áudio → transcrição → extração.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.schemas import HealthCheckResponse
from src.services.extractor import MedicalParameterExtraction, ParameterExtractor
from src.services.stt import GeminiSTT, STTAuthError, STTError, STTQuotaError

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

    Valida no startup que as dependências de ambiente estão disponíveis,
    evitando que a aplicação suba com configuração incompleta.

    Args:
        application: Instância da aplicação FastAPI.

    Yields:
        Controle para a aplicação em execução.

    Raises:
        RuntimeError: Se variáveis de ambiente obrigatórias estiverem ausentes.
    """
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. "
            "Defina a variável no arquivo .env antes de iniciar a aplicação."
        )
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
        429: {"description": "Cota da API Gemini excedida."},
        500: {"description": "Falha interna no STT ou no extractor."},
        503: {"description": "Falha de autenticação com a API Gemini."},
    },
)
async def extract_from_audio(
    audio_file: UploadFile,
) -> PipelineResponse:
    """Processa um arquivo de áudio pelo pipeline STT → LLM.

    Fluxo:
        1. Persiste o áudio recebido em arquivo temporário no disco.
        2. Transcreve o áudio via GeminiSTT.
        3. Extrai parâmetros médicos do texto via ParameterExtractor.
        4. Remove o arquivo temporário no bloco finally (garantido).

    Args:
        audio_file: Arquivo de áudio enviado via multipart/form-data.
            Formatos aceitos: .mp3, .wav, .ogg, .flac, .m4a.

    Returns:
        PipelineResponse: Transcrição e extração estruturada de parâmetros médicos.

    Raises:
        HTTPException(400): Se a transcrição retornar texto vazio.
        HTTPException(429): Se a cota da API Gemini for excedida.
        HTTPException(500): Se o STT ou o extractor falharem por erro interno.
        HTTPException(503): Se a autenticação com a API Gemini falhar.
    """
    temp_path: str | None = None

    try:
        # ------------------------------------------------------------------
        # 1. Persistência do áudio em arquivo temporário
        # ------------------------------------------------------------------
        suffix = _extract_suffix(audio_file.filename)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="vlab_audio_"
        ) as tmp:
            temp_path = tmp.name
            content = await audio_file.read()
            tmp.write(content)

        logger.info("Arquivo temporário criado: %s (%d bytes)", temp_path, len(content))

        # ------------------------------------------------------------------
        # 2. Transcrição via GeminiSTT
        # ------------------------------------------------------------------
        stt = GeminiSTT()
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

        # ------------------------------------------------------------------
        # 3. Extração de parâmetros médicos via ParameterExtractor
        # ------------------------------------------------------------------
        extractor = ParameterExtractor()
        extraction: MedicalParameterExtraction = await extractor.extract(transcription)

        logger.info(
            "Extração concluída: intent=%s, status=%s",
            extraction.intent,
            extraction.status,
        )

        return PipelineResponse(transcription=transcription, extraction=extraction)

    except HTTPException:
        # Re-propaga HTTPExceptions sem transformação
        raise

    except STTAuthError as exc:
        logger.error("Falha de autenticação com a API Gemini: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Falha de autenticação com o serviço de transcrição. "
                "Verifique a configuração da GEMINI_API_KEY."
            ),
        ) from exc

    except STTQuotaError as exc:
        logger.warning("Cota da API Gemini excedida: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Cota do serviço de transcrição excedida. Tente novamente mais tarde.",
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
        # ------------------------------------------------------------------
        # 4. Limpeza garantida do arquivo temporário
        # ------------------------------------------------------------------
        if temp_path is not None:
            try:
                os.remove(temp_path)
                logger.debug("Arquivo temporário removido: %s", temp_path)
            except OSError as cleanup_exc:
                # Falha no cleanup nunca deve mascarar o erro original.
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
