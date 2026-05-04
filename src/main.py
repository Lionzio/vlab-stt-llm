# src/main.py
"""Aplicação FastAPI — Pipeline STT + LLM para extração de parâmetros médicos.

Expõe dois endpoints:
    GET  /health                        — Verificação de saúde da aplicação.
    POST /api/v1/extract-from-audio     — Pipeline completo: áudio → transcrição → extração.

Utiliza Injeção de Dependência (DI) para compartilhar a mesma instância
do GeminiManager (estado, limites de cota e conexões HTTP) entre todas as requisições.
Implementa o padrão Graceful Degradation (Fallback offline) em caso de falha na API.
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

# Importando as vias alternativas (Graceful Degradation)
from src.services.heuristic_extractor import HeuristicParameterExtractor
from src.services.mock_stt import MockSTT

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
        extraction: Parâmetros médicos estruturados extraídos pelo LLM ou fallback.
    """

    transcription: str = Field(
        ...,
        description="Texto transcrito pelo STT a partir do áudio.",
        examples=["ajustar a frequência respiratória para quinze incursões por minuto"],
    )
    extraction: MedicalParameterExtraction = Field(
        ...,
        description="Parâmetros médicos extraídos e validados pelo LLM ou heurística.",
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
        "médicos estruturados via LLM. Implementa Graceful Degradation."
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
    """Recupera a instância global do GeminiManager anexada à aplicação."""
    return request.app.state.gemini_manager


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
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
        "Recebe um arquivo de áudio, transcreve o conteúdo "
        "via Gemini STT e extrai parâmetros médicos estruturados via LLM. "
        "Possui fallback automático para regras heurísticas em caso de limite de cota."
    ),
    tags=["Pipeline"],
)
async def extract_from_audio(
    audio_file: UploadFile,
    manager: GeminiManager = Depends(get_gemini_manager),
) -> PipelineResponse:
    """Processa um arquivo de áudio pelo pipeline STT → LLM (com Fallbacks)."""
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

        # --------------------------------------------------------------------
        # 2. TENTA STT via IA (Com Fallback para MOCK)
        # --------------------------------------------------------------------
        transcription: str | None = None
        try:
            stt = GeminiSTT(manager=manager)
            transcription = await stt.transcribe(temp_path)
        except RetryError as exc:
            # INTERCEPTA O 429 DO STT (Cota Esgotada após retries)
            original = exc.last_attempt.exception()
            if isinstance(original, (STTQuotaError, GeminiQuotaError)):
                logger.warning(
                    "Cota do STT esgotada. Acionando MockSTT (Graceful Degradation)."
                )
                mock_stt = MockSTT()
                # Passa o nome original do arquivo para o mock identificar o cenário
                transcription = await mock_stt.transcribe(audio_file.filename)
            else:
                raise  # Repropaga se for outro erro interno
        except (STTQuotaError, GeminiQuotaError):
            logger.warning(
                "Cota do STT esgotada. Acionando MockSTT (Graceful Degradation)."
            )
            mock_stt = MockSTT()
            transcription = await mock_stt.transcribe(audio_file.filename)

        if not transcription:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A transcrição do áudio retornou resultado vazio.",
            )

        logger.info("Transcrição concluída: %r", transcription)

        # --------------------------------------------------------------------
        # 3. TENTA EXTRAÇÃO via IA (Com Fallback para HEURÍSTICA OFFLINE)
        # --------------------------------------------------------------------
        extraction: MedicalParameterExtraction | None = None
        try:
            extractor = ParameterExtractor(manager=manager)
            extraction = await extractor.extract(transcription)
        except RetryError as exc:
            # INTERCEPTA O 429 DO EXTRACTOR (Cota Esgotada após retries)
            original = exc.last_attempt.exception()
            if isinstance(original, (STTQuotaError, GeminiQuotaError)):
                logger.warning(
                    "Cota do LLM esgotada. Acionando Extrator Heurístico (Graceful Degradation)."
                )
                heuristic_extractor = HeuristicParameterExtractor()
                extraction = heuristic_extractor.extract(transcription)
            else:
                raise
        except (STTQuotaError, GeminiQuotaError):
            logger.warning(
                "Cota do LLM esgotada. Acionando Extrator Heurístico (Graceful Degradation)."
            )
            heuristic_extractor = HeuristicParameterExtractor()
            extraction = heuristic_extractor.extract(transcription)

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
        raise

    except (STTAuthError, GeminiAuthError, ValueError) as exc:
        logger.error("Falha de autenticação com a API Gemini: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Falha de autenticação com o serviço de IA. Verifique as chaves de API.",
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
    """Extrai a extensão de um nome de arquivo para uso no arquivo temporário."""
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()
