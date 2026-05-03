# src/services/stt.py
"""Serviço de Speech-to-Text via API Google Gemini.

Requer a variável de ambiente GEMINI_API_KEY configurada no .env.
Utiliza o SDK moderno `google-genai` (pacote google.genai).
Inclui resiliência avançada: Exponential Backoff (Tenacity) para rate limits
e Model Fallback sequencial para indisponibilidade de modelo.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

_FILE_READY_TIMEOUT_S: int = 30
_FILE_READY_POLL_INTERVAL_S: float = 2.0
_GENERATION_TIMEOUT_S: int = 60

# Modelos estáveis em ordem de preferência (403 em 2.5 → fallback para 2.0)
_MODELS: list[str] = ["gemini-2.0-flash", "gemini-1.5-flash"]

TRANSCRIPTION_PROMPT = (
    "Você é um assistente de IA em uma UTI. Transcreva exatamente o que é dito "
    "neste áudio em português brasileiro. O áudio contém jargões médicos e nomes "
    "de parâmetros de ventiladores e monitores (ex: PEEP, FiO2, frequência "
    "respiratória, pressão arterial). "
    "Retorne apenas o texto falado, em minúsculas, sem pontuação."
)


# ---------------------------------------------------------------------------
# Hierarquia de exceções
# ---------------------------------------------------------------------------


class STTError(Exception):
    """Erro base do serviço de transcrição."""


class STTAuthError(STTError):
    """Falha de autenticação com a API."""


class STTQuotaError(STTError):
    """Cota da API excedida (Rate Limit 429)."""


class STTTranscriptionError(STTError):
    """Falha durante a transcrição do áudio após esgotar fallbacks."""


# ---------------------------------------------------------------------------
# Interface abstrata
# ---------------------------------------------------------------------------


class BaseSTT(ABC):
    """Interface abstrata para implementações de Speech-to-Text."""

    @abstractmethod
    async def transcribe(self, audio_filepath: str) -> str | None:
        """Transcreve um arquivo de áudio para texto.

        Args:
            audio_filepath: Caminho absoluto ou relativo para o arquivo de áudio.

        Returns:
            Texto transcrito em lowercase, ou None se a resposta for vazia.
        """


# ---------------------------------------------------------------------------
# Implementação Gemini
# ---------------------------------------------------------------------------


class GeminiSTT(BaseSTT):
    """Implementação de STT usando o SDK google.genai com fallback de modelos.

    Tenta os modelos em `_MODELS` sequencialmente. Erros 429 disparam
    Exponential Backoff via Tenacity antes de retentar. Erros 401/403 são
    fatais e interrompem o fluxo imediatamente.

    Attributes:
        _client: Instância do cliente genai autenticado.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Gemini com a chave de API do ambiente.

        Raises:
            STTAuthError: Se GEMINI_API_KEY não estiver definida.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise STTAuthError(
                "GEMINI_API_KEY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )
        self._client = genai.Client(api_key=api_key)

    async def _wait_for_file_active(self, file_name: str) -> genai_types.File:
        """Aguarda o arquivo atingir o estado ACTIVE via polling assíncrono.

        Args:
            file_name: Identificador do arquivo remoto (ex: "files/abc123").

        Returns:
            Objeto File com state == ACTIVE.

        Raises:
            STTTranscriptionError: Se o arquivo não atingir ACTIVE no prazo
                ou terminar em estado de falha.
        """
        deadline = time.monotonic() + _FILE_READY_TIMEOUT_S

        while True:
            file_ref = await self._client.aio.files.get(name=file_name)

            if file_ref.state == genai_types.FileState.ACTIVE:
                return file_ref

            if file_ref.state != genai_types.FileState.PROCESSING:
                raise STTTranscriptionError(
                    f"Arquivo {file_name} terminou em estado inesperado: "
                    f"{file_ref.state!r}."
                )

            if time.monotonic() > deadline:
                raise STTTranscriptionError(
                    f"Timeout de {_FILE_READY_TIMEOUT_S}s excedido aguardando "
                    f"o arquivo {file_name} ficar ativo."
                )

            await asyncio.sleep(_FILE_READY_POLL_INTERVAL_S)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(STTQuotaError),
        before_sleep=lambda rs: logger.warning(
            "[STT] Cota 429 atingida. Backoff exponencial " "(tentativa %d/5)...",
            rs.attempt_number,
        ),
    )
    async def _generate_content_with_fallback(
        self, active_file: genai_types.File
    ) -> str | None:
        """Tenta gerar a transcrição iterando pelos modelos disponíveis.

        O decorador @retry captura STTQuotaError (429) e aplica Exponential
        Backoff antes de retentar. STTAuthError (401/403) é relançada
        imediatamente sem retry, pois é um erro fatal de configuração.

        Args:
            active_file: Objeto File no estado ACTIVE no Gemini Files API.

        Returns:
            Texto transcrito ou None se a resposta estiver vazia.

        Raises:
            STTAuthError: Em caso de erro 401/403 (fatal, sem retry).
            STTQuotaError: Em caso de erro 429 (capturado pelo Tenacity).
            STTTranscriptionError: Se todos os modelos falharem.
        """
        last_exc: Exception | None = None

        for model_id in _MODELS:
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model_id,
                        contents=[TRANSCRIPTION_PROMPT, active_file],
                    ),
                    timeout=_GENERATION_TIMEOUT_S,
                )
                logger.info("[STT] Transcrição bem-sucedida com %s.", model_id)
                return response.text.strip().lower() if response.text else None

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

                if status in (401, 403):
                    # Erro fatal de autenticação — não tenta outros modelos
                    raise STTAuthError(
                        "Falha de autenticação com a API Gemini " f"(HTTP {status})."
                    ) from exc

                if status == 429:
                    # Cota global do projeto — interrompe o loop e aciona Tenacity
                    raise STTQuotaError(
                        "Cota da API Gemini excedida (HTTP 429)."
                    ) from exc

                last_exc = exc
                logger.warning(
                    "[STT] Modelo %s falhou (HTTP %s). Tentando próximo...",
                    model_id,
                    status,
                )

            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "[STT] Modelo %s excedeu timeout de %ds. " "Tentando próximo...",
                    model_id,
                    _GENERATION_TIMEOUT_S,
                )

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "[STT] Modelo %s falhou com erro inesperado: %s",
                    model_id,
                    exc,
                )

        raise STTTranscriptionError(
            f"Todos os modelos falharam na transcrição. " f"Último erro: {last_exc}"
        ) from last_exc

    async def transcribe(self, audio_filepath: str) -> str | None:
        """Orquestra upload, polling de estado, geração e cleanup remoto.

        Args:
            audio_filepath: Caminho para o arquivo de áudio local.

        Returns:
            Texto transcrito em lowercase sem espaços extras, ou None.

        Raises:
            STTAuthError: Em falha de autenticação.
            STTQuotaError: Se a cota for esgotada após todos os retries.
            STTTranscriptionError: Para outros erros de transcrição.
        """
        uploaded_file: genai_types.File | None = None

        try:
            logger.info("Iniciando upload do arquivo: %s", audio_filepath)
            uploaded_file = await self._client.aio.files.upload(
                file=audio_filepath,
            )
            logger.info(
                "Upload concluído: %s. Aguardando estado ACTIVE.",
                uploaded_file.name,
            )

            active_file = await self._wait_for_file_active(uploaded_file.name)
            logger.info("Arquivo %s ativo. Iniciando transcrição.", active_file.name)

            return await self._generate_content_with_fallback(active_file)

        finally:
            if uploaded_file is not None:
                try:
                    await self._client.aio.files.delete(name=uploaded_file.name)
                    logger.info("Arquivo remoto %s deletado.", uploaded_file.name)
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.warning(
                        "Não foi possível deletar arquivo remoto %s: %s",
                        uploaded_file.name,
                        cleanup_exc,
                    )
