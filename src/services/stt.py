# src/services/stt.py
"""Serviço de Speech-to-Text via API Google Gemini.

Requer a variável de ambiente GEMINI_API_KEY configurada no .env.
Utiliza o SDK moderno `google-genai` (pacote google.genai).
Inclui resiliência avançada: Exponential Backoff (Tenacity) para rate limits
e Model Fallback (2.5-flash -> 2.0-flash) para indisponibilidade.
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

# Tempo máximo (segundos) aguardando o arquivo ficar disponível no Gemini Files API
_FILE_READY_TIMEOUT_S: int = 30
_FILE_READY_POLL_INTERVAL_S: float = 2.0

# Timeout global da chamada de geração de conteúdo (segundos)
_GENERATION_TIMEOUT_S: int = 60

# Estratégia de Fallback de Modelos
_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

# Prompt contextualizado (Mitigação de erros de STT - Requisito 5.6)
TRANSCRIPTION_PROMPT = (
    "Você é um assistente de IA em uma UTI. Transcreva exatamente o que é dito neste áudio em "
    "português brasileiro. O áudio contém jargões médicos e nomes de parâmetros de ventiladores "
    "e monitores (ex: PEEP, FiO2, frequência respiratória, pressão arterial). "
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
    """Implementação de STT usando o SDK google.genai.

    Attributes:
        _client: Instância do cliente genai autenticado.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Gemini com a chave de API do ambiente."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise STTAuthError(
                "GEMINI_API_KEY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )
        self._client = genai.Client(api_key=api_key)

    async def _wait_for_file_active(self, file_name: str) -> genai_types.File:
        """Aguarda o arquivo atingir o estado ACTIVE via polling."""
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
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type(STTQuotaError),
        before_sleep=lambda rs: logger.warning(
            f"[STT] Cota 429 atingida. Aguardando para tentar novamente (Tentativa {rs.attempt_number}/5)..."
        ),
    )
    async def _generate_content_with_fallback(
        self, active_file: genai_types.File
    ) -> str | None:
        """Processa a geração lidando com Retry (429) e Fallback de Modelos (500)."""
        last_exc = None

        for model_id in _MODELS:
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model_id,
                        contents=[TRANSCRIPTION_PROMPT, active_file],
                    ),
                    timeout=_GENERATION_TIMEOUT_S,
                )
                return response.text.strip().lower() if response.text else None

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

                if status in (401, 403):
                    # Erro de credencial é fatal, não tenta outro modelo
                    raise STTAuthError(
                        "Falha de autenticação com a API Gemini."
                    ) from exc

                if status == 429:
                    # Erro de Cota é global por projeto. Interrompe os modelos e dispara o Tenacity.
                    raise STTQuotaError(
                        "Cota da API Gemini excedida (Rate Limit)."
                    ) from exc

                last_exc = exc
                logger.warning(
                    f"[STT] Modelo {model_id} falhou ({status}). Tentando fallback..."
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[STT] Modelo {model_id} falhou com erro inesperado: {exc}"
                )

        raise STTTranscriptionError(
            f"Todos os modelos falharam na transcrição. Último erro: {last_exc}"
        ) from last_exc

    async def transcribe(self, audio_filepath: str) -> str | None:
        """Upload do áudio, aguarda ativação, gera texto e deleta o arquivo."""
        uploaded_file: genai_types.File | None = None

        try:
            logger.info("Iniciando upload do arquivo: %s", audio_filepath)
            uploaded_file = await self._client.aio.files.upload(
                file=audio_filepath,
            )
            logger.info(
                "Upload concluído: %s. Aguardando estado ACTIVE.", uploaded_file.name
            )

            active_file = await self._wait_for_file_active(uploaded_file.name)
            logger.info("Arquivo %s ativo. Iniciando transcrição.", active_file.name)

            # Chama a função privada que cuida dos retentativas e fallbacks
            return await self._generate_content_with_fallback(active_file)

        finally:
            # Cleanup garantido no servidor do Google
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
