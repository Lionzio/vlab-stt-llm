# src/services/stt.py
"""Serviço de Speech-to-Text via API Google Gemini.

Versão síntese: combina o ciclo de vida robusto de upload/polling/cleanup
da versão original com o roteamento inteligente de chaves e modelos
do GeminiManager. O STT é o único serviço que necessita de lógica especial
de fallback de chave, pois arquivos upados pertencem ao escopo da chave
que os criou — um re-upload é necessário ao trocar de cliente.
"""

from __future__ import annotations

import asyncio
import logging
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

from src.services.gemini_manager import (
    GeminiAuthError,
    GeminiManager,
)

load_dotenv()

logger = logging.getLogger(__name__)

_FILE_READY_TIMEOUT_S: int = 30
_FILE_READY_POLL_INTERVAL_S: float = 2.0
_GENERATION_TIMEOUT_S: int = 60

TRANSCRIPTION_PROMPT = (
    "Você é um assistente de IA em uma UTI. Transcreva exatamente o que é dito "
    "neste áudio em português brasileiro. O áudio contém jargões médicos e nomes "
    "de parâmetros de ventiladores e monitores "
    "(ex: PEEP, FiO2, frequência respiratória, pressão arterial). "
    "Retorne apenas o texto falado, em minúsculas, sem pontuação."
)


# ---------------------------------------------------------------------------
# Hierarquia de exceções STT
# ---------------------------------------------------------------------------


class STTError(Exception):
    """Erro base do serviço de transcrição."""


class STTAuthError(STTError):
    """Falha de autenticação com a API (fatal)."""


class STTQuotaError(STTError):
    """Todas as cotas disponíveis foram esgotadas."""


class STTTranscriptionError(STTError):
    """Falha de transcrição após esgotar todos os fallbacks."""


# ---------------------------------------------------------------------------
# Interface abstrata
# ---------------------------------------------------------------------------


class BaseSTT(ABC):
    """Interface abstrata para implementações de Speech-to-Text."""

    @abstractmethod
    async def transcribe(self, audio_filepath: str) -> str | None:
        """Transcreve um arquivo de áudio para texto.

        Args:
            audio_filepath: Caminho absoluto ou relativo para o arquivo.

        Returns:
            Texto transcrito em lowercase, ou None se resposta vazia.
        """


# ---------------------------------------------------------------------------
# Implementação Gemini
# ---------------------------------------------------------------------------


class GeminiSTT(BaseSTT):
    """STT usando o GeminiManager para roteamento de clientes e modelos.

    Estratégia de fallback específica do STT:
        - Arquivos upados para a Files API pertencem ao escopo da chave
          que os criou. Trocar de cliente exige re-upload explícito.
        - Etapa 1: Upload + transcrição via chave primária.
        - Etapa 2 (se 429 na primária): Re-upload + transcrição via secundária.
        - Etapa 3 (se 429 na secundária): Backoff exponencial via Tenacity.

    Attributes:
        _manager: Instância do GeminiManager para acesso aos clientes.
    """

    def __init__(self, manager: GeminiManager | None = None) -> None:
        """Inicializa o GeminiSTT com um manager opcional.

        Args:
            manager: GeminiManager pré-configurado. Se None, cria um novo
                lendo GEMINI_API_KEY_PRIMARY e GEMINI_API_KEY_SECONDARY do .env.

        Raises:
            STTAuthError: Se a chave primária não estiver disponível.
        """
        try:
            self._manager = manager or GeminiManager()
        except GeminiAuthError as exc:
            raise STTAuthError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    async def _wait_for_file_active(
        self,
        client: genai.Client,
        file_name: str,
    ) -> genai_types.File:
        """Aguarda o arquivo atingir ACTIVE no escopo do cliente informado."""
        deadline = time.monotonic() + _FILE_READY_TIMEOUT_S

        while True:
            file_ref = await client.aio.files.get(name=file_name)

            if file_ref.state == genai_types.FileState.ACTIVE:
                return file_ref

            if file_ref.state != genai_types.FileState.PROCESSING:
                raise STTTranscriptionError(
                    f"Arquivo {file_name} terminou em estado inesperado: "
                    f"{file_ref.state!r}."
                )

            if time.monotonic() > deadline:
                raise STTTranscriptionError(
                    f"Timeout de {_FILE_READY_TIMEOUT_S}s excedido "
                    f"aguardando o arquivo {file_name} ficar ativo."
                )

            await asyncio.sleep(_FILE_READY_POLL_INTERVAL_S)

    async def _upload_and_transcribe(
        self,
        client: genai.Client,
        audio_filepath: str,
    ) -> str | None:
        """Ciclo de vida completo do áudio vinculado a um cliente específico."""
        uploaded_file: genai_types.File | None = None

        try:
            logger.info("[STT] Iniciando upload: %s", audio_filepath)
            uploaded_file = await client.aio.files.upload(file=audio_filepath)
            logger.info(
                "[STT] Upload concluído: %s. Aguardando ACTIVE.",
                uploaded_file.name,
            )

            active_file = await self._wait_for_file_active(client, uploaded_file.name)
            logger.info(
                "[STT] Arquivo %s ativo. Transcrevendo com %s.",
                active_file.name,
                self._manager.flash_model,
            )

            # --- Desativação de Filtros de Segurança para Contexto Clínico ---
            config = genai_types.GenerateContentConfig(
                safety_settings=[
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]
            )

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self._manager.flash_model,
                    contents=[TRANSCRIPTION_PROMPT, active_file],
                    config=config,
                ),
                timeout=_GENERATION_TIMEOUT_S,
            )

            if not response.text:
                logger.warning(
                    "[STT] A API não retornou texto. Candidatos: %s",
                    getattr(response, "candidates", "Nenhum"),
                )
                return None

            return response.text.strip().lower()

        except asyncio.TimeoutError as exc:
            raise STTTranscriptionError(
                f"Timeout de {_GENERATION_TIMEOUT_S}s excedido."
            ) from exc

        finally:
            if uploaded_file is not None:
                try:
                    await client.aio.files.delete(name=uploaded_file.name)
                    logger.info("[STT] Arquivo remoto %s deletado.", uploaded_file.name)
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.warning(
                        "[STT] Não foi possível deletar %s: %s",
                        uploaded_file.name,
                        cleanup_exc,
                    )

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type(STTQuotaError),
        reraise=True,  # Crucial para o FastAPI retornar 429 em vez de 500
        before_sleep=lambda rs: logger.warning(
            "[STT] Ambas as chaves esgotadas. "
            "Backoff exponencial (tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def transcribe(self, audio_filepath: str) -> str | None:
        """Transcreve o áudio com roteamento automático entre chaves."""
        # --- Tentativa 1: chave primária ---
        try:
            return await self._upload_and_transcribe(
                self._manager.primary_client, audio_filepath
            )

        except genai_errors.ClientError as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

            if status in (401, 403):
                raise STTAuthError(
                    f"Falha de autenticação com a chave primária (HTTP {status})."
                ) from exc

            if status != 429:
                raise STTTranscriptionError(
                    f"Erro de API na chave primária (HTTP {status}): {exc}"
                ) from exc

            # --- Tentativa 2: chave secundária (se 429 na primária) ---
            if self._manager.secondary_client is not None:
                try:
                    logger.warning(
                        "[STT] Chave primária com cota excedida (429). "
                        "Tentando chave secundária..."
                    )
                    return await self._upload_and_transcribe(
                        self._manager.secondary_client, audio_filepath
                    )
                except genai_errors.ClientError as exc_sec:
                    status_sec = getattr(exc_sec, "status_code", None) or getattr(
                        exc_sec, "code", None
                    )
                    if status_sec == 429:
                        raise STTQuotaError(
                            "Ambas as chaves de API estão com cota esgotada."
                        ) from exc_sec
                    raise STTTranscriptionError(
                        f"Erro na chave secundária: {exc_sec}"
                    ) from exc_sec
            else:
                raise STTQuotaError(
                    "Cota da chave primária esgotada e sem chave secundária."
                ) from exc
