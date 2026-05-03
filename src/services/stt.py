# src/services/stt.py

"""Serviço de Speech-to-Text via API Google Gemini.

Requer a variável de ambiente GEMINI_API_KEY configurada no .env.
Utiliza o SDK moderno `google-genai` (pacote google.genai).
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

load_dotenv()

logger = logging.getLogger(__name__)

# Tempo máximo (segundos) aguardando o arquivo ficar disponível no Gemini Files API
_FILE_READY_TIMEOUT_S: int = 30
_FILE_READY_POLL_INTERVAL_S: float = 2.0

# Timeout global da chamada de geração de conteúdo (segundos)
_GENERATION_TIMEOUT_S: int = 60

_MODEL_ID: str = "gemini-2.5-flash"

TRANSCRIPTION_PROMPT = (
    "Transcreva exatamente o que é dito neste áudio em português brasileiro. "
    "Retorne apenas o texto falado, sem pontuação adicional ou comentários."
)


# ---------------------------------------------------------------------------
# Hierarquia de exceções
# ---------------------------------------------------------------------------


class STTError(Exception):
    """Erro base do serviço de transcrição."""


class STTAuthError(STTError):
    """Falha de autenticação com a API."""


class STTQuotaError(STTError):
    """Cota da API excedida."""


class STTTranscriptionError(STTError):
    """Falha durante a transcrição do áudio."""


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
            Texto transcrito em lowercase, ou None se a resposta do modelo
            for vazia.

        Raises:
            STTAuthError: Se a autenticação com a API falhar.
            STTQuotaError: Se a cota da API for excedida.
            STTTranscriptionError: Se a transcrição falhar por outro motivo.
        """


# ---------------------------------------------------------------------------
# Implementação Gemini
# ---------------------------------------------------------------------------


class GeminiSTT(BaseSTT):
    """Implementação de STT usando o SDK google.genai com gemini-2.5-flash.

    Utiliza a interface assíncrona nativa do SDK moderno (`client.aio.*`)
    para upload, polling de estado e geração de conteúdo.

    Attributes:
        _client: Instância do cliente genai autenticado.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Gemini com a chave de API do ambiente.

        Raises:
            STTAuthError: Se GEMINI_API_KEY não estiver definida ou for inválida.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise STTAuthError(
                "GEMINI_API_KEY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )
        self._client = genai.Client(api_key=api_key)

    async def _wait_for_file_active(self, file_name: str) -> genai_types.File:
        """Aguarda o arquivo atingir o estado ACTIVE no Gemini Files API.

        O upload é processado de forma assíncrona no servidor. Submeter uma
        requisição de geração com um arquivo ainda em PROCESSING resulta em
        erro ou resposta vazia. Este método faz polling com deadline explícito.

        Args:
            file_name: Identificador do arquivo remoto (ex: "files/abc123").

        Returns:
            Objeto File com state == ACTIVE.

        Raises:
            STTTranscriptionError: Se o arquivo não atingir ACTIVE dentro do
                timeout ou terminar em estado de falha (ex: FAILED).
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

    async def transcribe(self, audio_filepath: str) -> str | None:
        """Transcreve um arquivo de áudio via Gemini Files API.

        Fluxo:
            1. Upload assíncrono do arquivo para o Gemini Files API.
            2. Polling assíncrono até o arquivo atingir o estado ACTIVE.
            3. Chamada `generate_content` com timeout explícito via
               `asyncio.wait_for`.
            4. Deleção garantida do arquivo remoto no bloco `finally`,
               sem mascarar exceções originais em caso de falha no cleanup.

        Args:
            audio_filepath: Caminho para o arquivo de áudio (.mp3, .wav, etc.).

        Returns:
            Texto transcrito em lowercase sem espaços extras, ou None se
            a resposta do modelo for vazia.

        Raises:
            STTAuthError: Se a API retornar erro de autenticação (401/403).
            STTQuotaError: Se a cota da API for excedida (429).
            STTTranscriptionError: Para timeout ou outros erros de transcrição.
        """
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

            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=_MODEL_ID,
                    contents=[TRANSCRIPTION_PROMPT, active_file],
                ),
                timeout=_GENERATION_TIMEOUT_S,
            )

            text: str | None = response.text
            if not text:
                logger.warning(
                    "Gemini retornou resposta vazia para o arquivo %s.",
                    active_file.name,
                )
                return None

            return text.strip().lower()

        except asyncio.TimeoutError as exc:
            raise STTTranscriptionError(
                f"Timeout de {_GENERATION_TIMEOUT_S}s excedido durante a geração."
            ) from exc

        except genai_errors.ClientError as exc:
            # O SDK google.genai mapeia 401/403 para ClientError com status 401/403
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status in (401, 403):
                raise STTAuthError("Falha de autenticação com a API Gemini.") from exc
            if status == 429:
                raise STTQuotaError("Cota da API Gemini excedida.") from exc
            raise STTTranscriptionError(
                f"Erro de cliente na API Gemini (status={status}): {exc}"
            ) from exc

        except genai_errors.ServerError as exc:
            raise STTTranscriptionError(
                f"Erro de servidor na API Gemini: {exc}"
            ) from exc

        except genai_errors.APIError as exc:
            raise STTTranscriptionError(
                f"Erro inesperado na API Gemini: {exc}"
            ) from exc

        finally:
            if uploaded_file is not None:
                try:
                    await self._client.aio.files.delete(name=uploaded_file.name)
                    logger.info("Arquivo remoto %s deletado.", uploaded_file.name)
                except Exception as cleanup_exc:  # noqa: BLE001
                    # Falha no cleanup não deve mascarar a exceção principal.
                    # Arquivos expiram automaticamente após 48h no Gemini Files API.
                    logger.warning(
                        "Não foi possível deletar arquivo remoto %s: %s",
                        uploaded_file.name,
                        cleanup_exc,
                    )
