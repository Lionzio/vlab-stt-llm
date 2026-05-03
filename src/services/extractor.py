# src/services/extractor.py
"""Serviço de Extração de Parâmetros Médicos — V1 (Direct Schema).

Usa Schema Enforcement nativo do Gemini com response_schema=Pydantic e
Exponential Backoff via Tenacity para erros 429 (quota).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.schemas.extraction import MedicalParameterExtraction

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """Você é um especialista em equipamentos médicos de suporte à vida e NLP.
Sua missão é extrair parâmetros, valores e intenções de transcrições de comandos de voz (STT).

REGRAS ESTRITAS:
1. MAPEAMENTO DE PARÂMETROS: Padronize os nomes
   (ex: "f i o dois" -> "fio2", "p a" -> "pressao_arterial").
2. INFERÊNCIA DE UNIDADE: Se o usuário pedir para "colocar a peep em cinco"
   e não disser a unidade, você DEVE inferir que é "cmH2O" com base no
   domínio clínico, e definir o status como "OK_INFERRED_UNIT".
3. LIMITES E SEGURANÇA: Se o valor for clinicamente absurdo para o parâmetro
   (ex: FiO2 maior que 100), o status DEVE ser "OUT_OF_BOUNDS".
4. AMBIGUIDADES: Se for "doze por oito", o value é null e o status
   "REQUIRES_CLARIFICATION", pois são dois valores (sistólica/diastólica).

EXEMPLOS (Few-Shot):
Input: "ajustar a frequência respiratória para quinze incursões por minuto"
Output: intent="ajustar_parametro", parameter="frequencia_respiratoria",
        value=15.0, unit="irpm", status="OK"

Input: "coloca a peep em cinco"
Output: intent="ajustar_parametro", parameter="peep",
        value=5.0, unit="cmH2O", status="OK_INFERRED_UNIT"
"""

# Modelos estáveis em ordem de preferência
_MODELS: list[str] = ["gemini-2.0-flash", "gemini-1.5-flash"]


def _is_quota_error(exc: BaseException) -> bool:
    """Verifica se a exceção representa um erro de cota (HTTP 429).

    Args:
        exc: Exceção a inspecionar.

    Returns:
        True se for um erro de cota do Gemini.
    """
    if not isinstance(exc, genai_errors.ClientError):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status == 429


class ParameterExtractor:
    """Extrai entidades médicas via Gemini com Schema Enforcement nativo.

    Attributes:
        _client: Instância do cliente genai autenticado.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Gemini com a chave de API do ambiente.

        Raises:
            ValueError: Se GEMINI_API_KEY não estiver definida.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada.")
        self._client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(genai_errors.ClientError),
        before_sleep=lambda rs: logger.warning(
            "[V1] Falha na API Gemini — backoff exponencial " "(tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Processa o texto e retorna um objeto Pydantic validado.

        Itera pelos modelos em _MODELS, tentando o próximo em caso de
        falha não-fatal. Erros 429 são capturados pelo Tenacity para
        backoff exponencial. Erros 401/403 são relançados imediatamente.

        Args:
            transcription_text: Texto transcrito pelo STT para análise.

        Returns:
            MedicalParameterExtraction validado ou None se resposta vazia.

        Raises:
            genai_errors.ClientError: Propagada para o Tenacity em caso de 429.
            ValueError: Se a autenticação falhar (401/403).
        """
        logger.info("[V1] Analisando texto via LLM: '%s'", transcription_text)
        last_exc: Exception | None = None

        for model_id in _MODELS:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_id,
                    contents=transcription_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=MedicalParameterExtraction,
                        temperature=0.0,
                    ),
                )

                if response.text:
                    return MedicalParameterExtraction.model_validate_json(response.text)
                return None

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

                if status in (401, 403):
                    raise ValueError(
                        f"Falha de autenticação com a API Gemini (HTTP {status})."
                    ) from exc

                if status == 429:
                    # Relança para o Tenacity capturar e aplicar backoff
                    raise

                last_exc = exc
                logger.warning(
                    "[V1] Modelo %s falhou (HTTP %s). Tentando próximo...",
                    model_id,
                    status,
                )

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "[V1] Modelo %s falhou com erro inesperado: %s",
                    model_id,
                    exc,
                )

        logger.error("[V1] Todos os modelos falharam. Último erro: %s", last_exc)
        return None
