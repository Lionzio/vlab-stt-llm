# src/services/extractor.py
"""Serviço de Extração de Parâmetros Médicos — V1 (Direct Schema).

Agora consome o GeminiManager para gerenciar cotas, clientes e fallback
de modelos cognitivos, mantendo sua própria resiliência de backoff
exponencial caso todas as chaves do Manager se esgotem.
"""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.schemas.extraction import MedicalParameterExtraction
from src.services.gemini_manager import (
    GeminiAuthError,
    GeminiManager,
    GeminiQuotaError,
)

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


class ParameterExtractor:
    """Extrai entidades médicas usando GeminiManager com Schema Enforcement.

    Attributes:
        _manager: Instância do GeminiManager para orquestração da API.
    """

    def __init__(self, manager: GeminiManager | None = None) -> None:
        """Inicializa o extrator com um manager opcional.

        Args:
            manager: GeminiManager pré-configurado. Se None, cria um novo.

        Raises:
            ValueError: Se a chave primária não estiver disponível (via Manager).
        """
        try:
            self._manager = manager or GeminiManager()
        except GeminiAuthError as exc:
            raise ValueError(str(exc)) from exc

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(GeminiQuotaError),
        reraise=True,  # Garante que o erro 429 chegue ao caller original (API/Scripts)
        before_sleep=lambda rs: logger.warning(
            "[V1] Ambas as chaves esgotadas. Backoff exponencial (tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Processa o texto e retorna um objeto Pydantic validado.

        O roteamento (Flash -> Pro) e a rotação de chaves são resolvidos
        pelo GeminiManager de forma transparente. Este método apenas aplica
        backoff se a cota global do Manager falhar.

        Args:
            transcription_text: Texto transcrito pelo STT para análise.

        Returns:
            MedicalParameterExtraction validado ou None se erro irrecuperável.

        Raises:
            GeminiQuotaError: Propagada para o Tenacity ou para o caller (via reraise).
            ValueError: Se a autenticação falhar (401/403).
        """
        logger.info("[V1] Analisando texto via LLM: '%s'", transcription_text)

        try:
            # O Manager cuida da complexidade e nos devolve o Pydantic pronto!
            return await self._manager.execute_structured_with_fallback(
                contents=transcription_text,
                response_schema=MedicalParameterExtraction,
                system_instruction=SYSTEM_INSTRUCTION,
                is_complex_task=False,  # V1 foca em velocidade via Flash
            )

        except GeminiAuthError as exc:
            # Erro fatal de autenticação
            raise ValueError(f"Falha de autenticação: {exc}") from exc

        except GeminiQuotaError:
            # Relança para o Tenacity capturar e aplicar backoff (ou reraise final)
            raise

        except Exception as exc:
            logger.error("[V1] Falha na extração após fallbacks do Manager: %s", exc)
            return None
