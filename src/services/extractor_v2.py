# src/services/extractor_v2.py
"""Extrator de parâmetros médicos V2 — Abordagem Chain-of-Thought (CoT).

Agora consome o GeminiManager para gerenciar cotas, clientes e fallback.
Ao contrário do V1, o V2 exige resposta em texto livre para que o LLM
possa externalizar seu raciocínio antes de preencher o JSON.
Inclui injeção de dependência e propagação direta de exceções de cota.
"""

from __future__ import annotations

import logging
import re

from google.genai import types
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

_COT_SYSTEM_INSTRUCTION = """\
Você é um especialista em NLP clínico e equipamentos de suporte à vida.
Ao receber uma transcrição de comando de voz médico, você DEVE seguir estas
etapas em ordem antes de produzir o JSON final:

ETAPA 1 — IDENTIFICAÇÃO DE INTENÇÃO:
Analise o verbo principal da frase. Mapeie para uma das intenções canônicas:
ajustar_parametro | iniciar_terapia | silenciar_alarme | consultar_status
| desconhecida. Justifique sua escolha.

ETAPA 2 — MAPEAMENTO DE PARÂMETRO:
Identifique o parâmetro clínico mencionado. Normalize siglas e variantes:
- "f i o dois", "fi dois", "FiO2" → "fio2"
- "p a", "PA", "pressão arterial" → "pressao_arterial"
- "pipe", "peep", "PEEP" → "peep"
Se ausente, justifique por que é null.

ETAPA 3 — EXTRAÇÃO DE VALOR:
Identifique o valor numérico. Converta por extenso: "quinze" → 15.0.
Se fracionado (ex: "doze por oito"), o value é null — justifique.

ETAPA 4 — INFERÊNCIA DE UNIDADE:
Se a unidade não foi explicitada, infira pela tabela de domínio:
peep/pi → cmH2O | fio2/spo2 → % | frequencia_respiratoria → irpm
frequencia_cardiaca → bpm | pressao_arterial → mmHg | volume_corrente → ml
Se inferida, o status deve ser OK_INFERRED_UNIT.

ETAPA 5 — VALIDAÇÃO DE LIMITES CLÍNICOS:
fio2: [21, 100] | peep: [0, 25] | frequencia_respiratoria: [4, 60]
volume_corrente: [200, 800] | frequencia_cardiaca: [20, 300]
Se fora dos limites, status = OUT_OF_BOUNDS.

ETAPA 6 — DETERMINAÇÃO DO STATUS FINAL:
OK | OK_INFERRED_UNIT | MISSING_VALUE | OUT_OF_BOUNDS
| REQUIRES_CLARIFICATION | ERROR

Formate sua resposta EXATAMENTE assim (não omita nenhuma seção):

<reasoning>
[Seu raciocínio passo a passo aqui]
</reasoning>
<json>
{"intent": "...", "parameter": "...", "value": ..., "unit": "...",
 "status": "...", "notes": "..."}
</json>
"""

_JSON_BLOCK_RE = re.compile(r"<json>\s*(\{.*?})\s*</json>", re.DOTALL)


class ParameterExtractorV2:
    """Extrai parâmetros via Chain-of-Thought usando GeminiManager.

    Attributes:
        _manager: Instância do GeminiManager para orquestração da API.
    """

    def __init__(self, manager: GeminiManager | None = None) -> None:
        """Inicializa o extrator V2 com um manager opcional."""
        try:
            self._manager = manager or GeminiManager()
        except GeminiAuthError as exc:
            raise ValueError(str(exc)) from exc

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(GeminiQuotaError),
        reraise=True,  # Propaga o erro de cota para o FastAPI tratar como 429
        before_sleep=lambda rs: logger.warning(
            "[V2-CoT] Ambas as chaves esgotadas. Backoff (tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Extrai parâmetros via Chain-of-Thought e faz parse do resultado."""
        logger.info("[V2-CoT] Analisando: '%s'", transcription_text)

        config = types.GenerateContentConfig(
            system_instruction=_COT_SYSTEM_INSTRUCTION,
            response_mime_type="text/plain",
            temperature=0.1,
        )

        try:
            # Roteamento transparente via Manager
            raw_text = await self._manager._execute_with_quota_fallback(
                model=self._manager.pro_model,
                contents=transcription_text,
                config=config,
            )

            logger.debug("[V2-CoT] Resposta bruta:\n%s", raw_text)

            match = _JSON_BLOCK_RE.search(raw_text)
            if not match:
                logger.error("[V2-CoT] Bloco <json> não encontrado na resposta.")
                return None

            json_str = match.group(1).strip()
            return MedicalParameterExtraction.model_validate_json(json_str)

        except GeminiAuthError as exc:
            raise ValueError(f"Falha de autenticação: {exc}") from exc

        except GeminiQuotaError:
            raise

        except Exception as exc:  # noqa: BLE001
            logger.error("[V2-CoT] Falha na extração após fallbacks: %s", exc)
            return None

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(GeminiQuotaError),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "[V2-CoT/Reasoning] Ambas as chaves esgotadas. Backoff (tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract_with_reasoning(
        self, transcription_text: str
    ) -> tuple[MedicalParameterExtraction | None, str]:
        """Extrai parâmetros e retorna também o raciocínio CoT do modelo."""
        config = types.GenerateContentConfig(
            system_instruction=_COT_SYSTEM_INSTRUCTION,
            response_mime_type="text/plain",
            temperature=0.1,
        )

        try:
            raw_text = await self._manager._execute_with_quota_fallback(
                model=self._manager.pro_model,
                contents=transcription_text,
                config=config,
            )

            reasoning_match = re.search(
                r"<reasoning>\s*(.*?)\s*</reasoning>", raw_text, re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            json_match = _JSON_BLOCK_RE.search(raw_text)
            if not json_match:
                return None, reasoning

            extraction = MedicalParameterExtraction.model_validate_json(
                json_match.group(1).strip()
            )
            return extraction, reasoning

        except GeminiAuthError as exc:
            raise ValueError(f"Falha de autenticação: {exc}") from exc

        except GeminiQuotaError:
            raise

        except Exception as exc:  # noqa: BLE001
            logger.error("[V2-CoT/Reasoning] Falha inesperada: %s", exc)
            return None, ""
