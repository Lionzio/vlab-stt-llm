# src/services/extractor_v2.py
"""Extrator de parâmetros médicos V2 — Chain-of-Thought (CoT).

Força o modelo a externalizar o raciocínio antes de preencher o JSON,
reduzindo alucinações em cenários ambíguos ao custo de maior latência.
Inclui Exponential Backoff via Tenacity para erros 429.
"""

from __future__ import annotations

import logging
import os
import re

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

# Modelos estáveis em ordem de preferência
_MODELS: list[str] = ["gemini-2.0-flash", "gemini-1.5-flash"]


class ParameterExtractorV2:
    """Extrai parâmetros médicos usando Chain-of-Thought antes do JSON final.

    Diferença vs V1:
        V1 usa `response_mime_type="application/json"` com schema enforcement
        nativo — resposta direta, menor latência.
        V2 usa `response_mime_type="text/plain"` com bloco <reasoning> + parse
        manual do bloco <json> — maior robustez em ambiguidades.

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
            raise ValueError(
                "GEMINI_API_KEY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )
        self._client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(genai_errors.ClientError),
        before_sleep=lambda rs: logger.warning(
            "[V2-CoT] Falha na API Gemini — backoff exponencial " "(tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Extrai parâmetros via Chain-of-Thought com fallback de modelos.

        O modelo produz um bloco <reasoning> com análise passo a passo e um
        bloco <json> com o resultado. Apenas o bloco JSON é parseado e
        validado via Pydantic.

        Args:
            transcription_text: Texto transcrito pelo STT para análise.

        Returns:
            MedicalParameterExtraction validado ou None se o bloco <json>
            não for encontrado na resposta.

        Raises:
            genai_errors.ClientError: Propagada para o Tenacity em caso de 429.
            ValueError: Se a autenticação falhar (401/403).
        """
        logger.info("[V2-CoT] Analisando: '%s'", transcription_text)
        last_exc: Exception | None = None

        for model_id in _MODELS:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_id,
                    contents=transcription_text,
                    config=types.GenerateContentConfig(
                        system_instruction=_COT_SYSTEM_INSTRUCTION,
                        response_mime_type="text/plain",
                        temperature=0.1,
                    ),
                )

                raw_text = response.text or ""
                logger.debug("[V2-CoT] Resposta bruta:\n%s", raw_text)

                match = _JSON_BLOCK_RE.search(raw_text)
                if not match:
                    logger.error(
                        "[V2-CoT] Bloco <json> não encontrado " "(modelo %s): %.200s",
                        model_id,
                        raw_text,
                    )
                    return None

                return MedicalParameterExtraction.model_validate_json(
                    match.group(1).strip()
                )

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

                if status in (401, 403):
                    raise ValueError(
                        f"Falha de autenticação com a API Gemini " f"(HTTP {status})."
                    ) from exc

                if status == 429:
                    raise

                last_exc = exc
                logger.warning(
                    "[V2-CoT] Modelo %s falhou (HTTP %s). " "Tentando próximo...",
                    model_id,
                    status,
                )

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "[V2-CoT] Modelo %s falhou com erro inesperado: %s",
                    model_id,
                    exc,
                )

        logger.error("[V2-CoT] Todos os modelos falharam. Último erro: %s", last_exc)
        return None

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(genai_errors.ClientError),
        before_sleep=lambda rs: logger.warning(
            "[V2-CoT/reasoning] Backoff exponencial (tentativa %d/4)...",
            rs.attempt_number,
        ),
    )
    async def extract_with_reasoning(
        self, transcription_text: str
    ) -> tuple[MedicalParameterExtraction | None, str]:
        """Extrai parâmetros e retorna também o raciocínio CoT do modelo.

        Útil para análise qualitativa e debugging em cenários de borda.

        Args:
            transcription_text: Texto transcrito para análise.

        Returns:
            Tupla (extração_validada, raciocínio_bruto). Retorna (None, "")
            em caso de falha no parse do bloco <json>.

        Raises:
            genai_errors.ClientError: Propagada para Tenacity em caso de 429.
        """
        last_exc: Exception | None = None

        for model_id in _MODELS:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model_id,
                    contents=transcription_text,
                    config=types.GenerateContentConfig(
                        system_instruction=_COT_SYSTEM_INSTRUCTION,
                        response_mime_type="text/plain",
                        temperature=0.1,
                    ),
                )

                raw_text = response.text or ""

                reasoning_match = re.search(
                    r"<reasoning>\s*(.*?)\s*</reasoning>",
                    raw_text,
                    re.DOTALL,
                )
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

                json_match = _JSON_BLOCK_RE.search(raw_text)
                if not json_match:
                    return None, reasoning

                extraction = MedicalParameterExtraction.model_validate_json(
                    json_match.group(1).strip()
                )
                return extraction, reasoning

            except genai_errors.ClientError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 429:
                    raise
                last_exc = exc
                logger.warning(
                    "[V2-CoT/reasoning] Modelo %s falhou (HTTP %s).",
                    model_id,
                    status,
                )

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "[V2-CoT/reasoning] Modelo %s erro inesperado: %s",
                    model_id,
                    exc,
                )

        logger.error("[V2-CoT/reasoning] Todos os modelos falharam: %s", last_exc)
        return None, ""
