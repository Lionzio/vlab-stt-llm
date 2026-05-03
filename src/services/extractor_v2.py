# src/services/extractor_v2.py
"""Extrator de parâmetros médicos V2 — Abordagem Chain-of-Thought (CoT).

Diferença fundamental em relação ao V1:
    - V1 usa Zero-Shot direto com regras explícitas e exemplos few-shot.
    - V2 força o modelo a externalizar o raciocínio antes de preencher o JSON,
      utilizando a técnica Chain-of-Thought. O modelo explica cada decisão
      (mapeamento de parâmetro, inferência de unidade, validação de limites)
      antes de produzir a saída estruturada. Isso reduz alucinações em
      cenários ambíguos ao custo de maior latência e uso de tokens.
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas.extraction import MedicalParameterExtraction

load_dotenv()

logger = logging.getLogger(__name__)

# O prompt CoT instrui o modelo a raciocinar em etapas antes de gerar o JSON.
# A tag <reasoning> é descartada no parse — serve apenas como scratchpad.
_COT_SYSTEM_INSTRUCTION = """\
Você é um especialista em NLP clínico e equipamentos de suporte à vida.
Ao receber uma transcrição de comando de voz médico, você DEVE seguir estas etapas
em ordem antes de produzir o JSON final:

ETAPA 1 — IDENTIFICAÇÃO DE INTENÇÃO:
Analise o verbo principal da frase. Mapeie para uma das intenções canônicas:
ajustar_parametro | iniciar_terapia | silenciar_alarme | consultar_status | desconhecida.
Justifique sua escolha.

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
Verifique se o valor está dentro dos limites seguros:
fio2: [21, 100] | peep: [0, 25] | frequencia_respiratoria: [4, 60]
volume_corrente: [200, 800] | frequencia_cardiaca: [20, 300]
Se fora dos limites, status = OUT_OF_BOUNDS.

ETAPA 6 — DETERMINAÇÃO DO STATUS FINAL:
OK | OK_INFERRED_UNIT | MISSING_VALUE | OUT_OF_BOUNDS | REQUIRES_CLARIFICATION | ERROR

Formate sua resposta EXATAMENTE assim (não omita nenhuma seção):

<reasoning>
[Seu raciocínio passo a passo aqui]
</reasoning>
<json>
{"intent": "...", "parameter": "...", "value": ..., "unit": "...", "status": "...", "notes": "..."}
</json>
"""

_JSON_BLOCK_RE = re.compile(r"<json>\s*(\{.*?})\s*</json>", re.DOTALL)


class ParameterExtractorV2:
    """Extrai parâmetros médicos usando Chain-of-Thought antes do JSON final.

    A abordagem CoT força o modelo a externalizar o raciocínio em cada etapa
    (identificação de intenção, mapeamento, validação de limites) antes de
    preencher o schema. Isso melhora a precisão em cenários ambíguos como
    siglas homofônicas e unidades implícitas.

    Diferença vs V1:
        - V1: `response_mime_type="application/json"` com schema enforcement nativo.
              Resposta direta, menor latência, pode errar em casos ambíguos.
        - V2: `response_mime_type="text/plain"` com CoT estruturado + parse manual
              do bloco `<json>`. Maior latência/tokens, mais robusto em ambiguidades.

    Attributes:
        _client: Instância do cliente genai autenticado.
        _model_id: Identificador do modelo Gemini utilizado.
    """

    def __init__(self) -> None:
        """Inicializa o cliente Gemini com a chave de API do ambiente.

        Raises:
            ValueError: Se GEMINI_API_KEY não estiver definida ou for inválida.
        """
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_id = "gemini-2.5-flash"

    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Extrai parâmetros médicos via raciocínio Chain-of-Thought.

        O modelo produz um bloco <reasoning> com análise passo a passo seguido
        de um bloco <json> com o resultado estruturado. Apenas o bloco JSON é
        parseado e validado via Pydantic.

        Args:
            transcription_text: Texto transcrito pelo STT para análise.

        Returns:
            MedicalParameterExtraction validado pelo Pydantic, ou None se o
            parse do bloco JSON falhar.
        """
        logger.info("[V2-CoT] Analisando: '%s'", transcription_text)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_id,
                contents=transcription_text,
                config=types.GenerateContentConfig(
                    system_instruction=_COT_SYSTEM_INSTRUCTION,
                    # Texto livre para permitir o bloco <reasoning> antes do JSON
                    response_mime_type="text/plain",
                    temperature=0.1,  # Levemente acima de 0 para permitir raciocínio
                ),
            )

            raw_text = response.text or ""
            logger.debug("[V2-CoT] Resposta bruta:\n%s", raw_text)

            match = _JSON_BLOCK_RE.search(raw_text)
            if not match:
                logger.error(
                    "[V2-CoT] Bloco <json> não encontrado na resposta: %.200s",
                    raw_text,
                )
                return None

            json_str = match.group(1).strip()
            return MedicalParameterExtraction.model_validate_json(json_str)

        except Exception as exc:  # noqa: BLE001
            logger.error("[V2-CoT] Falha na extração: %s", exc)
            return None

    async def extract_with_reasoning(
        self, transcription_text: str
    ) -> tuple[MedicalParameterExtraction | None, str]:
        """Extrai parâmetros e retorna também o raciocínio CoT do modelo.

        Útil para análise qualitativa e debugging do comportamento do modelo
        em cenários de borda.

        Args:
            transcription_text: Texto transcrito para análise.

        Returns:
            Tupla (extração_validada, raciocínio_bruto). O raciocínio é a
            string completa dentro das tags <reasoning>. Retorna (None, "")
            em caso de falha.
        """
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_id,
                contents=transcription_text,
                config=types.GenerateContentConfig(
                    system_instruction=_COT_SYSTEM_INSTRUCTION,
                    response_mime_type="text/plain",
                    temperature=0.1,
                ),
            )

            raw_text = response.text or ""

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

        except Exception as exc:  # noqa: BLE001
            logger.error("[V2-CoT] Falha em extract_with_reasoning: %s", exc)
            return None, ""
