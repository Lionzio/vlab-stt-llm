# src/services/extractor.py
"""Serviço de Extração de Parâmetros Médicos usando LLM e Schema Enforcement."""

import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.schemas.extraction import MedicalParameterExtraction

load_dotenv()
logger = logging.getLogger(__name__)

# System Prompt rigoroso com Few-Shot Learning focado no domínio clínico
SYSTEM_INSTRUCTION = """Você é um especialista em equipamentos médicos de suporte à vida e NLP.
Sua missão é extrair parâmetros, valores e intenções de transcrições de comandos de voz (STT).

REGRAS ESTritas:
1. MAPEAMENTO DE PARÂMETROS: Padronize os nomes (ex: "f i o dois" -> "fio2", "p a" -> "pressao_arterial").
2. INFERÊNCIA DE UNIDADE: Se o usuário pedir para "colocar a peep em cinco" e não disser a unidade, você DEVE inferir que é "cmH2O" com base no domínio clínico, e definir o status como "OK_INFERRED_UNIT".
3. LIMITES E SEGURANÇA: Se o valor for clinicamente absurdo para o parâmetro (ex: FiO2 maior que 100), o status DEVE ser "OUT_OF_BOUNDS".
4. AMBIGUIDADES: Se for "doze por oito", o value é null e o status "REQUIRES_CLARIFICATION", pois são dois valores (sistólica/diastólica).

EXEMPLOS (Few-Shot):
Input: "ajustar a frequência respiratória para quinze incursões por minuto"
Output: intent="ajustar_parametro", parameter="frequencia_respiratoria", value=15.0, unit="irpm", status="OK"

Input: "coloca a peep em cinco"
Output: intent="ajustar_parametro", parameter="peep", value=5.0, unit="cmH2O", status="OK_INFERRED_UNIT"
"""


class ParameterExtractor:
    """Extrai entidades médicas de textos utilizando o Gemini com response_schema."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada.")

        self._client = genai.Client(api_key=api_key)
        # Atualizado para o modelo estável mais recente
        self._model_id = "gemini-2.0-flash"

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"Falha na API Gemini (Tentativa {retry_state.attempt_number}/4). Aguardando para tentar de novo..."
        ),
    )
    async def extract(
        self, transcription_text: str
    ) -> MedicalParameterExtraction | None:
        """Processa o texto e retorna um objeto Pydantic validado.

        Lança exceções nativas do genai em caso de falha de API,
        permitindo que o decorador @retry faça o backoff exponencial.
        """
        logger.info(f"Analisando texto via LLM (V1): '{transcription_text}'")

        response = await self._client.aio.models.generate_content(
            model=self._model_id,
            contents=transcription_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                # Schema Enforcement Nativo
                response_schema=MedicalParameterExtraction,
                temperature=0.0,  # Zero alucinação
            ),
        )

        # O SDK garante o JSON aderente ao schema.
        # Injetamos no nosso objeto Pydantic manualmente para validações adicionais.
        if response.text:
            return MedicalParameterExtraction.model_validate_json(response.text)

        return None
