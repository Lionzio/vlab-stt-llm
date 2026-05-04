# src/services/heuristic_extractor.py
"""Extrator Heurístico (Offline) para Fallback Determinístico.

Utilizado quando a cota da API LLM é esgotada (HTTP 429). Implementa extração
baseada em regras (Regex e dicionários) para garantir a continuidade do serviço
(Graceful Degradation), retornando o mesmo contrato Pydantic.
"""

import re
import logging
from typing import Any

from src.schemas.extraction import MedicalParameterExtraction

logger = logging.getLogger(__name__)


class HeuristicParameterExtractor:
    """Extrai parâmetros médicos usando Regras Lexicais e Regex."""

    # Dicionário de conversão de números por extenso para float
    _NUMBERS = {
        "um": 1.0,
        "dois": 2.0,
        "tres": 3.0,
        "três": 3.0,
        "quatro": 4.0,
        "cinco": 5.0,
        "seis": 6.0,
        "sete": 7.0,
        "oito": 8.0,
        "nove": 9.0,
        "dez": 10.0,
        "onze": 11.0,
        "doze": 12.0,
        "quinze": 15.0,
        "vinte": 20.0,
        "cem": 100.0,
        "duzentos": 200.0,
        "seiscentos": 600.0,
    }

    # Dicionários de sinônimos clínicos
    _INTENTS = {
        r"\b(ajustar?|colocar?|mudar|configurar?)\b": "ajustar_parametro",
        r"\b(iniciar?|ligar)\b": "iniciar_terapia",
        r"\b(silenciar?|parar alarme)\b": "silenciar_alarme",
    }

    _PARAMETERS = {
        r"\b(frequ[eê]ncia respirat[oó]ria)\b": "frequencia_respiratoria",
        r"\b(peep|pipe)\b": "peep",
        r"\b(p\s?a|press[aã]o arterial)\b": "pressao_arterial",
        r"\b(f\s?i\s?o\s?dois|fio2)\b": "fio2",
        r"\b(modo de ventila[cç][aã]o|modo ventilat[oó]rio)\b": "modo_ventilatorio",
        r"\b(volume corrente)\b": "volume_corrente",
    }

    _UNITS = {
        r"\b(incurs[oõ]es por minuto|irpm)\b": "irpm",
        r"\b(cmh2o|cent[ií]metros de [aá]gua)\b": "cmH2O",
        r"\b(mmhg|mil[ií]metros de merc[uú]rio)\b": "mmHg",
        r"\b(por cento|%)\b": "%",
        r"\b(mililitros?|ml)\b": "ml",
    }

    def _extract_number(self, text: str) -> tuple[float | None, str | None]:
        """Tenta extrair um valor numérico (explícito ou por extenso)."""
        # 1. Checa frações/ambiguidades clínicas (ex: "doze por oito", "12/8")
        fraction_match = re.search(r"(\w+|\d+)\s+(por|/)\s+(\w+|\d+)", text)
        if fraction_match:
            return None, "REQUIRES_CLARIFICATION"

        # 2. Busca por números em texto
        for word, value in self._NUMBERS.items():
            if re.search(rf"\b{word}\b", text):
                return value, None

        # 3. Busca por dígitos
        digit_match = re.search(r"\b(\d+(\.\d+)?)\b", text)
        if digit_match:
            return float(digit_match.group(1)), None

        return None, "MISSING_VALUE"

    def extract(self, transcription_text: str) -> MedicalParameterExtraction:
        """Processa a string usando regras offline."""
        logger.info("[Heurística] Acionando extração determinística de fallback.")
        text = transcription_text.lower()

        # Extração de Intenção
        intent = "desconhecida"
        for pattern, canon_intent in self._INTENTS.items():
            if re.search(pattern, text):
                intent = canon_intent
                break

        # Extração de Parâmetro
        parameter = None
        for pattern, canon_param in self._PARAMETERS.items():
            if re.search(pattern, text):
                parameter = canon_param
                break

        # Extração de Unidade
        unit = None
        for pattern, canon_unit in self._UNITS.items():
            if re.search(pattern, text):
                unit = canon_unit
                break

        # Extração de Valor Numérico
        value, validation_status = self._extract_number(text)

        # Montagem do Status
        status = "OK"
        notes = "Extração realizada por fallback heurístico offline."

        if validation_status == "REQUIRES_CLARIFICATION":
            status = "REQUIRES_CLARIFICATION"
            notes = "Valores fracionados exigem value_systolic e value_diastolic."
        elif validation_status == "MISSING_VALUE" and intent == "ajustar_parametro":
            status = "MISSING_VALUE"
            notes = "Valor numérico não encontrado na transcrição."

        # Se não for ajustar_parametro (ex: iniciar_terapia), falta de valor é normal
        if intent != "ajustar_parametro" and value is None:
            status = "MISSING_VALUE"

        extraction_data = {
            "intent": intent,
            "parameter": parameter,
            "value": value,
            "unit": unit,
            "status": status,
            "notes": notes,
        }

        # Magia acontece aqui: Devolvemos para o Pydantic aplicar as Hard Rules (limites/unidades default)
        return MedicalParameterExtraction(**extraction_data)
