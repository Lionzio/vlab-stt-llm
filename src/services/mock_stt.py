# src/services/mock_stt.py
"""STT Mock para Bypass Emergencial (Fallback de Cota)."""

import os
import logging

logger = logging.getLogger(__name__)


class MockSTT:
    """Retorna a transcrição baseada no nome do arquivo (Fallback para testes)."""

    _GROUND_TRUTH_MAP = {
        "tc-001": "ajustar a frequência respiratória para quinze incursões por minuto",
        "tc-002": "coloca a peep em cinco",
        "tc-003": "mudar a pa para doze por oito",
        "tc-004": "configurar fio2 para duzentos por cento",
        "tc-005": "inicia o modo de ventilação",
        "tc-006": "ajusta o volume corrente pra seiscentos [ruído] mililitros",
    }

    async def transcribe(self, audio_filepath: str) -> str:
        logger.warning(
            "[STT Mock] Acionando transcrição simulada por falta de cota da API."
        )

        # Extrai nome base do arquivo (ex: "TC-001.mp3" -> "tc-001")
        basename = os.path.basename(audio_filepath).lower()
        filename_without_ext = os.path.splitext(basename)[0]

        # Faz fuzzy matching simples no nome do arquivo
        for key, transcript in self._GROUND_TRUTH_MAP.items():
            if key in filename_without_ext:
                return transcript

        return "transcrição simulada por contingência de cota"
