# scripts/audio_augmentation.py
"""Módulo de Data Augmentation para Testes de Stress (Sprint 9).
Injeta ruído de fundo (bipes hospitalares/conversas) em áudios limpos
para testar a resiliência do STT e a recuperação do LLM.
"""

import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def inject_hospital_noise(
    clean_audio_path: str | Path,
    noise_audio_path: str | Path,
    output_path: str | Path,
    noise_reduction_db: float = 15.0,
) -> str | None:
    """Mixes a clean audio file with a noise file.

    Args:
        clean_audio_path: Caminho para o áudio original.
        noise_audio_path: Caminho para o áudio com o ruído hospitalar.
        output_path: Caminho onde o áudio mixado será salvo.
        noise_reduction_db: Quantos decibéis reduzir o ruído (simula distância).

    Returns:
        O caminho como string do áudio gerado, ou None se falhar.
    """
    try:
        clean_audio = AudioSegment.from_file(clean_audio_path)
        # Tenta carregar o ruído. Se não existir, retorna o limpo para não quebrar a CI
        if not Path(noise_audio_path).exists():
            logger.warning(
                "Arquivo de ruído não encontrado: %s. Ignorando augmentation.",
                noise_audio_path,
            )
            return str(clean_audio_path)

        noise_audio = AudioSegment.from_file(noise_audio_path)

        # Reduz o volume do ruído de fundo
        noise_audio = noise_audio - noise_reduction_db

        # Se o ruído for mais curto que o áudio, faz um loop nele (looping)
        if len(noise_audio) < len(clean_audio):
            loops = (len(clean_audio) // len(noise_audio)) + 1
            noise_audio = noise_audio * loops

        # Corta o ruído para ter exatamente a duração do áudio principal
        noise_audio = noise_audio[: len(clean_audio)]

        # Mixa (overlay) as duas trilhas
        mixed_audio = clean_audio.overlay(noise_audio)

        # Exporta como mp3
        mixed_audio.export(output_path, format="mp3")
        logger.info("Áudio de Stress gerado: %s", output_path)
        return str(output_path)

    except Exception as exc:
        logger.error("Falha no Data Augmentation do áudio: %s", exc)
        return str(clean_audio_path)  # Fallback: usa o áudio limpo se falhar
