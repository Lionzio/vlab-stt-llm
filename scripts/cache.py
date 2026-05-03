# scripts/cache.py
"""Camada de cache em disco para resultados STT e LLM do pipeline de avaliação.

Persiste resultados em arquivos JSON indexados por hash SHA-256 do input,
permitindo reruns do avaliador sem consumir cota da API Gemini.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_STT_CACHE_FILE = _CACHE_DIR / "cache_stt.json"
_LLM_CACHE_FILE = _CACHE_DIR / "cache_llm.json"


def _load(path: Path) -> dict[str, Any]:
    """Carrega o arquivo de cache do disco, retornando dict vazio se ausente.

    Args:
        path: Caminho para o arquivo JSON de cache.

    Returns:
        Dicionário com entradas cacheadas ou dict vazio.
    """
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache corrompido em %s, reiniciando: %s", path, exc)
    return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    """Persiste o dicionário de cache no disco de forma atômica.

    Args:
        path: Caminho para o arquivo JSON de cache.
        data: Dicionário completo a ser serializado.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _sha256(value: str) -> str:
    """Calcula o hash SHA-256 de uma string para uso como chave de cache.

    Args:
        value: String a ser hasheada.

    Returns:
        Hex digest SHA-256 de 64 caracteres.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# API pública — STT cache
# ---------------------------------------------------------------------------


def get_stt(audio_path: str) -> str | None:
    """Recupera a transcrição cacheada para um arquivo de áudio.

    A chave de cache é o hash SHA-256 do caminho absoluto do arquivo.
    Arquivos diferentes com mesmo nome em pastas distintas são diferenciados.

    Args:
        audio_path: Caminho do arquivo de áudio.

    Returns:
        Transcrição cacheada ou None se ausente.
    """
    key = _sha256(str(Path(audio_path).resolve()))
    result = _load(_STT_CACHE_FILE).get(key)
    if result is not None:
        logger.info("Cache STT HIT para %s", Path(audio_path).name)
    return result


def set_stt(audio_path: str, transcript: str) -> None:
    """Armazena a transcrição de um arquivo de áudio no cache.

    Args:
        audio_path: Caminho do arquivo de áudio.
        transcript: Texto transcrito pelo STT.
    """
    key = _sha256(str(Path(audio_path).resolve()))
    data = _load(_STT_CACHE_FILE)
    data[key] = transcript
    _save(_STT_CACHE_FILE, data)
    logger.debug("Cache STT SET para %s", Path(audio_path).name)


# ---------------------------------------------------------------------------
# API pública — LLM cache
# ---------------------------------------------------------------------------


def get_llm(transcript: str, extractor_version: str = "v1") -> dict[str, Any] | None:
    """Recupera o resultado de extração LLM cacheado para uma transcrição.

    Args:
        transcript: Texto transcrito que foi enviado ao LLM.
        extractor_version: Identificador da versão do extrator (ex: "v1", "v2").
            Versões diferentes com o mesmo texto produzem resultados distintos
            e são armazenadas separadamente.

    Returns:
        Dicionário com o resultado de extração cacheado ou None se ausente.
    """
    key = _sha256(f"{extractor_version}::{transcript}")
    result = _load(_LLM_CACHE_FILE).get(key)
    if result is not None:
        logger.info("Cache LLM HIT [%s] para %.40s...", extractor_version, transcript)
    return result


def set_llm(
    transcript: str, result: dict[str, Any], extractor_version: str = "v1"
) -> None:
    """Armazena o resultado de extração LLM no cache.

    Args:
        transcript: Texto transcrito que foi enviado ao LLM.
        result: Dicionário serializável com o resultado da extração.
        extractor_version: Identificador da versão do extrator.
    """
    key = _sha256(f"{extractor_version}::{transcript}")
    data = _load(_LLM_CACHE_FILE)
    data[key] = result
    _save(_LLM_CACHE_FILE, data)
    logger.debug("Cache LLM SET [%s] para %.40s...", extractor_version, transcript)


def clear() -> None:
    """Remove todos os arquivos de cache do disco.

    Útil para forçar reavaliação completa do pipeline.
    """
    for f in [_STT_CACHE_FILE, _LLM_CACHE_FILE]:
        if f.exists():
            f.unlink()
            logger.info("Cache removido: %s", f)
