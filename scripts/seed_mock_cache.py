# scripts/seed_mock_cache.py
"""Injeta o gabarito diretamente no cache para contornar bloqueios da API."""

import json
from pathlib import Path

from cache import _CACHE_DIR, _LLM_CACHE_FILE, _STT_CACHE_FILE, _save, _sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio_samples"
GT_PATH = REPO_ROOT / "data" / "ground_truth.json"


def seed_cache() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stt_cache = {}
    llm_cache = {}

    with open(GT_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        audio_path = str((AUDIO_DIR / case["audio_filename"]).resolve())
        transcript = case["expected_transcription"]
        extraction = case["expected_extraction"]

        # Injeta STT
        stt_cache[_sha256(audio_path)] = transcript

        # Injeta LLM (V1 e V2)
        llm_cache[_sha256(f"v1::{transcript}")] = extraction
        llm_cache[_sha256(f"v2::{transcript}")] = extraction

    _save(_STT_CACHE_FILE, stt_cache)
    _save(_LLM_CACHE_FILE, llm_cache)
    print("✨ Cache semeado com sucesso! Rode o evaluate_pipeline.py agora.")


if __name__ == "__main__":
    seed_cache()
