"""Script automatizado para geração do Mock Dataset (Áudios sintéticos e Ground Truth)."""

import json
import logging
from pathlib import Path

from gtts import gTTS

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Mapeamento de Diretórios Base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio_samples"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"

# Definição Estrita dos Casos de Teste (Mock Data)
TEST_CASES = [
    {
        "id": "TC-001",
        "scenario_type": "ideal",
        "tts_text": "Ajustar a frequência respiratória para quinze incursões por minuto.",
        "expected_transcription": "ajustar a frequência respiratória para quinze incursões por minuto",
        "expected_extraction": {
            "intent": "ajustar_parametro",
            "parameter": "frequencia_respiratoria",
            "value": 15.0,
            "unit": "irpm",
            "status": "OK",
        },
    },
    {
        "id": "TC-002",
        "scenario_type": "unidade_omitida",
        "tts_text": "Coloca a PEEP em cinco.",
        "expected_transcription": "coloca a peep em cinco",
        "expected_extraction": {
            "intent": "ajustar_parametro",
            "parameter": "peep",
            "value": 5.0,
            "unit": "cmH2O",
            "status": "OK_INFERRED_UNIT",
        },
    },
    {
        "id": "TC-003",
        "scenario_type": "ambiguidade_terminologica",
        "tts_text": "Mudar a P A para doze por oito.",
        "expected_transcription": "mudar a pa para doze por oito",
        "expected_extraction": {
            "intent": "ajustar_parametro",
            "parameter": "pressao_arterial",
            "value": None,
            "unit": "mmHg",
            "status": "REQUIRES_CLARIFICATION",
            "notes": "Valores fracionados exigem value_systolic e value_diastolic",
        },
    },
    {
        "id": "TC-004",
        "scenario_type": "fora_do_padrao_limites",
        "tts_text": "Configurar F i O dois para duzentos por cento.",
        "expected_transcription": "configurar fio2 para duzentos por cento",
        "expected_extraction": {
            "intent": "ajustar_parametro",
            "parameter": "fio2",
            "value": 200.0,
            "unit": "%",
            "status": "OUT_OF_BOUNDS",
        },
    },
    {
        "id": "TC-005",
        "scenario_type": "comando_incompleto",
        "tts_text": "Inicia o modo de ventilação",
        "expected_transcription": "inicia o modo de ventilação",
        "expected_extraction": {
            "intent": "iniciar_terapia",
            "parameter": "modo_ventilatorio",
            "value": None,
            "unit": None,
            "status": "MISSING_VALUE",
        },
    },
    {
        "id": "TC-006",
        "scenario_type": "ruido_simulado",
        "tts_text": "Ajusta o volume corrente pra seiscentos coffe coffe mililitros.",
        "expected_transcription": "ajusta o volume corrente pra seiscentos [ruído] mililitros",
        "expected_extraction": {
            "intent": "ajustar_parametro",
            "parameter": "volume_corrente",
            "value": 600.0,
            "unit": "ml",
            "status": "OK",
        },
    },
]


def generate_dataset() -> None:
    """Orquestra a geração de áudios e a compilação do arquivo JSON de gabarito."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth_data = []

    for case in TEST_CASES:
        audio_filename = f"{case['id']}.mp3"
        audio_path = AUDIO_DIR / audio_filename

        logging.info(f"Sintetizando áudio para o cenário: {case['id']}...")

        # Geração do Áudio via gTTS
        tts = gTTS(text=case["tts_text"], lang="pt", slow=False)
        tts.save(str(audio_path))

        # Montagem do Gabarito (descartando tts_text)
        gt_entry = {
            "id": case["id"],
            "audio_filename": audio_filename,
            "scenario_type": case["scenario_type"],
            "expected_transcription": case["expected_transcription"],
            "expected_extraction": case["expected_extraction"],
        }
        ground_truth_data.append(gt_entry)

    # Persistência do JSON
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth_data, f, ensure_ascii=False, indent=2)

    logging.info(f"Sucesso! Dataset e Gabarito salvos em: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
