# tests/test_pipeline.py
"""Suíte de testes unitários para validação determinística do Schema (Inteligência Híbrida).

Garante que as regras rígidas (hard rules) do Pydantic atuam como uma rede
de segurança clínica contra alucinações matemáticas ou omissões do LLM.
Testes executados 100% offline.
"""

from src.schemas.extraction import MedicalParameterExtraction


def test_caso_1_valido_simples():
    """TC-001: Caso ideal onde o LLM extrai todos os dados corretamente."""
    mock_llm_output = {
        "intent": "ajustar_parametro",
        "parameter": "frequencia_respiratoria",
        "value": 15.0,
        "unit": "irpm",
        "status": "OK",
        "notes": None,
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.intent == "ajustar_parametro"
    assert extraction.parameter == "frequencia_respiratoria"
    assert extraction.value == 15.0
    assert extraction.unit == "irpm"
    assert extraction.status == "OK"


def test_caso_2_unidade_omitida_inferencia_por_regra():
    """TC-002: O LLM esqueceu a unidade. O Pydantic DEVE injetar a unidade canônica."""
    mock_llm_output = {
        "intent": "ajustar_parametro",
        "parameter": "peep",
        "value": 5.0,
        "unit": None,  # Omissão do LLM
        "status": "OK",
        "notes": None,
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.parameter == "peep"
    assert extraction.value == 5.0
    assert extraction.unit == "cmH2O"  # Injetado pela Hard Rule
    assert extraction.status == "OK_INFERRED_UNIT_BY_RULE"
    assert "injetada por regra" in extraction.notes


def test_caso_3_ambiguo_requer_clarificacao():
    """TC-003: Valores fracionados (12 por 8) exigem clarificação."""
    mock_llm_output = {
        "intent": "ajustar_parametro",
        "parameter": "pressao_arterial",
        "value": None,  # Valor não extraível como float único
        "unit": "mmHg",
        "status": "REQUIRES_CLARIFICATION",
        "notes": "Valores fracionados exigem clarificação sistólica/diastólica",
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.value is None
    assert extraction.status == "REQUIRES_CLARIFICATION"


def test_caso_4_valor_fora_do_limite_clinico():
    """TC-004: O LLM aceitou ingenuamente um FiO2 de 200%. O Pydantic DEVE barrar."""
    mock_llm_output = {
        "intent": "ajustar_parametro",
        "parameter": "fio2",
        "value": 200.0,  # Alucinação/Absurdo clínico (máx 100)
        "unit": "%",
        "status": "OK",  # LLM não percebeu o erro
        "notes": None,
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.value == 200.0
    assert extraction.status == "OUT_OF_BOUNDS"  # Corrigido pela Hard Rule
    assert "fora dos limites seguros" in extraction.notes


def test_caso_5_incompleto_missing_value():
    """TC-005: Comando interrompido. Parâmetro reconhecido, mas sem valor."""
    mock_llm_output = {
        "intent": "iniciar_terapia",
        "parameter": "modo_ventilatorio",
        "value": None,
        "unit": None,
        "status": "MISSING_VALUE",
        "notes": "Valor numérico não fornecido no áudio",
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.intent == "iniciar_terapia"
    assert extraction.value is None
    assert extraction.status == "MISSING_VALUE"


def test_caso_6_ruido_simulado():
    """TC-006: O LLM deve ter ignorado o ruído no texto e extraído o valor corretamente."""
    # O LLM recebe: "ajusta o volume corrente pra seiscentos [ruído] mililitros"
    mock_llm_output = {
        "intent": "ajustar_parametro",
        "parameter": "volume_corrente",
        "value": 600.0,
        "unit": "ml",
        "status": "OK",
        "notes": "Ruído na fala ignorado durante a extração",
    }

    extraction = MedicalParameterExtraction(**mock_llm_output)

    assert extraction.parameter == "volume_corrente"
    assert extraction.value == 600.0
    assert extraction.unit == "ml"
    assert extraction.status == "OK"
