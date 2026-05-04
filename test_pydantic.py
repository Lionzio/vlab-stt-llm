# test_pydantic.py
import json
from src.schemas.extraction import MedicalParameterExtraction

def run_tests():
    print("🧪 Iniciando Testes da Sprint 2: Inteligência Híbrida (Pydantic)\n")

    # ---------------------------------------------------------
    # Cenário 1: Simulando TC-002 (Unidade Omitida)
    # O LLM entendeu o parâmetro e o valor, mas esqueceu a unidade.
    # ---------------------------------------------------------
    mock_llm_tc002 = {
        "intent": "ajustar_parametro",
        "parameter": "peep",
        "value": 5.0,
        "unit": None,  # O LLM enviou nulo!
        "status": "OK",
        "notes": None
    }

    print("--- CENÁRIO 1: PEEP sem unidade ---")
    print(f"Entrada (LLM): {json.dumps(mock_llm_tc002, indent=2)}")
    
    result_1 = MedicalParameterExtraction(**mock_llm_tc002)
    
    print("\n✅ Saída Corrigida (Pydantic):")
    print(f" -> Unidade Final : '{result_1.unit}'")
    print(f" -> Status Final  : '{result_1.status}'")
    print(f" -> Rastro/Notas  : '{result_1.notes}'\n")
    print("-" * 50 + "\n")


    # ---------------------------------------------------------
    # Cenário 2: Simulando TC-004 (Valor Absurdo)
    # O LLM aceitou ingenuamente um FiO2 de 200%.
    # ---------------------------------------------------------
    mock_llm_tc004 = {
        "intent": "ajustar_parametro",
        "parameter": "fio2",
        "value": 200.0, # Valor absurdo! (Máx clínico é 100%)
        "unit": "%",
        "status": "OK", # O LLM achou que estava tudo bem...
        "notes": None
    }

    print("--- CENÁRIO 2: FiO2 Fora dos Limites Clínicos (200%) ---")
    print(f"Entrada (LLM): {json.dumps(mock_llm_tc004, indent=2)}")
    
    result_2 = MedicalParameterExtraction(**mock_llm_tc004)
    
    print("\n✅ Saída Corrigida (Pydantic):")
    print(f" -> Valor Final  : {result_2.value}")
    print(f" -> Status Final : '{result_2.status}'")
    print(f" -> Rastro/Notas : '{result_2.notes}'\n")

if __name__ == "__main__":
    run_tests()