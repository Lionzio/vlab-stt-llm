"""Schemas para extração estruturada de parâmetros médicos via LLM."""

from typing import Literal

from pydantic import BaseModel, Field


class MedicalParameterExtraction(BaseModel):
    """Schema estrito para a saída da extração de parâmetros via voz.
    O LLM será forçado a preencher os campos com base neste contrato.
    """

    intent: Literal[
        "ajustar_parametro",
        "iniciar_terapia",
        "silenciar_alarme",
        "consultar_status",
        "desconhecida",
    ] = Field(
        ...,
        description="A intenção principal do comando do usuário.",
    )

    parameter: str | None = Field(
        None,
        description="O parâmetro médico alvo. Ex: 'peep', 'fio2', 'frequencia_respiratoria'. Se não houver, retorne nulo.",
    )

    value: float | None = Field(
        None,
        description="O valor numérico a ser ajustado. Se for fracionado ou não for numérico, retorne nulo e preencha 'notes'.",
    )

    unit: str | None = Field(
        None,
        description="A unidade de medida inferida ou declarada. Ex: 'cmH2O', '%', 'mmHg', 'irpm'. Se não couber unidade, retorne nulo.",
    )

    status: Literal[
        "OK",
        "OK_INFERRED_UNIT",
        "MISSING_VALUE",
        "OUT_OF_BOUNDS",
        "REQUIRES_CLARIFICATION",
        "ERROR",
    ] = Field(
        ...,
        description="O status da extração. Use OK_INFERRED se a unidade foi omitida mas deduzida. Use REQUIRES_CLARIFICATION para ambiguidades.",
    )

    notes: str | None = Field(
        None,
        description="Observações cruciais. Obrigatório preencher caso o status não seja OK.",
    )
