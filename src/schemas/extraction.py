# src/schemas/extraction.py
"""Schemas para extração estruturada de parâmetros médicos via LLM."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Tabelas de Domínio Canônico (Hard Rules)
# ---------------------------------------------------------------------------

DEFAULT_UNITS: dict[str, str] = {
    "peep": "cmH2O",
    "fio2": "%",
    "frequencia_respiratoria": "irpm",
    "pressao_arterial": "mmHg",
    "volume_corrente": "ml",
    "frequencia_cardiaca": "bpm",
}

# Limites clínicos seguros (mínimo, máximo)
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "peep": (0.0, 25.0),
    "fio2": (21.0, 100.0),
    "frequencia_respiratoria": (4.0, 60.0),
    "volume_corrente": (200.0, 800.0),
    "frequencia_cardiaca": (20.0, 300.0),
    # A pressão arterial (quando dita "doze por oito") não entra aqui
    # pois o value é extraído como None, ativando REQUIRES_CLARIFICATION.
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MedicalParameterExtraction(BaseModel):
    """Schema estrito para a saída da extração de parâmetros via voz.

    O LLM é forçado a preencher os campos com base neste contrato.
    Possui uma camada híbrida de pós-processamento para garantir unidades
    seguras e barrar valores clinicamente absurdos.
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
        description="O parâmetro médico alvo. Ex: 'peep', 'fio2'. Se não houver, retorne nulo.",
    )

    value: float | None = Field(
        None,
        description="O valor numérico a ser ajustado. Se for fracionado, retorne nulo e preencha 'notes'.",
    )

    unit: str | None = Field(
        None,
        description="A unidade de medida. Ex: 'cmH2O', '%'. Se não couber unidade, retorne nulo.",
    )

    status: Literal[
        "OK",
        "OK_INFERRED_UNIT",
        "OK_INFERRED_UNIT_BY_RULE",  # Adicionado pela abordagem híbrida
        "MISSING_VALUE",
        "OUT_OF_BOUNDS",
        "REQUIRES_CLARIFICATION",
        "ERROR",
    ] = Field(
        ...,
        description="O status da extração. Use OK_INFERRED_UNIT se a unidade foi deduzida pelo LLM.",
    )

    notes: str | None = Field(
        None,
        description="Observações cruciais. Obrigatório preencher caso o status não seja OK.",
    )

    @model_validator(mode="after")
    def apply_deterministic_rules(self) -> "MedicalParameterExtraction":
        """Abordagem Híbrida: Fallback e Limites Clínicos (Hard Rules).

        Atua como uma rede de segurança contra alucinações matemáticas ou
        esquecimentos do LLM, garantindo integridade dos dados finais.
        """
        if not self.parameter:
            return self

        normalized_param = self.parameter.lower().strip()

        # 1. Injeção Determinística de Unidades
        if not self.unit and normalized_param in DEFAULT_UNITS:
            self.unit = DEFAULT_UNITS[normalized_param]

            # Só alteramos o status se ele não for um erro mais grave (ex: OUT_OF_BOUNDS)
            if self.status in ("OK", "MISSING_VALUE"):
                self.status = "OK_INFERRED_UNIT_BY_RULE"

            note_msg = f"Unidade '{self.unit}' injetada por regra determinística."
            self.notes = f"{self.notes} | {note_msg}" if self.notes else note_msg

        # 2. Validação de Limites de Segurança (Safety Bounds)
        if self.value is not None and normalized_param in PARAMETER_BOUNDS:
            min_val, max_val = PARAMETER_BOUNDS[normalized_param]

            if not (min_val <= self.value <= max_val):
                self.status = "OUT_OF_BOUNDS"
                bound_msg = (
                    f"ALERTA CLÍNICO: Valor {self.value} fora dos limites seguros "
                    f"para {normalized_param} ({min_val}-{max_val})."
                )
                self.notes = f"{self.notes} | {bound_msg}" if self.notes else bound_msg

        return self
