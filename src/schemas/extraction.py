# src/schemas/extraction.py
"""Schemas para extração estruturada de parâmetros médicos via LLM."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Tabela de domínio canônico para inferência determinística (Hard Rules)
DEFAULT_UNITS: dict[str, str] = {
    "peep": "cmH2O",
    "fio2": "%",
    "frequencia_respiratoria": "irpm",
    "pressao_arterial": "mmHg",
    "volume_corrente": "ml",
    "frequencia_cardiaca": "bpm",
}


class MedicalParameterExtraction(BaseModel):
    """Schema estrito para a saída da extração de parâmetros via voz.

    O LLM é forçado a preencher os campos com base neste contrato.
    Possui uma camada híbrida de pós-processamento para garantir unidades seguras.
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
        "OK_INFERRED_UNIT_BY_RULE",  # Status adicionado para a abordagem híbrida
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
        """Abordagem Híbrida: Fallback Determinístico de Unidades.

        Atua como uma rede de segurança. Se o modelo probabilístico (LLM)
        falhar em inferir a unidade de medida para um parâmetro conhecido,
        esta regra de software (hard rule) injeta a unidade padrão correta
        e ajusta o status de auditoria. Isso garante consistência de schema
        sem depender exclusivamente de IA.
        """
        # Verifica se temos um parâmetro válido, mas a unidade veio vazia/nula
        if self.parameter and not self.unit:
            # Normaliza o parâmetro para busca
            normalized_param = self.parameter.lower().strip()

            # Se o parâmetro estiver na nossa tabela de domínio canônica
            if normalized_param in DEFAULT_UNITS:
                self.unit = DEFAULT_UNITS[normalized_param]
                self.status = "OK_INFERRED_UNIT_BY_RULE"

                # Se 'notes' estiver vazio, adiciona um rastro de auditoria
                if not self.notes:
                    self.notes = (
                        f"Unidade '{self.unit}' injetada por regra determinística."
                    )

        return self
