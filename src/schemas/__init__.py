from pydantic import BaseModel, Field

from src.schemas.extraction import MedicalParameterExtraction

# Isso ensina ao Python (e ao Ruff) quais classes pertencem à interface pública deste pacote
__all__ = ["HealthCheckResponse", "MedicalParameterExtraction"]


class HealthCheckResponse(BaseModel):
    """Schema de resposta para o endpoint de health check.

    Attributes:
        status: Status da aplicação. Esperado: "ok".
        service: Identificador canônico do serviço.
        version: Versão semântica atual da API.
        message: Mensagem legível confirmando a operação.
    """

    status: str = Field(..., examples=["ok"])
    service: str = Field(..., examples=["vlab-stt-llm"])
    version: str = Field(..., examples=["0.1.0"])
    message: str = Field(..., examples=["API operacional"])
