# src/services/gemini_manager.py
"""Gerenciador centralizado de clientes e roteamento de modelos Gemini.

Implementa o padrão Service/Manager para encapsular toda a lógica de:
- Inicialização segura de múltiplos clientes (chave primária e secundária)
- Fallback transparente de cota entre chaves
- Roteamento entre modelos Flash (alta velocidade e volume)

Projetado para ser injetado como dependência stateless em controladores
FastAPI e scripts de avaliação.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de modelo e limites
# ---------------------------------------------------------------------------

# Modelos Flash — alta velocidade, processamento em massa
_FLASH_MODELS: list[str] = ["gemini-2.5-flash", "gemini-3.1-flash-lite"]

# Timeout por chamada de geração (segundos)
_GENERATION_TIMEOUT_S: int = 90


# ---------------------------------------------------------------------------
# Exceções do Manager
# ---------------------------------------------------------------------------


class GeminiManagerError(Exception):
    """Erro base do GeminiManager."""


class GeminiAuthError(GeminiManagerError):
    """Falha de autenticação (401/403) — erro fatal de configuração."""


class GeminiQuotaError(GeminiManagerError):
    """Todas as cotas disponíveis foram esgotadas."""


class GeminiGenerationError(GeminiManagerError):
    """Falha de geração após esgotar todos os fallbacks."""


# ---------------------------------------------------------------------------
# GeminiManager
# ---------------------------------------------------------------------------


class GeminiManager:
    """Gerencia clientes Gemini, roteamento de modelos e fallback de cotas.

    Padrão de uso (injeção de dependência):

        manager = GeminiManager()  # lê .env automaticamente
        result = await manager.generate(contents="Resuma este texto.")

    Attributes:
        primary_client: Cliente autenticado com GEMINI_API_KEY_PRIMARY.
        secondary_client: Cliente com GEMINI_API_KEY_SECONDARY (pode ser None).
        flash_model: Nome canônico do modelo Flash primário a usar.
    """

    def __init__(
        self,
        primary_key: str | None = None,
        secondary_key: str | None = None,
        flash_models: list[str] | None = None,
    ) -> None:
        """Inicializa o manager lendo as chaves do ambiente ou dos parâmetros.

        A chave primária é obrigatória. A secundária é opcional mas
        fortemente recomendada para produção em volume.

        Args:
            primary_key: Chave primária explícita (sobrescreve .env).
            secondary_key: Chave secundária explícita (sobrescreve .env).
            flash_models: Lista de modelos Flash em ordem de preferência.

        Raises:
            GeminiAuthError: Se GEMINI_API_KEY_PRIMARY não estiver disponível.
        """
        _primary = (primary_key or os.getenv("GEMINI_API_KEY_PRIMARY", "")).strip()
        _secondary = (
            secondary_key or os.getenv("GEMINI_API_KEY_SECONDARY", "")
        ).strip()

        if not _primary:
            raise GeminiAuthError(
                "GEMINI_API_KEY_PRIMARY não encontrada. "
                "Defina a variável no arquivo .env na raiz do projeto."
            )

        self.primary_client = genai.Client(api_key=_primary)
        self.secondary_client = genai.Client(api_key=_secondary) if _secondary else None

        self._flash_models: list[str] = flash_models or _FLASH_MODELS
        self.flash_model: str = self._flash_models[0]

        logger.info(
            "[GeminiManager] Inicializado. Flash Principal=%s | Secundária=%s",
            self.flash_model,
            "ativa" if self.secondary_client else "ausente",
        )

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    async def generate(
        self,
        contents: Any,
        config: genai_types.GenerateContentConfig | None = None,
    ) -> genai_types.GenerateContentResponse:
        """Gera conteúdo roteando automaticamente entre chaves e modelos Flash.

        Fluxo de roteamento:
            1. Flash com chave primária → fallback para chave secundária.
            2. Se secundária der 429 → fallback para modelos Flash alternativos.
            3. Se todos os caminhos falharem → GeminiGenerationError.

        Args:
            contents: Conteúdo a ser enviado ao modelo (str, list, etc.).
            config: Configuração opcional de geração (temperatura, schema, etc.).

        Returns:
            GenerateContentResponse do modelo que respondeu com sucesso.

        Raises:
            GeminiAuthError: Se as credenciais forem inválidas (fatal).
            GeminiQuotaError: Se todas as cotas forem esgotadas.
            GeminiGenerationError: Se todos os fallbacks falharem.
        """
        return await self._generate_with_fallback(contents, config)

    async def execute_structured_with_fallback(
        self,
        contents: Any,
        response_schema: type,
        system_instruction: str,
    ) -> Any:
        """Executa geração estruturada (JSON com schema Pydantic) via Manager.

        Args:
            contents: Texto ou conteúdo a ser enviado ao modelo.
            response_schema: Classe Pydantic que define o schema de resposta.
            system_instruction: Instrução de sistema para o modelo.

        Returns:
            Instância do response_schema preenchida e validada.
        """
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
        )

        response = await self.generate(
            contents=contents,
            config=config,
        )

        if not response.text:
            raise GeminiGenerationError(
                "API retornou resposta vazia para geração estruturada."
            )

        return response_schema.model_validate_json(response.text)

    async def _execute_with_quota_fallback(
        self,
        contents: Any,
        config: Any,
    ) -> str:
        """Executa geração de texto livre com fallback de cotas e modelos.

        Args:
            contents: Texto ou conteúdo a ser enviado ao modelo.
            config: GenerateContentConfig com mime_type e instrução de sistema.

        Returns:
            Texto bruto retornado pelo modelo (response.text).
        """
        response = await self.generate(
            contents=contents,
            config=config,
        )

        if not response.text:
            raise GeminiGenerationError(
                "API retornou resposta vazia para geração de texto livre."
            )

        return response.text

    # ------------------------------------------------------------------
    # Roteamento com fallback de chave e modelo
    # ------------------------------------------------------------------

    async def _generate_with_fallback(
        self,
        contents: Any,
        config: genai_types.GenerateContentConfig | None,
    ) -> genai_types.GenerateContentResponse:
        """Tenta Flash na chave primária; roteia para secundária em 429."""
        last_exc: Exception | None = None

        # --- Tentativa 1: Flash primário na chave primária ---
        try:
            return await self._call_model(
                client=self.primary_client,
                model_id=self.flash_model,
                contents=contents,
                config=config,
            )
        except GeminiAuthError:
            raise
        except GeminiQuotaError as exc:
            last_exc = exc
            logger.warning(
                "[GeminiManager] Chave PRIMÁRIA com cota esgotada. "
                "Tentando chave SECUNDÁRIA..."
            )
        except GeminiGenerationError as exc:
            last_exc = exc
            logger.warning(
                "[GeminiManager] Flash primário falhou: %s. "
                "Tentando chave secundária...",
                exc,
            )

        # --- Tentativa 2: Flash primário na chave secundária ---
        if self.secondary_client is not None:
            try:
                return await self._call_model(
                    client=self.secondary_client,
                    model_id=self.flash_model,
                    contents=contents,
                    config=config,
                )
            except GeminiAuthError:
                raise
            except GeminiQuotaError as exc:
                last_exc = exc
                logger.warning(
                    "[GeminiManager] Chave SECUNDÁRIA também com cota "
                    "esgotada. Tentando modelos alternativos..."
                )
            except GeminiGenerationError as exc:
                last_exc = exc
                logger.warning("[GeminiManager] Chamada na secundária falhou: %s.", exc)

        # --- Tentativa 3: Modelos alternativos na chave primária ---
        for alt_model in self._flash_models[1:]:
            try:
                logger.info(
                    "[GeminiManager] Tentando modelo alternativo: %s",
                    alt_model,
                )
                return await self._call_model(
                    client=self.primary_client,
                    model_id=alt_model,
                    contents=contents,
                    config=config,
                )
            except GeminiAuthError:
                raise
            except (GeminiQuotaError, GeminiGenerationError) as exc:
                last_exc = exc
                logger.warning("[GeminiManager] Modelo %s falhou: %s", alt_model, exc)

        # Todos os caminhos esgotados
        if isinstance(last_exc, GeminiQuotaError):
            raise GeminiQuotaError(
                "Todas as cotas disponíveis foram esgotadas "
                "(primária, secundária e modelos alternativos)."
            ) from last_exc

        raise GeminiGenerationError(
            f"Todos os modelos e chaves falharam. Último erro: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Chamada atômica a um modelo específico
    # ------------------------------------------------------------------

    async def _call_model(
        self,
        client: genai.Client,
        model_id: str,
        contents: Any,
        config: genai_types.GenerateContentConfig | None,
    ) -> genai_types.GenerateContentResponse:
        """Executa uma única chamada de geração com timeout explícito."""
        kwargs: dict[str, Any] = {
            "model": model_id,
            "contents": contents,
        }
        if config is not None:
            kwargs["config"] = config

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(**kwargs),
                timeout=_GENERATION_TIMEOUT_S,
            )
            logger.debug("[GeminiManager] Sucesso: modelo=%s", model_id)
            return response

        except genai_errors.ClientError as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

            if status in (401, 403):
                raise GeminiAuthError(
                    f"Falha de autenticação com a API Gemini (HTTP {status})."
                ) from exc

            if status == 429:
                raise GeminiQuotaError(
                    f"Cota da API excedida no modelo {model_id} (HTTP 429)."
                ) from exc

            raise GeminiGenerationError(
                f"Erro de cliente na API Gemini "
                f"(modelo={model_id}, HTTP {status}): {exc}"
            ) from exc

        except genai_errors.ServerError as exc:
            raise GeminiGenerationError(
                f"Erro de servidor na API Gemini (modelo={model_id}): {exc}"
            ) from exc

        except asyncio.TimeoutError as exc:
            raise GeminiGenerationError(
                f"Timeout de {_GENERATION_TIMEOUT_S}s excedido "
                f"para o modelo {model_id}."
            ) from exc

        except genai_errors.APIError as exc:
            raise GeminiGenerationError(
                f"Erro inesperado da API Gemini (modelo={model_id}): {exc}"
            ) from exc
