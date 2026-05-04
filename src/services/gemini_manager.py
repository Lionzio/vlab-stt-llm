# src/services/gemini_manager.py
"""Gerenciador centralizado de clientes e roteamento de modelos Gemini.

Implementa o padrão Service/Manager para encapsular toda a lógica de:
- Inicialização segura de múltiplos clientes (chave primária e secundária)
- Roteamento de modelos (Flash para volume, Pro para tarefas complexas)
- Fallback transparente de cota entre chaves
- Controle de rate limit para o modelo Pro (uso restrito)

Projetado para ser injetado como dependência stateless em controladores
FastAPI e scripts de avaliação.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
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
_FLASH_MODELS: list[str] = ["gemini-2.0-flash", "gemini-1.5-flash"]

# Modelo Pro — raciocínio avançado, uso restrito
_PRO_MODEL: str = "gemini-2.5-pro"

# Limite diário conservador para o Pro no Free Tier
_PRO_DAILY_CALL_LIMIT: int = 5

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


class GeminiProLimitError(GeminiManagerError):
    """Limite diário de chamadas ao modelo Pro atingido."""


class GeminiGenerationError(GeminiManagerError):
    """Falha de geração após esgotar todos os fallbacks."""


# ---------------------------------------------------------------------------
# Estado interno de uso do modelo Pro
# ---------------------------------------------------------------------------


@dataclass
class _ProUsageTracker:
    """Rastreia uso diário do modelo Pro para evitar estouro de cota.

    Attributes:
        call_count: Número de chamadas realizadas no dia corrente.
        reset_timestamp: Unix timestamp do início do dia de rastreamento.
    """

    call_count: int = 0
    reset_timestamp: float = field(default_factory=time.time)

    def _reset_if_new_day(self) -> None:
        """Zera o contador se já passou mais de 24h desde o reset."""
        if time.time() - self.reset_timestamp >= 86_400:
            self.call_count = 0
            self.reset_timestamp = time.time()
            logger.info("[Pro] Contador diário resetado.")

    def check_and_increment(self, limit: int) -> None:
        """Verifica o limite e incrementa o contador se dentro do limite.

        Args:
            limit: Número máximo de chamadas permitidas por dia.

        Raises:
            GeminiProLimitError: Se o limite diário já foi atingido.
        """
        self._reset_if_new_day()
        if self.call_count >= limit:
            raise GeminiProLimitError(
                f"Limite diário do modelo Pro atingido "
                f"({self.call_count}/{limit}). "
                "Aguarde o reset de 24h ou eleve o _PRO_DAILY_CALL_LIMIT."
            )
        self.call_count += 1
        logger.info("[Pro] Chamada %d/%d autorizada.", self.call_count, limit)


# ---------------------------------------------------------------------------
# GeminiManager
# ---------------------------------------------------------------------------


class GeminiManager:
    """Gerencia clientes Gemini, roteamento de modelos e fallback de cotas.

    Padrão de uso (injeção de dependência):

        manager = GeminiManager()  # lê .env automaticamente

        # Tarefa simples via Flash:
        result = await manager.generate(contents="Resuma este texto.")

        # Tarefa complexa via Pro (com fallback automático para Flash):
        result = await manager.generate(
            contents="Analise em profundidade...",
            is_complex_task=True,
        )

    Attributes:
        primary_client: Cliente autenticado com GEMINI_API_KEY_PRIMARY.
        secondary_client: Cliente com GEMINI_API_KEY_SECONDARY (pode ser None).
        flash_model: Nome canônico do modelo Flash a usar.
        pro_model: Nome canônico do modelo Pro a usar.
    """

    def __init__(
        self,
        primary_key: str | None = None,
        secondary_key: str | None = None,
        flash_models: list[str] | None = None,
        pro_model: str | None = None,
        pro_daily_limit: int = _PRO_DAILY_CALL_LIMIT,
    ) -> None:
        """Inicializa o manager lendo as chaves do ambiente ou dos parâmetros.

        A chave primária é obrigatória. A secundária é opcional mas
        fortemente recomendada para produção em volume.

        Args:
            primary_key: Chave primária explícita (sobrescreve .env).
            secondary_key: Chave secundária explícita (sobrescreve .env).
            flash_models: Lista de modelos Flash em ordem de preferência.
            pro_model: Identificador do modelo Pro a utilizar.
            pro_daily_limit: Limite máximo de chamadas diárias ao modelo Pro.

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
        self.pro_model: str = pro_model or _PRO_MODEL
        self._pro_daily_limit: int = pro_daily_limit
        self._pro_tracker = _ProUsageTracker()

        logger.info(
            "[GeminiManager] Inicializado. Flash=%s | Pro=%s | "
            "Secundária=%s | Pro limit/dia=%d",
            self.flash_model,
            self.pro_model,
            "ativa" if self.secondary_client else "ausente",
            self._pro_daily_limit,
        )

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    async def generate(
        self,
        contents: Any,
        config: genai_types.GenerateContentConfig | None = None,
        is_complex_task: bool = False,
    ) -> genai_types.GenerateContentResponse:
        """Gera conteúdo roteando automaticamente entre modelos e chaves.

        Fluxo de roteamento:
            1. Se is_complex_task=True → tenta modelo Pro diretamente.
               Se Pro falhar (cota/limite) → cai para Flash.
            2. Flash com chave primária → fallback para chave secundária
               em caso de 429 → fallback para modelos Flash alternativos.
            3. Se todos os caminhos falharem → GeminiGenerationError.

        Args:
            contents: Conteúdo a ser enviado ao modelo (str, list, etc.).
            config: Configuração opcional de geração (temperatura, schema, etc.).
            is_complex_task: Se True, roteia primeiro para o modelo Pro.

        Returns:
            GenerateContentResponse do modelo que respondeu com sucesso.

        Raises:
            GeminiAuthError: Se as credenciais forem inválidas (fatal).
            GeminiQuotaError: Se todas as cotas forem esgotadas.
            GeminiGenerationError: Se todos os fallbacks falharem.
        """
        if is_complex_task:
            return await self._generate_with_pro_fallback(contents, config)
        return await self._generate_with_flash_fallback(contents, config)

    async def execute_structured_with_fallback(
        self,
        contents: Any,
        response_schema: type,
        system_instruction: str,
        is_complex_task: bool = False,
    ) -> Any:
        """Executa geração estruturada (JSON com schema Pydantic) via Manager.

        Usado pelo ParameterExtractor (V1) para obter output diretamente
        validado contra um schema Pydantic, com roteamento de cotas e
        fallback de modelos gerenciados centralmente.

        Args:
            contents: Texto ou conteúdo a ser enviado ao modelo.
            response_schema: Classe Pydantic que define o schema de resposta.
            system_instruction: Instrução de sistema para o modelo.
            is_complex_task: Se True, roteia para o modelo Pro primeiro.

        Returns:
            Instância do response_schema preenchida e validada.

        Raises:
            GeminiAuthError: Se as credenciais forem inválidas (fatal).
            GeminiQuotaError: Se todas as cotas forem esgotadas.
            GeminiGenerationError: Se todos os fallbacks falharem.
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
            is_complex_task=is_complex_task,
        )

        if not response.text:
            raise GeminiGenerationError(
                "API retornou resposta vazia para geração estruturada."
            )

        return response_schema.model_validate_json(response.text)

    async def _execute_with_quota_fallback(
        self,
        model: str,
        contents: Any,
        config: Any,
    ) -> str:
        """Executa geração de texto livre com fallback de cotas e modelos.

        Usado pelo ParameterExtractorV2 (CoT) que precisa de resposta em
        texto livre (não JSON estruturado) para permitir o bloco <reasoning>.
        Roteia pela lógica de fallback do Manager e retorna o texto bruto.

        Args:
            model: Identificador do modelo preferido. Se não disponível,
                o Manager degrada para os modelos Flash disponíveis.
            contents: Texto ou conteúdo a ser enviado ao modelo.
            config: GenerateContentConfig com mime_type e instrução de sistema.

        Returns:
            Texto bruto retornado pelo modelo (response.text).

        Raises:
            GeminiAuthError: Se as credenciais forem inválidas (fatal).
            GeminiQuotaError: Se todas as cotas forem esgotadas.
            GeminiGenerationError: Se todos os fallbacks falharem ou resposta vazia.
        """
        # Determina se é tarefa complexa baseado no modelo solicitado
        is_complex = model == self.pro_model

        response = await self.generate(
            contents=contents,
            config=config,
            is_complex_task=is_complex,
        )

        if not response.text:
            raise GeminiGenerationError(
                "API retornou resposta vazia para geração de texto livre."
            )

        return response.text

    # ------------------------------------------------------------------
    # Roteamento Pro
    # ------------------------------------------------------------------

    async def _generate_with_pro_fallback(
        self,
        contents: Any,
        config: genai_types.GenerateContentConfig | None,
    ) -> genai_types.GenerateContentResponse:
        """Tenta Pro; se falhar por limite/cota, degrada para Flash.

        Args:
            contents: Conteúdo para geração.
            config: Configuração de geração.

        Returns:
            Response do modelo que respondeu com sucesso.
        """
        try:
            self._pro_tracker.check_and_increment(self._pro_daily_limit)
        except GeminiProLimitError as exc:
            logger.warning("[GeminiManager] %s — degradando para Flash.", exc)
            return await self._generate_with_flash_fallback(contents, config)

        try:
            return await self._call_model(
                client=self.primary_client,
                model_id=self.pro_model,
                contents=contents,
                config=config,
            )

        except GeminiAuthError:
            raise

        except (GeminiQuotaError, GeminiGenerationError) as exc:
            logger.warning(
                "[GeminiManager] Pro falhou (%s) — degradando para Flash.",
                exc,
            )
            # Estorna o incremento do contador Pro (chamada não completou)
            self._pro_tracker.call_count = max(0, self._pro_tracker.call_count - 1)
            return await self._generate_with_flash_fallback(contents, config)

    # ------------------------------------------------------------------
    # Roteamento Flash com fallback de chave e modelo
    # ------------------------------------------------------------------

    async def _generate_with_flash_fallback(
        self,
        contents: Any,
        config: genai_types.GenerateContentConfig | None,
    ) -> genai_types.GenerateContentResponse:
        """Tenta Flash na chave primária; roteia para secundária em 429.

        Se a secundária também falhar por cota, itera pelos modelos Flash
        alternativos na chave primária antes de desistir.

        Args:
            contents: Conteúdo para geração.
            config: Configuração de geração.

        Returns:
            Response do primeiro modelo/chave que responder com sucesso.

        Raises:
            GeminiAuthError: Se as credenciais forem inválidas.
            GeminiQuotaError: Se todas as cotas forem esgotadas.
            GeminiGenerationError: Se todos os modelos e chaves falharem.
        """
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
                    "esgotada. Tentando modelos Flash alternativos..."
                )
            except GeminiGenerationError as exc:
                last_exc = exc
                logger.warning("[GeminiManager] Flash na secundária falhou: %s.", exc)

        # --- Tentativa 3: Modelos Flash alternativos na chave primária ---
        for alt_model in self._flash_models[1:]:
            try:
                logger.info(
                    "[GeminiManager] Tentando modelo Flash alternativo: %s",
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
        """Executa uma única chamada de geração com timeout explícito.

        Classifica as exceções do SDK em tipos semânticos do Manager,
        evitando que erros estruturais sejam mascarados por blocos genéricos.

        Args:
            client: Cliente Gemini autenticado para esta chamada.
            model_id: Identificador do modelo a invocar.
            contents: Conteúdo a ser enviado.
            config: Configuração opcional de geração.

        Returns:
            GenerateContentResponse em caso de sucesso.

        Raises:
            GeminiAuthError: Para erros 401/403 (fatal).
            GeminiQuotaError: Para erros 429 (tratável por fallback).
            GeminiGenerationError: Para outros erros de API ou timeout.
        """
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
