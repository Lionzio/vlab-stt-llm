# scripts/evaluate_pipeline.py
"""Script de avaliação automatizada A/B do pipeline vlab-stt-llm.

Executa cada caso de teste contra V1 (Direct) e V2 (Chain-of-Thought),
utilizando cache local para contornar limites de cota da API.
Controla concorrência via asyncio.Semaphore para nunca ultrapassar 2
requisições paralelas ativas — mitigando erros 429 do Free Tier.
Gera relatório com métricas WER/CER, Precision, Recall e F1-Score.

Injeta o GeminiManager nos serviços para gerenciar rotação de chaves
e roteamento de modelos.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cache import get_llm, get_stt, set_llm, set_stt  # noqa: E402
from scripts.metrics import TranscriptionMetrics  # noqa: E402
from scripts.metrics import compute as compute_metrics  # noqa: E402
from src.services.extractor import (  # noqa: E402
    MedicalParameterExtraction,
    ParameterExtractor,
)
from src.services.extractor_v2 import ParameterExtractorV2  # noqa: E402
from src.services.gemini_manager import GeminiManager  # noqa: E402
from src.services.stt import GeminiSTT, STTError  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

# True  → usa cache local (.cache/) sempre que disponível (sem consumir API)
# False → modo live, consome API do Google diretamente
USE_CACHE: bool = True

# Número máximo de requisições PARALELAS ativas à API do Gemini.
# Valor 2 evita explosão de 429 no Free Tier (RPM baixo).
_SEMAPHORE_LIMIT: int = 2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("evaluate_pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio_samples"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
DOCS_DIR = REPO_ROOT / "docs"
REPORT_PATH = DOCS_DIR / "evaluation_report.md"

# ---------------------------------------------------------------------------
# Augmentation (opcional — só ativo se o módulo existir)
# ---------------------------------------------------------------------------

try:
    from scripts.audio_augmentation import inject_hospital_noise as _inject_noise

    NOISE_SAMPLE_PATH = DATA_DIR / "noise_samples" / "hospital_ambient.mp3"
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    NOISE_SAMPLE_PATH = None  # type: ignore[assignment]

    def _inject_noise(*args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Modelos de dados
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Representa um caso de teste carregado do ground_truth.json.

    Attributes:
        id: Identificador único do caso (ex: "TC-001").
        audio_filename: Nome do arquivo .mp3 em data/audio_samples/.
        scenario_type: Categoria do cenário de teste.
        expected_transcription: Transcrição ideal esperada do STT.
        expected_intent: Intenção esperada na extração semântica.
        expected_parameter: Parâmetro clínico esperado (pode ser None).
        expected_value: Valor numérico esperado (pode ser None).
        expected_unit: Unidade canônica esperada (pode ser None).
        expected_status: Status de validação esperado pelo pipeline.
        notes: Observações adicionais do caso de borda.
    """

    id: str
    audio_filename: str
    scenario_type: str
    expected_transcription: str
    expected_intent: str
    expected_parameter: str | None
    expected_value: float | None
    expected_unit: str | None
    expected_status: str
    notes: str = ""


@dataclass
class CaseResult:
    """Resultado da execução de um caso de teste no pipeline.

    Attributes:
        ground_truth: Caso de teste de referência.
        extractor_version: Identificador da versão do extrator.
        stt_success: True se o STT produziu texto não-vazio.
        stt_transcript: Texto transcrito pelo STT.
        stt_from_cache: True se a transcrição veio do cache local.
        schema_adherence: True se o Extractor retornou objeto Pydantic válido.
        extraction: Objeto de extração retornado (None em caso de falha).
        llm_from_cache: True se a extração LLM veio do cache local.
        intent_match: True se o intent extraído bate com o esperado.
        parameter_match: True se o parameter extraído bate com o esperado.
        status_match: True se o status de validação bate com o esperado.
        metrics: Métricas WER/CER (None se STT falhou).
        latency_s: Tempo total de execução do caso em segundos.
        error_message: Mensagem de erro capturada, se houver.
    """

    ground_truth: GroundTruth
    extractor_version: str = "v1"
    stt_success: bool = False
    stt_transcript: str = ""
    stt_from_cache: bool = False
    schema_adherence: bool = False
    extraction: MedicalParameterExtraction | None = None
    llm_from_cache: bool = False
    intent_match: bool = False
    parameter_match: bool = False
    status_match: bool = False
    metrics: TranscriptionMetrics | None = None
    latency_s: float = 0.0
    error_message: str = ""

    @property
    def overall_pass(self) -> bool:
        """Caso passa se STT, schema e intent estão todos corretos."""
        return self.stt_success and self.schema_adherence and self.intent_match


@dataclass
class MLEvaluationMetrics:
    """Precision, Recall e F1-Score para uma entidade extraída.

    Attributes:
        precision: Fração de extrações corretas sobre total extraído.
        recall: Fração de extrações corretas sobre total esperado.
        f1: Média harmônica entre precision e recall.
    """

    precision: float
    recall: float
    f1: float

    @property
    def precision_pct(self) -> str:
        """Precision formatada como percentual."""
        return f"{self.precision * 100:.1f}%"

    @property
    def recall_pct(self) -> str:
        """Recall formatado como percentual."""
        return f"{self.recall * 100:.1f}%"

    @property
    def f1_pct(self) -> str:
        """F1-Score formatado como percentual."""
        return f"{self.f1 * 100:.1f}%"


# ---------------------------------------------------------------------------
# Carregamento do ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(path: Path) -> list[GroundTruth]:
    """Carrega e deserializa os casos de teste do arquivo JSON.

    Args:
        path: Caminho para o arquivo ground_truth.json.

    Returns:
        Lista de instâncias GroundTruth.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    with path.open(encoding="utf-8") as fh:
        raw: list[dict[str, Any]] = json.load(fh)

    cases: list[GroundTruth] = []
    for item in raw:
        extraction = item.get("expected_extraction", {})
        cases.append(
            GroundTruth(
                id=item["id"],
                audio_filename=item["audio_filename"],
                scenario_type=item["scenario_type"],
                expected_transcription=item["expected_transcription"],
                expected_intent=extraction.get("intent", ""),
                expected_parameter=extraction.get("parameter"),
                expected_value=extraction.get("value"),
                expected_unit=extraction.get("unit"),
                expected_status=extraction.get("status", ""),
                notes=extraction.get("notes", ""),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Execução de um caso (com Semaphore)
# ---------------------------------------------------------------------------


async def run_case(
    gt: GroundTruth,
    stt: GeminiSTT,
    extractor: ParameterExtractor | ParameterExtractorV2,
    extractor_version: str,
    semaphore: asyncio.Semaphore,
) -> CaseResult:
    """Executa o pipeline completo para um único caso de teste."""
    result = CaseResult(ground_truth=gt, extractor_version=extractor_version)
    original_audio_path = AUDIO_DIR / gt.audio_filename
    t_start = time.monotonic()

    try:
        if not original_audio_path.exists():
            result.error_message = f"Arquivo não encontrado: {original_audio_path}"
            logger.warning("[%s] %s", gt.id, result.error_message)
            return result

        # --- SPRINT 9: TESTE DE STRESS (DATA AUGMENTATION) ---
        target_audio_path = original_audio_path
        ignore_cache_for_stress = False

        if gt.scenario_type == "ruido_simulado" and AUGMENTATION_AVAILABLE:
            stress_output_path = AUDIO_DIR / f"{gt.id}_stress_tested.mp3"
            logger.info("[%s] Iniciando Teste de Stress (Injeção de Ruído)...", gt.id)

            target_path_str = _inject_noise(
                clean_audio_path=original_audio_path,
                noise_audio_path=NOISE_SAMPLE_PATH,
                output_path=stress_output_path,
            )
            target_audio_path = (
                Path(target_path_str) if target_path_str else original_audio_path
            )
            # Força consulta à API para ver o desempenho real contra o ruído gerado agora
            ignore_cache_for_stress = True
        # --------------------------------------------------------

        # ------------------------------------------------------------------
        # Etapa 1 — STT (cache → API com Semaphore)
        # ------------------------------------------------------------------
        # Modificamos a regra de cache para ignorar se for um teste de stress
        cached_transcript = None
        if USE_CACHE and not ignore_cache_for_stress:
            cached_transcript = get_stt(str(target_audio_path))

        if cached_transcript is not None:
            transcript = cached_transcript
            result.stt_from_cache = True
            logger.info("[%s] STT via cache: %r", gt.id, transcript)
        else:
            async with semaphore:
                logger.info("[%s] STT via API (semaphore adquirido)...", gt.id)
                transcript = await stt.transcribe(str(target_audio_path))

            if not transcript:
                result.error_message = "STT retornou transcrição vazia."
                logger.warning("[%s] %s", gt.id, result.error_message)
                return result

            if not ignore_cache_for_stress:
                set_stt(str(target_audio_path), transcript)

        result.stt_success = True
        result.stt_transcript = transcript

        # ------------------------------------------------------------------
        # Etapa 1b — WER/CER
        # ------------------------------------------------------------------
        result.metrics = compute_metrics(
            reference=gt.expected_transcription,
            hypothesis=transcript,
        )
        logger.info(
            "[%s] WER=%s CER=%s",
            gt.id,
            result.metrics.wer_pct,
            result.metrics.cer_pct,
        )

        if transcript.strip() != gt.expected_transcription.strip().lower():
            logger.info(
                "[%s] Divergência STT:\n  Esperado: %r\n  Obtido  : %r",
                gt.id,
                gt.expected_transcription,
                transcript,
            )

        # ------------------------------------------------------------------
        # Etapa 2 — Extração LLM (cache → API com Semaphore)
        # ------------------------------------------------------------------
        cached_llm = get_llm(transcript, extractor_version) if USE_CACHE else None

        if cached_llm is not None:
            extraction = MedicalParameterExtraction.model_validate(cached_llm)
            result.llm_from_cache = True
            logger.info("[%s][%s] Extração via cache.", gt.id, extractor_version)
        else:
            async with semaphore:
                logger.info(
                    "[%s][%s] Extração via API (semaphore adquirido)...",
                    gt.id,
                    extractor_version,
                )
                extraction = await extractor.extract(transcript)

            if extraction is None:
                result.error_message = "Extractor retornou None."
                logger.warning("[%s] %s", gt.id, result.error_message)
                return result

            set_llm(transcript, extraction.model_dump(), extractor_version)

        result.schema_adherence = True
        result.extraction = extraction

        # ------------------------------------------------------------------
        # Etapa 3 — Comparação com gabarito
        # ------------------------------------------------------------------
        result.intent_match = extraction.intent == gt.expected_intent

        if gt.expected_parameter is None:
            result.parameter_match = extraction.parameter is None
        else:
            result.parameter_match = (
                extraction.parameter or ""
            ).lower() == gt.expected_parameter.lower()

        result.status_match = extraction.status == gt.expected_status

        logger.info(
            "[%s][%s] intent=%s param=%s status=%s",
            gt.id,
            extractor_version,
            "✓" if result.intent_match else "✗",
            "✓" if result.parameter_match else "✗",
            "✓" if result.status_match else "✗",
        )

    except STTError as exc:
        result.error_message = f"STTError: {exc}"
        logger.error("[%s] %s", gt.id, result.error_message)

    except Exception as exc:  # noqa: BLE001
        result.schema_adherence = False
        result.error_message = f"Extractor falhou: {exc}"
        logger.error("[%s] %s", gt.id, result.error_message)

    finally:
        result.latency_s = time.monotonic() - t_start

    return result


# ---------------------------------------------------------------------------
# Métricas de ML
# ---------------------------------------------------------------------------


def calculate_ml_metrics(results: list[CaseResult], field: str) -> MLEvaluationMetrics:
    """Calcula Precision, Recall e F1-Score para um campo de extração.

    Args:
        results: Lista de resultados de casos avaliados.
        field: Campo a avaliar — "intent" ou "parameter".

    Returns:
        MLEvaluationMetrics com precision, recall e f1.
    """
    tp = fp = fn = 0

    for r in results:
        gt_val = getattr(r.ground_truth, f"expected_{field}")
        ext_val = getattr(r.extraction, field) if r.extraction else None

        if isinstance(gt_val, str):
            gt_val = gt_val.strip().lower()
        if isinstance(ext_val, str):
            ext_val = ext_val.strip().lower()

        if gt_val == ext_val:
            if gt_val:
                tp += 1
        else:
            if ext_val:
                fp += 1
            if gt_val:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return MLEvaluationMetrics(precision, recall, f1)


# ---------------------------------------------------------------------------
# Exibição Rich
# ---------------------------------------------------------------------------

_STATUS_ICON = {True: "✅", False: "❌"}


def print_rich_summary_ab(
    results_v1: list[CaseResult],
    results_v2: list[CaseResult],
) -> None:
    """Exibe tabelas comparativas A/B e de métricas ML no terminal.

    Args:
        results_v1: Resultados do extrator V1 (Direct).
        results_v2: Resultados do extrator V2 (CoT).
    """
    table = Table(
        title="[bold cyan]Avaliação A/B — V1 (Direct) vs V2 (CoT)[/bold cyan]",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="bold", width=8)
    table.add_column("Cenário", style="cyan", max_width=24)
    table.add_column("WER", justify="center", width=7)
    table.add_column("V1 Intent", justify="center", width=10)
    table.add_column("V2 Intent", justify="center", width=10)
    table.add_column("V1 Status", justify="center", width=10)
    table.add_column("V2 Status", justify="center", width=10)
    table.add_column("Cache", justify="center", width=7)

    for r1, r2 in zip(results_v1, results_v2, strict=False):
        gt = r1.ground_truth
        wer = r1.metrics.wer_pct if r1.metrics else "N/A"
        cache_label = "✓" if (r1.stt_from_cache and r1.llm_from_cache) else "—"

        table.add_row(
            gt.id,
            gt.scenario_type,
            wer,
            "[green]✓[/green]" if r1.intent_match else "[red]✗[/red]",
            "[green]✓[/green]" if r2.intent_match else "[red]✗[/red]",
            "[green]✓[/green]" if r1.status_match else "[red]✗[/red]",
            "[green]✓[/green]" if r2.status_match else "[red]✗[/red]",
            f"[dim]{cache_label}[/dim]",
        )

    console.print()
    console.print(table)

    ml_table = Table(
        title=(
            "[bold yellow]Métricas de ML "
            "(Precision / Recall / F1-Score)[/bold yellow]"
        ),
        show_lines=True,
        header_style="bold yellow",
    )
    ml_table.add_column("Entidade", style="bold", width=12)
    ml_table.add_column("V1 Precision", justify="center", width=12)
    ml_table.add_column("V1 Recall", justify="center", width=12)
    ml_table.add_column("V1 F1-Score", justify="center", width=12, style="bold green")
    ml_table.add_column("V2 Precision", justify="center", width=12)
    ml_table.add_column("V2 Recall", justify="center", width=12)
    ml_table.add_column("V2 F1-Score", justify="center", width=12, style="bold green")

    i1 = calculate_ml_metrics(results_v1, "intent")
    i2 = calculate_ml_metrics(results_v2, "intent")
    p1 = calculate_ml_metrics(results_v1, "parameter")
    p2 = calculate_ml_metrics(results_v2, "parameter")

    ml_table.add_row(
        "Intent",
        i1.precision_pct,
        i1.recall_pct,
        i1.f1_pct,
        i2.precision_pct,
        i2.recall_pct,
        i2.f1_pct,
    )
    ml_table.add_row(
        "Parameter",
        p1.precision_pct,
        p1.recall_pct,
        p1.f1_pct,
        p2.precision_pct,
        p2.recall_pct,
        p2.f1_pct,
    )

    console.print()
    console.print(ml_table)
    console.print()


# ---------------------------------------------------------------------------
# Geração do relatório Markdown
# ---------------------------------------------------------------------------


def _narrative_for(result: CaseResult) -> str:
    """Gera análise narrativa contextualizada por tipo de cenário.

    Args:
        result: Resultado do caso de teste avaliado.

    Returns:
        String com a análise narrativa ou string vazia.
    """
    scenario = result.ground_truth.scenario_type
    ext = result.extraction

    narratives: dict[str, str] = {
        "ideal": (
            "Cenário de caminho feliz. " "Espera-se extração perfeita sem inferências."
        ),
        "unidade_omitida": (
            "O LLM deve inferir a unidade canônica. "
            f"Unidade obtida: `{ext.unit if ext else 'N/A'}`."
        ),
        "ambiguidade_terminologica": (
            "Cenário de sigla ambígua ('PA'). O LLM deve mapear corretamente "
            "para pressao_arterial e pedir clarificação (12 por 8)."
        ),
        "fora_do_padrao_limites": (
            "Valor inválido intencionalmente (FiO2=200%). O Pydantic deve "
            f"bloquear. Status obtido: `{ext.status if ext else 'N/A'}`."
        ),
        "comando_incompleto": (
            "Frase interrompida. " "O LLM não deve alucinar parâmetros inexistentes."
        ),
        "ruido_simulado": (
            "Artefato de ruído inserido. O LLM deve ignorar tokens espúrios "
            "e extrair o valor corretamente."
        ),
    }
    return narratives.get(scenario, "")


def generate_report_ab(
    results_v1: list[CaseResult],
    results_v2: list[CaseResult],
    elapsed_total_s: float,
) -> str:
    """Gera o relatório Markdown completo com análise comparativa A/B.

    Args:
        results_v1: Resultados do extrator V1.
        results_v2: Resultados do extrator V2 (CoT).
        elapsed_total_s: Tempo total da avaliação em segundos.

    Returns:
        String com o conteúdo Markdown do relatório.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results_v1)

    wer_values = [r.metrics.wer for r in results_v1 if r.metrics]
    cer_values = [r.metrics.cer for r in results_v1 if r.metrics]
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0

    i1 = calculate_ml_metrics(results_v1, "intent")
    i2 = calculate_ml_metrics(results_v2, "intent")
    p1 = calculate_ml_metrics(results_v1, "parameter")
    p2 = calculate_ml_metrics(results_v2, "parameter")

    lines: list[str] = []
    a = lines.append

    a("# Evaluation Report — vlab-stt-llm Pipeline (A/B Testing)")
    a("")
    a(f"**Gerado em:** {ts}  ")
    a(f"**Tempo total de execução:** {elapsed_total_s:.1f}s  ")
    a(f"**Casos avaliados:** {total}  ")
    a(f"**Semaphore limit:** {_SEMAPHORE_LIMIT} req paralelas  ")
    a("")
    a("## Análise Comparativa A/B: V1 (Direct) vs V2 (Chain-of-Thought)")
    a("")
    a("### Metodologia")
    a("")
    a("| Aspecto | V1 — Direct Schema | V2 — Chain-of-Thought |")
    a("|---------|--------------------|-----------------------|")
    a(
        "| Técnica | Zero-Shot + `response_schema` nativo | "
        "CoT estruturado com `<reasoning>` |"
    )
    a(
        "| Vantagem principal | Velocidade e previsibilidade | "
        "Robustez em casos ambíguos |"
    )
    a("")
    a("### Métricas de Transcrição STT (Compartilhadas)")
    a("")
    a("| ID | Cenário | WER | CER | Cache STT |")
    a("|----|---------|:---:|:---:|:---------:|")

    for r in results_v1:
        gt = r.ground_truth
        wer = r.metrics.wer_pct if r.metrics else "N/A"
        cer = r.metrics.cer_pct if r.metrics else "N/A"
        cache = "✓" if r.stt_from_cache else "—"
        a(f"| {gt.id} | `{gt.scenario_type}` | {wer} | {cer} | {cache} |")

    a("")
    a(
        f"**WER médio:** `{avg_wer * 100:.1f}%` | "
        f"**CER médio:** `{avg_cer * 100:.1f}%`"
    )
    a("")
    a("### Avaliação de Extração (Precision, Recall, F1-Score)")
    a("")
    a(
        "As métricas abaixo avaliam a precisão dos modelos em extrair "
        "Entidades e Intenções em relação ao Gabarito (Ground Truth)."
    )
    a("")
    a(
        "| Entidade | V1 Precision | V1 Recall | **V1 F1-Score** "
        "| V2 Precision | V2 Recall | **V2 F1-Score** |"
    )
    a(
        "|----------|:----------:|:---------:|:-------------:"
        "|:----------:|:---------:|:-------------:|"
    )
    a(
        f"| **Intent** | {i1.precision_pct} | {i1.recall_pct} | "
        f"**{i1.f1_pct}** | {i2.precision_pct} | {i2.recall_pct} | "
        f"**{i2.f1_pct}** |"
    )
    a(
        f"| **Parameter** | {p1.precision_pct} | {p1.recall_pct} | "
        f"**{p1.f1_pct}** | {p2.precision_pct} | {p2.recall_pct} | "
        f"**{p2.f1_pct}** |"
    )
    a("")
    a("## Análise Detalhada por Caso (Comparativo)")
    a("")

    for r1, r2 in zip(results_v1, results_v2, strict=False):
        gt = r1.ground_truth
        ext1, ext2 = r1.extraction, r2.extraction

        def _val(ext: MedicalParameterExtraction | None, field: str) -> str:
            return str(getattr(ext, field)) if ext else "N/A"

        a(f"### {gt.id} — `{gt.scenario_type}`")
        a("")
        a(f"**Transcrição Obtida:** `{r1.stt_transcript}`")
        a("")
        a("| Campo | Esperado | Obtido (V1) | Match V1 " "| Obtido (V2) | Match V2 |")
        a("|-------|----------|-------------|----------" "|-------------|----------|")
        a(
            f"| intent | `{gt.expected_intent}` "
            f"| `{_val(ext1, 'intent')}` | {_STATUS_ICON[r1.intent_match]} "
            f"| `{_val(ext2, 'intent')}` | {_STATUS_ICON[r2.intent_match]} |"
        )
        a(
            f"| param | `{gt.expected_parameter}` "
            f"| `{_val(ext1, 'parameter')}` | {_STATUS_ICON[r1.parameter_match]} "
            f"| `{_val(ext2, 'parameter')}` | {_STATUS_ICON[r2.parameter_match]} |"
        )
        a(
            f"| status | `{gt.expected_status}` "
            f"| `{_val(ext1, 'status')}` | {_STATUS_ICON[r1.status_match]} "
            f"| `{_val(ext2, 'status')}` | {_STATUS_ICON[r2.status_match]} |"
        )
        a("")

        narrative = _narrative_for(r1)
        if narrative:
            a(f"> **Análise:** {narrative}")
            a("")

        if r1.error_message:
            a(f"> ⚠️ **Erro V1:** `{r1.error_message}`")
        if r2.error_message:
            a(f"> ⚠️ **Erro V2:** `{r2.error_message}`")
        if r1.error_message or r2.error_message:
            a("")

    a("### Conclusão Comparativa")
    a(
        "A abordagem **V1** é recomendada por menor latência para produção "
        "direta. A **V2** traz ganhos interpretativos para ambientes de testes "
        "e homologação rigorosa de hardware médico."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orquestrador com Semaphore
# ---------------------------------------------------------------------------


async def main() -> None:
    """Orquestra a avaliação A/B com controle de concorrência via Semaphore.

    Cria um asyncio.Semaphore(_SEMAPHORE_LIMIT) compartilhado entre todas
    as coroutines de run_case. Isso garante que no máximo _SEMAPHORE_LIMIT
    chamadas à API Gemini estejam ativas ao mesmo tempo, evitando 429s
    causados por concorrência agressiva.
    """
    console.print(
        Panel.fit(
            "[bold cyan]vlab-stt-llm — Pipeline Evaluation (A/B)[/bold cyan]\n"
            f"[dim]Semaphore limit: {_SEMAPHORE_LIMIT} req paralelas[/dim]\n"
            + (
                "[dim yellow]Modo: CACHE (API ignorada onde disponível)[/dim yellow]"
                if USE_CACHE
                else "[bold red]Modo: LIVE (consumindo API do Google)[/bold red]"
            ),
            border_style="cyan",
        )
    )

    ground_truth_cases = load_ground_truth(GROUND_TRUTH_PATH)
    logger.info("%d casos carregados.", len(ground_truth_cases))

    # --- WIRING DO GERENCIADOR ---
    # Aqui instanciamos o GeminiManager UMA ÚNICA VEZ e o injetamos
    # nos três serviços, centralizando cota e roteamento de modelos.
    manager = GeminiManager()

    stt = GeminiSTT(manager=manager)
    extractor_v1 = ParameterExtractor(manager=manager)
    extractor_v2 = ParameterExtractorV2(manager=manager)

    # Semaphore compartilhado: limita concorrência global de chamadas à API
    semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)

    t_start_total = time.monotonic()
    total_calls = len(ground_truth_cases) * 2

    # Cria todas as tasks de uma vez — o Semaphore controla a concorrência
    tasks_v1 = [
        run_case(gt, stt, extractor_v1, "v1", semaphore) for gt in ground_truth_cases
    ]
    tasks_v2 = [
        run_case(gt, stt, extractor_v2, "v2", semaphore) for gt in ground_truth_cases
    ]
    all_tasks = tasks_v1 + tasks_v2

    results_v1: list[CaseResult] = []
    results_v2: list[CaseResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_bar = progress.add_task(
            "[cyan]Avaliando pipeline A/B...", total=total_calls
        )

        # gather com return_exceptions para não abortar em caso de falha pontual
        raw_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        progress.update(task_bar, completed=total_calls)

    # Separa resultados V1 e V2; substitui exceções por CaseResult de erro
    n = len(ground_truth_cases)
    for i, raw in enumerate(raw_results):
        gt = ground_truth_cases[i % n]
        version = "v1" if i < n else "v2"

        if isinstance(raw, Exception):
            logger.error("[%s][%s] Falha não tratada: %s", gt.id, version, raw)
            fallback = CaseResult(
                ground_truth=gt,
                extractor_version=version,
                error_message=f"Exceção não tratada: {raw}",
            )
            if version == "v1":
                results_v1.append(fallback)
            else:
                results_v2.append(fallback)
        else:
            if version == "v1":
                results_v1.append(raw)
            else:
                results_v2.append(raw)

    elapsed_total = time.monotonic() - t_start_total

    print_rich_summary_ab(results_v1, results_v2)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = generate_report_ab(results_v1, results_v2, elapsed_total)
    REPORT_PATH.write_text(report_content, encoding="utf-8")

    passed_v1 = sum(1 for r in results_v1 if r.overall_pass)
    passed_v2 = sum(1 for r in results_v2 if r.overall_pass)
    total = len(ground_truth_cases)

    console.print(
        Panel.fit(
            f"[bold]Relatório salvo em:[/bold] [yellow]{REPORT_PATH}[/yellow]\n"
            f"[bold]V1 (Direct):[/bold] "
            f"[{'green' if passed_v1 == total else 'red'}]"
            f"{passed_v1}/{total}[/]\n"
            f"[bold]V2 (CoT):   [/bold] "
            f"[{'green' if passed_v2 == total else 'red'}]"
            f"{passed_v2}/{total}[/]",
            border_style=(
                "green" if passed_v1 == total and passed_v2 == total else "yellow"
            ),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
