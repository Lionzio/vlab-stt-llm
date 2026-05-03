# scripts/evaluate_pipeline.py
"""Script de avaliação automatizada A/B do pipeline vlab-stt-llm.

Executa cada caso de teste contra V1 (Direct) e V2 (Chain-of-Thought),
utilizando cache local para contornar limites de cota da API.
Gera relatório com métricas WER/CER e narrativas detalhadas.
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

# --- Bypass para importar src.* mantendo PEP8 ---
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
from src.services.stt import GeminiSTT, STTError  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração de logging com Rich
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
# Modelos de dados
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
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
        return self.stt_success and self.schema_adherence and self.intent_match


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def load_ground_truth(path: Path) -> list[GroundTruth]:
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


async def run_case(
    gt: GroundTruth,
    stt: GeminiSTT,
    extractor: ParameterExtractor | ParameterExtractorV2,
    extractor_version: str = "v1",
) -> CaseResult:
    result = CaseResult(ground_truth=gt, extractor_version=extractor_version)
    audio_path = AUDIO_DIR / gt.audio_filename
    t_start = time.monotonic()

    try:
        if not audio_path.exists():
            result.error_message = f"Arquivo de áudio não encontrado: {audio_path}"
            return result

        # STT com Cache
        cached_transcript = get_stt(str(audio_path))
        if cached_transcript is not None:
            transcript = cached_transcript
            result.stt_from_cache = True
        else:
            transcript = await stt.transcribe(str(audio_path))
            if not transcript:
                result.error_message = "STT retornou transcrição vazia."
                return result
            set_stt(str(audio_path), transcript)

        result.stt_success = True
        result.stt_transcript = transcript

        # Métricas WER/CER
        result.metrics = compute_metrics(
            reference=gt.expected_transcription,
            hypothesis=transcript,
        )

        # Extração com Cache
        cached_llm = get_llm(transcript, extractor_version)
        if cached_llm is not None:
            extraction = MedicalParameterExtraction.model_validate(cached_llm)
            result.llm_from_cache = True
        else:
            extraction = await extractor.extract(transcript)
            if extraction is None:
                result.error_message = "Extractor retornou None."
                return result
            set_llm(transcript, extraction.model_dump(), extractor_version)

        result.schema_adherence = True
        result.extraction = extraction

        # Avaliação
        result.intent_match = extraction.intent == gt.expected_intent
        if gt.expected_parameter is None:
            result.parameter_match = extraction.parameter is None
        else:
            result.parameter_match = (
                extraction.parameter or ""
            ).lower() == gt.expected_parameter.lower()
        result.status_match = extraction.status == gt.expected_status

    except STTError as exc:
        result.error_message = f"STTError: {exc}"
    except Exception as exc:  # noqa: BLE001
        result.schema_adherence = False
        result.error_message = f"Extractor falhou: {exc}"
    finally:
        result.latency_s = time.monotonic() - t_start

    return result


# ---------------------------------------------------------------------------
# Geração de Relatório A/B
# ---------------------------------------------------------------------------
_STATUS_ICON = {True: "✅", False: "❌"}
_MATCH_ICON = {True: "✓", False: "✗", None: "—"}


def print_rich_summary_ab(
    results_v1: list[CaseResult], results_v2: list[CaseResult]
) -> None:
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
        cache_label = "✓" if r1.stt_from_cache and r1.llm_from_cache else "—"

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
    console.print()


def _narrative_for(result: CaseResult) -> str:
    scenario = result.ground_truth.scenario_type
    ext = result.extraction
    narratives: dict[str, str] = {
        "ideal": "Cenário de caminho feliz. Espera-se extração perfeita sem inferências.",
        "unidade_omitida": f"O LLM deve inferir a unidade canônica. Unidade obtida: `{ext.unit if ext else 'N/A'}`.",
        "ambiguidade_terminologica": "Cenário de sigla ambígua ('PA'). O LLM deve mapear corretamente para pressao_arterial e pedir clarificação (12 por 8).",
        "fora_do_padrao_limites": f"Valor inválido intencionalmente (FiO2=200%). O Pydantic deve bloquear. Status obtido: `{ext.status if ext else 'N/A'}`.",
        "comando_incompleto": "Frase interrompida. O LLM não deve alucinar parâmetros inexistentes.",
        "ruido_simulado": "Artefato de ruído inserido. O LLM deve ignorar tokens espúrios e extrair o valor corretamente.",
    }
    return narratives.get(scenario, "")


def generate_report_ab(
    results_v1: list[CaseResult], results_v2: list[CaseResult], elapsed_total_s: float
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results_v1)
    passed_v1 = sum(1 for r in results_v1 if r.overall_pass)
    passed_v2 = sum(1 for r in results_v2 if r.overall_pass)

    wer_values = [r.metrics.wer for r in results_v1 if r.metrics]
    cer_values = [r.metrics.cer for r in results_v1 if r.metrics]
    avg_wer = sum(wer_values) / len(wer_values) if wer_values else 0.0
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0

    lines = []
    a = lines.append
    a("# Evaluation Report — vlab-stt-llm Pipeline (A/B Testing)")
    a("")
    a(f"**Gerado em:** {ts}  ")
    a(f"**Tempo total de execução:** {elapsed_total_s:.1f}s  ")
    a(f"**Casos avaliados:** {total}  ")
    a("")
    a("## Análise Comparativa A/B: V1 (Direct) vs V2 (Chain-of-Thought)")
    a("")
    a("### Metodologia")
    a("| Aspecto | V1 — Direct Schema | V2 — Chain-of-Thought |")
    a("|---------|--------------------|-----------------------|")
    a(
        "| Técnica | Zero-Shot + `response_schema` nativo | CoT estruturado com `<reasoning>` |"
    )
    a(
        "| Vantagem principal | Velocidade e previsibilidade | Robustez em casos ambíguos |"
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
    a(f"**WER médio:** `{avg_wer * 100:.1f}%` | **CER médio:** `{avg_cer * 100:.1f}%`")
    a("")

    a("## Análise Detalhada por Caso (Comparativo)")
    a("")
    for r1, r2 in zip(results_v1, results_v2, strict=False):
        gt = r1.ground_truth
        a(f"### {gt.id} — `{gt.scenario_type}`")
        a("")
        a(f"**Transcrição Obtida:** `{r1.stt_transcript}`")
        a("")
        a("| Campo | Esperado | Obtido (V1) | Match V1 | Obtido (V2) | Match V2 |")
        a("|-------|----------|-------------|----------|-------------|----------|")

        ext1, ext2 = r1.extraction, r2.extraction

        def _val(ext, field):  # type: ignore
            return getattr(ext, field) if ext else "N/A"

        a(
            f"| intent | `{gt.expected_intent}` | `{_val(ext1, 'intent')}` | {_STATUS_ICON[r1.intent_match]} | `{_val(ext2, 'intent')}` | {_STATUS_ICON[r2.intent_match]} |"
        )
        a(
            f"| param  | `{gt.expected_parameter}`| `{_val(ext1, 'parameter')}`| {_STATUS_ICON[r1.parameter_match]} | `{_val(ext2, 'parameter')}`| {_STATUS_ICON[r2.parameter_match]} |"
        )
        a(
            f"| status | `{gt.expected_status}`   | `{_val(ext1, 'status')}`   | {_STATUS_ICON[r1.status_match]} | `{_val(ext2, 'status')}`   | {_STATUS_ICON[r2.status_match]} |"
        )
        a("")
        narrative = _narrative_for(r1)
        if narrative:
            a(f"> **Análise:** {narrative}")
            a("")

        if r1.error_message or r2.error_message:
            if r1.error_message:
                a(f"> ⚠️ **Erro V1:** `{r1.error_message}`")
            if r2.error_message:
                a(f"> ⚠️ **Erro V2:** `{r2.error_message}`")
            a("")

    a("### Conclusão Comparativa")
    a(
        "A abordagem **V1** é recomendada por menor latência para produção direta. A **V2** traz ganhos interpretativos para ambientes de testes e homologação rigorosa de hardware médico."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------
async def main() -> None:
    console.print(
        Panel.fit("[bold cyan]vlab-stt-llm — Pipeline Evaluation (A/B)[/bold cyan]")
    )
    ground_truth_cases = load_ground_truth(GROUND_TRUTH_PATH)

    stt = GeminiSTT()
    extractor_v1 = ParameterExtractor()
    extractor_v2 = ParameterExtractorV2()

    results_v1: list[CaseResult] = []
    results_v2: list[CaseResult] = []
    t_start_total = time.monotonic()

    total_calls = len(ground_truth_cases) * 2

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Avaliando pipeline A/B...", total=total_calls)

        for idx, gt in enumerate(ground_truth_cases):
            # V1
            progress.update(task, description=f"[cyan]{gt.id} [V1]...")
            r_v1 = await run_case(gt, stt, extractor_v1, "v1")
            results_v1.append(r_v1)
            progress.advance(task)

            # Rate limit mitigation only if NOT hitting cache
            if (
                not r_v1.stt_from_cache
                and not r_v1.llm_from_cache
                and idx < len(ground_truth_cases) - 1
            ):
                await asyncio.sleep(5)

            # V2
            progress.update(task, description=f"[cyan]{gt.id} [V2-CoT]...")
            r_v2 = await run_case(gt, stt, extractor_v2, "v2")
            results_v2.append(r_v2)
            progress.advance(task)

            if not r_v2.llm_from_cache and idx < len(ground_truth_cases) - 1:
                await asyncio.sleep(5)

    elapsed_total = time.monotonic() - t_start_total
    print_rich_summary_ab(results_v1, results_v2)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = generate_report_ab(results_v1, results_v2, elapsed_total)
    REPORT_PATH.write_text(report_content, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
