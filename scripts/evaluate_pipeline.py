# scripts/evaluate_pipeline.py

"""Script de avaliação automatizada do pipeline vlab-stt-llm.

Executa cada caso de teste do ground_truth.json contra o pipeline completo
(GeminiSTT → ParameterExtractor), coleta métricas de desempenho e gera
um relatório de benchmarking em docs/evaluation_report.md.

Usage:
    python scripts/evaluate_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import logging

# ---------------------------------------------------------------------------
# Sys-path bootstrap — permite importar src.* sem instalar o pacote
# ---------------------------------------------------------------------------
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
sys.path.insert(0, str(REPO_ROOT))

from src.services.extractor import (  # noqa: E402
    MedicalParameterExtraction,
    ParameterExtractor,
)
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
# Modelos de dados internos
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
        stt_success: True se o STT produziu texto não-vazio.
        stt_transcript: Texto transcrito pelo STT (vazio em caso de falha).
        schema_adherence: True se o Extractor retornou objeto Pydantic válido.
        extraction: Objeto de extração retornado (None em caso de falha).
        intent_match: True se o intent extraído bate com o esperado.
        parameter_match: True se o parameter extraído bate com o esperado.
        status_match: True se o status de validação bate com o esperado.
        latency_s: Tempo total de execução do caso em segundos.
        error_message: Mensagem de erro capturada, se houver.
    """

    ground_truth: GroundTruth
    stt_success: bool = False
    stt_transcript: str = ""
    schema_adherence: bool = False
    extraction: MedicalParameterExtraction | None = None
    intent_match: bool = False
    parameter_match: bool = False
    status_match: bool = False
    latency_s: float = 0.0
    error_message: str = ""

    @property
    def overall_pass(self) -> bool:
        """Caso passa se STT, schema e intent estão todos corretos."""
        return self.stt_success and self.schema_adherence and self.intent_match


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
        KeyError: Se algum campo obrigatório estiver ausente no JSON.
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
# Execução de um caso de teste
# ---------------------------------------------------------------------------


async def run_case(
    gt: GroundTruth,
    stt: GeminiSTT,
    extractor: ParameterExtractor,
) -> CaseResult:
    """Executa o pipeline completo para um único caso de teste.

    Args:
        gt: Caso de teste com os dados de referência.
        stt: Instância do serviço de transcrição.
        extractor: Instância do extrator de parâmetros médicos.

    Returns:
        CaseResult preenchido com todas as métricas do caso.
    """
    result = CaseResult(ground_truth=gt)
    audio_path = AUDIO_DIR / gt.audio_filename
    t_start = time.monotonic()

    try:
        # ------------------------------------------------------------------
        # Etapa 1 — STT
        # ------------------------------------------------------------------
        if not audio_path.exists():
            result.error_message = f"Arquivo de áudio não encontrado: {audio_path}"
            logger.warning("[%s] %s", gt.id, result.error_message)
            return result

        logger.info("[%s] Transcrevendo %s...", gt.id, gt.audio_filename)
        transcript = await stt.transcribe(str(audio_path))

        if not transcript:
            result.error_message = "STT retornou transcrição vazia."
            logger.warning("[%s] %s", gt.id, result.error_message)
            return result

        result.stt_success = True
        result.stt_transcript = transcript

        # Log da diferença de transcrição
        if transcript.strip() != gt.expected_transcription.strip().lower():
            logger.info(
                "[%s] Divergência STT:\n  Esperado : %r\n  Obtido   : %r",
                gt.id,
                gt.expected_transcription,
                transcript,
            )
        else:
            logger.info("[%s] Transcrição idêntica ao esperado.", gt.id)

        # ------------------------------------------------------------------
        # Etapa 2 — Extração de parâmetros
        # ------------------------------------------------------------------
        logger.info("[%s] Extraindo parâmetros do texto transcrito...", gt.id)
        extraction = await extractor.extract(transcript)

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
            "[%s] intent=%s param=%s status=%s",
            gt.id,
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
# Geração do relatório Markdown
# ---------------------------------------------------------------------------

_STATUS_ICON = {True: "✅", False: "❌"}
_MATCH_ICON = {True: "✓", False: "✗", None: "—"}


def _match(value: bool | None) -> str:
    return _MATCH_ICON.get(value, "—")


def _icon(value: bool) -> str:
    return _STATUS_ICON[value]


def generate_report(results: list[CaseResult], elapsed_total_s: float) -> str:
    """Gera o conteúdo completo do relatório de avaliação em Markdown.

    Args:
        results: Lista de resultados de cada caso de teste.
        elapsed_total_s: Tempo total de execução da avaliação em segundos.

    Returns:
        String com o conteúdo Markdown do relatório.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    stt_ok = sum(1 for r in results if r.stt_success)
    schema_ok = sum(1 for r in results if r.schema_adherence)
    intent_ok = sum(1 for r in results if r.intent_match)
    param_ok = sum(1 for r in results if r.parameter_match)
    status_ok = sum(1 for r in results if r.status_match)

    lines: list[str] = []
    a = lines.append

    # Cabeçalho
    a("# Evaluation Report — vlab-stt-llm Pipeline")
    a("")
    a(f"**Gerado em:** {ts}  ")
    a(f"**Tempo total de execução:** {elapsed_total_s:.1f}s  ")
    a(f"**Casos avaliados:** {total}  ")
    a(f"**Taxa de aprovação geral:** {passed}/{total} ({100 * passed // total}%)  ")
    a("")
    a("---")
    a("")

    # Métricas resumidas
    a("## Métricas Resumidas")
    a("")
    a("| Métrica | Aprovados | Total | Taxa |")
    a("|---------|-----------|-------|------|")
    a(f"| STT bem-sucedido | {stt_ok} | {total} | {100 * stt_ok // total}% |")
    a(
        f"| Aderência ao Schema Pydantic | {schema_ok} | {total} | {100 * schema_ok // total}% |"
    )
    a(f"| Intent correto | {intent_ok} | {total} | {100 * intent_ok // total}% |")
    a(f"| Parameter correto | {param_ok} | {total} | {100 * param_ok // total}% |")
    a(
        f"| Status de validação correto | {status_ok} | {total} | {100 * status_ok // total}% |"
    )
    a("")
    a("---")
    a("")

    # Tabela de resultados por caso
    a("## Resultados por Caso de Teste")
    a("")
    a(
        "| ID | Cenário | STT | Schema | Intent | Parâmetro | Status Val. | Latência | Resultado |"
    )
    a(
        "|----|---------|-----|--------|--------|-----------|-------------|----------|-----------|"
    )

    for r in results:
        gt = r.ground_truth
        latency = f"{r.latency_s:.1f}s"
        overall = _icon(r.overall_pass)
        a(
            f"| {gt.id} "
            f"| `{gt.scenario_type}` "
            f"| {_icon(r.stt_success)} "
            f"| {_icon(r.schema_adherence)} "
            f"| {_match(r.intent_match)} "
            f"| {_match(r.parameter_match)} "
            f"| {_match(r.status_match)} "
            f"| {latency} "
            f"| {overall} |"
        )

    a("")
    a("---")
    a("")

    # Análise detalhada por caso
    a("## Análise Detalhada por Caso")
    a("")

    for r in results:
        gt = r.ground_truth
        a(f"### {gt.id} — `{gt.scenario_type}`")
        a("")
        a(
            f"**Resultado geral:** {_icon(r.overall_pass)} {'APROVADO' if r.overall_pass else 'REPROVADO'}  "
        )
        a(f"**Latência:** {r.latency_s:.2f}s  ")
        a("")

        a("**Transcrição:**")
        a("")
        a(f"- Esperada : `{gt.expected_transcription}`")
        a(f"- Obtida   : `{r.stt_transcript or '(vazia)'}`")
        a("")

        if r.extraction:
            ext = r.extraction
            a("**Extração:**")
            a("")
            a("| Campo | Esperado | Obtido | Match |")
            a("|-------|----------|--------|-------|")
            a(
                f"| intent | `{gt.expected_intent}` | `{ext.intent}` | {_match(r.intent_match)} |"
            )
            a(
                f"| parameter | `{gt.expected_parameter}` | `{ext.parameter}` | {_match(r.parameter_match)} |"
            )
            a(f"| value | `{gt.expected_value}` | `{ext.value}` | — |")
            a(f"| unit | `{gt.expected_unit}` | `{ext.unit}` | — |")
            a(
                f"| status | `{gt.expected_status}` | `{ext.status}` | {_match(r.status_match)} |"
            )
            a("")

        if r.error_message:
            a(f"> ⚠️ **Erro capturado:** `{r.error_message}`")
            a("")

        # Análise narrativa automática baseada no cenário
        narrative = _narrative_for(r)
        if narrative:
            a(f"**Análise:** {narrative}")
            a("")

        a("---")
        a("")

    # Conclusão
    a("## Conclusão")
    a("")
    if passed == total:
        a(
            "✅ **Todos os casos aprovados.** O pipeline demonstra robustez "
            "nos cenários cobertos pelo dataset de avaliação."
        )
    else:
        failed_ids = [r.ground_truth.id for r in results if not r.overall_pass]
        a(
            f"⚠️ **{total - passed} caso(s) reprovado(s):** {', '.join(failed_ids)}. "
            "Consulte a análise detalhada acima para identificar os pontos de falha."
        )
    a("")
    a(
        "> *Relatório gerado automaticamente por `scripts/evaluate_pipeline.py`. "
        "Revisão humana recomendada para casos com `status_match=✗`.*"
    )

    return "\n".join(lines)


def _narrative_for(result: CaseResult) -> str:
    """Gera uma análise narrativa contextualizada para o cenário do caso.

    Args:
        result: Resultado do caso de teste avaliado.

    Returns:
        String com a análise narrativa, ou string vazia se não aplicável.
    """
    scenario = result.ground_truth.scenario_type
    ext = result.extraction

    narratives: dict[str, str] = {
        "caso_ideal": (
            "Cenário de caminho feliz. Comando completo com todos os campos "
            "explícitos. Espera-se extração perfeita sem inferências."
        ),
        "unidade_omitida": (
            "O LLM deve inferir a unidade canônica a partir do mapeamento de domínio "
            "(parâmetro → unidade default). "
            + (
                f"Unidade obtida: `{ext.unit}` — "
                + (
                    "inferência correta."
                    if ext and ext.unit
                    else "inferência ausente ou incorreta."
                )
                if ext
                else "Extração não disponível."
            )
        ),
        "ambiguidade_fonetica_terminologica": (
            "Cenário de sigla homofônica ('PA'). O STT pode normalizar para "
            "'pressão arterial' ou manter a sigla. O LLM deve mapear corretamente "
            "para o parâmetro canônico independentemente da forma transcrita."
        ),
        "valor_fora_do_intervalo": (
            "Valor inválido intencionalmente injetado (FiO2=200%). "
            "A camada de validação Pydantic deve bloquear e retornar `out_of_range`. "
            + (f"Status obtido: `{ext.status}`." if ext else "Extração não disponível.")
        ),
        "comando_incompleto": (
            "Frase interrompida sem especificação de parâmetro ou modo. "
            "O LLM não deve alucinar — `value` e `parameter` devem ser nulos, "
            "`requires_human_confirmation` deve ser verdadeiro."
        ),
        "ruido_erro_transcricao": (
            "Artefato de ruído (tosse) inserido entre valor e unidade. "
            "O LLM deve ignorar tokens espúrios e extrair `value=600.0` e `unit='mL'` "
            "corretamente."
        ),
    }

    return narratives.get(scenario, "")


# ---------------------------------------------------------------------------
# Exibição da tabela Rich no terminal
# ---------------------------------------------------------------------------


def print_rich_summary(results: list[CaseResult]) -> None:
    """Exibe uma tabela resumo colorida no terminal usando Rich.

    Args:
        results: Lista de resultados de todos os casos avaliados.
    """
    table = Table(
        title="[bold cyan]Resumo da Avaliação — vlab-stt-llm[/bold cyan]",
        show_lines=True,
        header_style="bold magenta",
    )

    table.add_column("ID", style="bold", width=8)
    table.add_column("Cenário", style="cyan", max_width=30)
    table.add_column("STT", justify="center", width=5)
    table.add_column("Schema", justify="center", width=8)
    table.add_column("Intent", justify="center", width=8)
    table.add_column("Param.", justify="center", width=8)
    table.add_column("Status Val.", justify="center", width=11)
    table.add_column("Latência", justify="right", width=9)
    table.add_column("Resultado", justify="center", width=10)

    for r in results:
        gt = r.ground_truth
        row_style = "green" if r.overall_pass else "red"
        table.add_row(
            gt.id,
            gt.scenario_type,
            "✅" if r.stt_success else "❌",
            "✅" if r.schema_adherence else "❌",
            "✓" if r.intent_match else "✗",
            "✓" if r.parameter_match else "✗",
            "✓" if r.status_match else "✗",
            f"{r.latency_s:.1f}s",
            "[green]PASS[/green]" if r.overall_pass else "[red]FAIL[/red]",
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------


async def main() -> None:
    """Orquestra a avaliação completa do pipeline.

    Carrega o ground truth, inicializa os serviços, executa cada caso de
    teste com barra de progresso, exibe o resumo no terminal e persiste
    o relatório Markdown.
    """
    console.print(
        Panel.fit(
            "[bold cyan]vlab-stt-llm — Pipeline Evaluation[/bold cyan]\n"
            f"Ground truth: [yellow]{GROUND_TRUTH_PATH}[/yellow]\n"
            f"Áudios: [yellow]{AUDIO_DIR}[/yellow]",
            border_style="cyan",
        )
    )

    # Carregamento
    logger.info("Carregando ground truth...")
    ground_truth_cases = load_ground_truth(GROUND_TRUTH_PATH)
    logger.info("%d casos carregados.", len(ground_truth_cases))

    # Inicialização dos serviços (uma única instância por avaliação)
    stt = GeminiSTT()
    extractor = ParameterExtractor()

    results: list[CaseResult] = []
    t_start_total = time.monotonic()

    # Execução com barra de progresso Rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processando casos de teste...", total=len(ground_truth_cases)
        )

        for idx, gt in enumerate(ground_truth_cases):
            progress.update(task, description=f"[cyan]Processando {gt.id}...")
            result = await run_case(gt, stt, extractor)
            results.append(result)
            progress.advance(task)
            
            # Rate Limit Bypass (Free Tier) - Espera 5 segundos entre cada caso
            if idx < len(ground_truth_cases) - 1:
                logger.info("Aguardando 5s para evitar Rate Limit (429) do Gemini Free Tier...")
                await asyncio.sleep(5)

    elapsed_total = time.monotonic() - t_start_total

    # Exibição do resumo no terminal
    print_rich_summary(results)

    # Persistência do relatório Markdown
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = generate_report(results, elapsed_total)
    REPORT_PATH.write_text(report_content, encoding="utf-8")

    passed = sum(1 for r in results if r.overall_pass)
    total = len(results)

    console.print(
        Panel.fit(
            f"[bold]Relatório salvo em:[/bold] [yellow]{REPORT_PATH}[/yellow]\n"
            f"[bold]Resultado final:[/bold] "
            f"[{'green' if passed == total else 'red'}]{passed}/{total} aprovados[/]",
            border_style="green" if passed == total else "red",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
