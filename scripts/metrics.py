# scripts/metrics.py
"""Cálculo de métricas de qualidade de transcrição STT.

Expõe WER (Word Error Rate) e CER (Character Error Rate) via biblioteca jiwer,
com normalização de texto adequada para português clínico.
"""

from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, process_words
from jiwer.transforms import (
    Compose,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
)

# Pipeline de normalização aplicado antes do cálculo das métricas.
# Garante comparações justas independente de capitalização e pontuação.
_NORMALIZER = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ]
)


@dataclass(frozen=True)
class TranscriptionMetrics:
    """Métricas de qualidade de transcrição para um par referência/hipótese.

    Attributes:
        wer: Word Error Rate [0.0–1.0+]. Valores > 1.0 são possíveis quando
            há muitas inserções.
        cer: Character Error Rate [0.0–1.0+]. Métrica mais granular que o WER.
        wer_pct: WER formatado como percentual (ex: "12.5%").
        cer_pct: CER formatado como percentual (ex: "8.3%").
    """

    wer: float
    cer: float

    @property
    def wer_pct(self) -> str:
        """WER formatado como string percentual com uma casa decimal."""
        return f"{self.wer * 100:.1f}%"

    @property
    def cer_pct(self) -> str:
        """CER formatado como string percentual com uma casa decimal."""
        return f"{self.cer * 100:.1f}%"


def compute(reference: str, hypothesis: str) -> TranscriptionMetrics:
    """Calcula WER e CER entre uma transcrição de referência e uma hipótese.

    Ambas as strings passam pelo pipeline de normalização antes do cálculo,
    tornando a comparação agnóstica a capitalização e pontuação.

    Args:
        reference: Transcrição esperada (ground truth).
        hypothesis: Transcrição produzida pelo STT.

    Returns:
        TranscriptionMetrics com WER e CER calculados. Retorna métricas com
        valor 0.0 se a hipótese estiver vazia e a referência também; retorna
        1.0 de WER se a hipótese for vazia mas a referência não.
    """
    ref_norm = _NORMALIZER([reference])[0] if reference else ""
    hyp_norm = _NORMALIZER([hypothesis])[0] if hypothesis else ""

    if not ref_norm and not hyp_norm:
        return TranscriptionMetrics(wer=0.0, cer=0.0)

    if not hyp_norm:
        return TranscriptionMetrics(wer=1.0, cer=1.0)

    output = process_words(ref_norm, hyp_norm)
    wer_value = output.wer

    cer_value = cer(ref_norm, hyp_norm)

    return TranscriptionMetrics(wer=wer_value, cer=cer_value)
