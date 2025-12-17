"""Preprocess stage."""

from __future__ import annotations

from src.preprocess.strategies.blur import BlurStrategy
from src.preprocess.strategies.normal import NormalStrategy

BLUR_T = 100.0


def choose_preprocess_strategy(blur_score: float, normal: NormalStrategy, blur: BlurStrategy):
    if blur_score < BLUR_T:
        return blur
    return normal
