"""Rare-class-aware stable consensus feature selection.

This module implements rare-class-aware consensus feature selection.  It exchanges only clipped
feature-level sufficient statistics.  The in-process summation used here is an
executable reference for the values that a secure-aggregation transport would
sum; it is not itself a cryptographic protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RareClassSummary:
    stability_numerator: np.ndarray
    stability_denominator: float
    class_score_numerator: np.ndarray
    class_score_denominator: np.ndarray
    class_support: np.ndarray


def _standardized_class_scores(x: np.ndarray, y: np.ndarray, classes: int, clip: float) -> np.ndarray:
    """Return clipped one-versus-rest standardized mean differences."""
    scale = np.std(x, axis=0) + 1e-8
    result = np.zeros((classes, x.shape[1]), dtype=float)
    for c in range(classes):
        positive = y == c
        if positive.sum() < 2 or (~positive).sum() < 2:
            continue
        delta = np.abs(x[positive].mean(axis=0) - x[~positive].mean(axis=0)) / scale
        result[c] = np.clip(delta, 0.0, clip) / clip
    return result


def client_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    classes: int,
    candidate_budget: int,
    bootstraps: int,
    seed: int,
    score_clip: float = 4.0,
    support_cap: int = 128,
) -> RareClassSummary:
    """Build one clipped client summary from training-only records."""
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be aligned 2-D/1-D arrays")
    if not 0 < candidate_budget <= x.shape[1]:
        raise ValueError("candidate_budget must be within feature dimension")
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")

    rng = np.random.default_rng(seed)
    frequency = np.zeros(x.shape[1], dtype=float)
    for _ in range(bootstraps):
        indices = rng.integers(0, len(y), size=len(y))
        xb, yb = x[indices], y[indices]
        per_class = _standardized_class_scores(xb, yb, classes, score_clip)
        prevalence = np.asarray([(yb == c).mean() for c in range(classes)])
        conventional = prevalence @ per_class
        chosen = np.argsort(-conventional, kind="stable")[:candidate_budget]
        frequency[chosen] += 1.0

    stability = frequency / bootstraps
    per_class = _standardized_class_scores(x, y, classes, score_clip)
    support = np.asarray([(y == c).sum() for c in range(classes)], dtype=float)
    capped_support = np.minimum(support, float(support_cap))
    return RareClassSummary(
        stability_numerator=stability * min(len(y), support_cap * classes),
        stability_denominator=float(min(len(y), support_cap * classes)),
        class_score_numerator=per_class * capped_support[:, None],
        class_score_denominator=capped_support,
        class_support=support,
    )


def aggregate_rare_class_mask(
    summaries: list[RareClassSummary],
    *,
    feature_budget: int,
    reserve_per_class: int = 1,
    beta: float = 0.75,
    rare_weight_cap: float = 4.0,
    minimum_global_support: int = 8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate summaries and return one shape-compatible global mask.

    Supported classes first reserve ``reserve_per_class`` distinct features.
    Remaining slots are filled by the composite stability/discrimination score.
    """
    if not summaries:
        raise ValueError("at least one summary is required")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    feature_count = summaries[0].stability_numerator.size
    class_count = summaries[0].class_score_numerator.shape[0]
    if not 0 < feature_budget <= feature_count:
        raise ValueError("invalid feature budget")

    stability_num = sum((s.stability_numerator for s in summaries), np.zeros(feature_count))
    stability_den = sum(s.stability_denominator for s in summaries)
    class_num = sum((s.class_score_numerator for s in summaries), np.zeros((class_count, feature_count)))
    class_den = sum((s.class_score_denominator for s in summaries), np.zeros(class_count))
    support = sum((s.class_support for s in summaries), np.zeros(class_count))

    stability = stability_num / max(stability_den, 1e-12)
    class_scores = np.divide(class_num, class_den[:, None], out=np.zeros_like(class_num), where=class_den[:, None] > 0)
    supported = support >= minimum_global_support
    inverse = np.zeros(class_count, dtype=float)
    inverse[supported] = 1.0 / np.sqrt(np.maximum(support[supported], 1.0))
    if inverse.sum() > 0:
        positive = inverse[inverse > 0]
        inverse = np.minimum(inverse / positive.min(), rare_weight_cap)
        class_weights = inverse / inverse.sum()
    else:
        class_weights = np.ones(class_count) / class_count
    balanced = class_weights @ class_scores
    composite = (1.0 - beta) * stability + beta * balanced

    selected: list[int] = []
    for c in np.flatnonzero(supported):
        for j in np.argsort(-class_scores[c], kind="stable"):
            if int(j) not in selected:
                selected.append(int(j))
                if sum(1 for v in selected if v in np.argsort(-class_scores[c], kind="stable")[:reserve_per_class]) >= reserve_per_class:
                    break
        if len(selected) >= feature_budget:
            break
    for j in np.argsort(-composite, kind="stable"):
        if int(j) not in selected:
            selected.append(int(j))
        if len(selected) == feature_budget:
            break

    mask = np.zeros(feature_count, dtype=bool)
    mask[selected[:feature_budget]] = True
    diagnostics = {
        "stability": stability,
        "class_scores": class_scores,
        "class_support": support,
        "class_weights": class_weights,
        "composite": composite,
    }
    return mask, diagnostics


def aggregate_prevalence_mask(
    summaries: list[RareClassSummary], *, feature_budget: int
) -> np.ndarray:
    """Conventional prevalence-weighted comparator without class reservation."""
    class_num = sum((s.class_score_numerator for s in summaries), np.zeros_like(summaries[0].class_score_numerator))
    class_den = sum((s.class_score_denominator for s in summaries), np.zeros_like(summaries[0].class_score_denominator))
    scores = np.divide(class_num, class_den[:, None], out=np.zeros_like(class_num), where=class_den[:, None] > 0)
    support = sum((s.class_support for s in summaries), np.zeros_like(summaries[0].class_support))
    prevalence = support / support.sum()
    conventional = prevalence @ scores
    chosen = np.argsort(-conventional, kind="stable")[:feature_budget]
    mask = np.zeros(conventional.size, dtype=bool)
    mask[chosen] = True
    return mask
