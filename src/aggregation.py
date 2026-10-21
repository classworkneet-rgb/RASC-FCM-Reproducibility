"""Fairness-bounded drift-calibrated aggregation for FCM-DCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class DriftConfig:
    update_clip_norm: float = 5.0
    norm_drift_scale: float = 0.25
    drift_cap: float = 4.0
    drift_decay: float = 1.0
    fairness_floor: float = 0.2
    sample_cap: int = 100_000
    server_learning_rate: float = 1.0
    epsilon: float = 1e-12

    def validate(self) -> None:
        if self.update_clip_norm <= 0 or self.drift_cap < 0 or self.drift_decay < 0:
            raise ValueError("clip norm must be positive and drift limits nonnegative")
        if self.norm_drift_scale < 0:
            raise ValueError("norm_drift_scale must be nonnegative")
        if not 0 < self.fairness_floor <= 1:
            raise ValueError("fairness_floor must satisfy 0 < gamma <= 1")
        if self.sample_cap <= 0 or self.server_learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("sample cap, learning rate, and epsilon must be positive")


@dataclass(frozen=True)
class ClientUpdate:
    delta: Array
    sample_count: int


@dataclass(frozen=True)
class ClientUpdateDiagnostic:
    clipped_norm: float
    angular_drift: float
    norm_drift: float
    combined_drift: float
    reliability: float
    unnormalized_weight: float


@dataclass(frozen=True)
class AggregationResult:
    new_parameters: Array
    global_delta: Array
    normalized_weights: Array
    diagnostics: tuple[ClientUpdateDiagnostic, ...]


def clip_update(delta: Array, max_norm: float, epsilon: float = 1e-12) -> Array:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("delta must be a finite parameter vector")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = float(np.linalg.norm(values))
    return values * min(1.0, max_norm / (norm + epsilon))


def drift_components(delta: Array, previous_direction: Array, epsilon: float = 1e-12) -> tuple[float, float]:
    update = np.asarray(delta, dtype=np.float64)
    direction = np.asarray(previous_direction, dtype=np.float64)
    if update.shape != direction.shape or update.ndim != 1:
        raise ValueError("delta and previous_direction must be same-length vectors")
    update_norm = float(np.linalg.norm(update))
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= epsilon or update_norm <= epsilon:
        # There is no defensible angular reference in the first/zero-update round.
        angular = 0.0
    else:
        cosine = float(np.dot(update, direction) / (update_norm * direction_norm + epsilon))
        angular = (1.0 - float(np.clip(cosine, -1.0, 1.0))) / 2.0
    norm_drift = abs(float(np.log((update_norm + epsilon) / (direction_norm + epsilon)))) if direction_norm > epsilon else 0.0
    return angular, norm_drift


def reliability_from_drift(drift: float, decay: float, fairness_floor: float) -> float:
    if drift < 0 or decay < 0 or not 0 < fairness_floor <= 1:
        raise ValueError("invalid drift reliability parameters")
    return float(fairness_floor + (1.0 - fairness_floor) * np.exp(-decay * drift))


def aggregate_updates(
    current_parameters: Array,
    updates: Sequence[ClientUpdate],
    previous_direction: Array,
    config: DriftConfig,
) -> AggregationResult:
    """Apply the manuscript's clipped, fairness-bounded FCM-DCA rule."""

    config.validate()
    current = np.asarray(current_parameters, dtype=np.float64)
    previous = np.asarray(previous_direction, dtype=np.float64)
    if current.ndim != 1 or previous.shape != current.shape:
        raise ValueError("global parameter and direction vectors must have identical shape")
    if not updates:
        raise ValueError("at least one client update is required")

    clipped_updates: list[Array] = []
    weights: list[float] = []
    diagnostics: list[ClientUpdateDiagnostic] = []
    for update in updates:
        raw = np.asarray(update.delta, dtype=np.float64)
        if raw.shape != current.shape or update.sample_count <= 0:
            raise ValueError("client update is incompatible or has invalid sample count")
        clipped = clip_update(raw, config.update_clip_norm, config.epsilon)
        angular, norm_drift = drift_components(clipped, previous, config.epsilon)
        drift = float(np.clip(angular + config.norm_drift_scale * norm_drift, 0.0, config.drift_cap))
        reliability = reliability_from_drift(drift, config.drift_decay, config.fairness_floor)
        weight = float(min(update.sample_count, config.sample_cap) * reliability)
        clipped_updates.append(clipped)
        weights.append(weight)
        diagnostics.append(
            ClientUpdateDiagnostic(
                clipped_norm=float(np.linalg.norm(clipped)),
                angular_drift=angular,
                norm_drift=norm_drift,
                combined_drift=drift,
                reliability=reliability,
                unnormalized_weight=weight,
            )
        )
    weight_vector = np.asarray(weights, dtype=np.float64)
    # Every valid client has a positive sample count and r_k >= gamma > 0, so
    # the denominator is strictly positive.  Dividing by the exact sum keeps
    # Proposition 2 literal: normalized weights sum to one.
    normalized = weight_vector / weight_vector.sum()
    global_delta = np.sum(np.stack(clipped_updates) * normalized[:, None], axis=0)
    new_parameters = current + config.server_learning_rate * global_delta
    return AggregationResult(
        new_parameters=new_parameters,
        global_delta=global_delta,
        normalized_weights=normalized,
        diagnostics=tuple(diagnostics),
    )


def sample_weighted_fedavg_delta(updates: Sequence[ClientUpdate]) -> Array:
    if not updates:
        raise ValueError("at least one client update is required")
    counts = np.asarray([u.sample_count for u in updates], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("sample counts must be positive")
    stacked = np.stack([np.asarray(u.delta, dtype=np.float64) for u in updates])
    return np.sum(stacked * (counts / counts.sum())[:, None], axis=0)
