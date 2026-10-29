"""A compact NumPy reference implementation of the fixed gated DBN-CNN.

This implementation is intentionally dependency-light.  It supplies the exact
fixed topology, common input gate, two representation paths, learnable fusion
gate, class-weighted loss, deterministic parameter serialization, and local
supervised training needed to validate the federated method.  A later benchmark
step may port the same architecture to an accelerated framework without
changing shapes or equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax(x: Array) -> Array:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class ArchitectureSpec:
    feature_count: int
    class_count: int
    dbn_hidden: int = 24
    fusion_width: int = 12
    conv_filters: int = 8
    conv_kernel: int = 3

    def validate(self) -> None:
        values = (self.feature_count, self.class_count, self.dbn_hidden, self.fusion_width, self.conv_filters, self.conv_kernel)
        if any(v <= 0 for v in values):
            raise ValueError("all architecture dimensions must be positive")
        if self.class_count < 2 or self.conv_kernel > self.feature_count:
            raise ValueError("class_count must be >=2 and kernel cannot exceed feature count")


class FixedGatedDBNCNN:
    """Fixed-shape gated dual-path classifier with manual NumPy gradients."""

    PARAMETER_ORDER = (
        "W_dbn1", "b_dbn1", "W_dbn2", "b_dbn2", "K_conv", "b_conv",
        "W_cnn", "b_cnn", "W_gate", "b_gate", "W_out", "b_out",
    )

    def __init__(self, spec: ArchitectureSpec, *, seed: int = 0) -> None:
        spec.validate()
        self.spec = spec
        rng = np.random.default_rng(seed)
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)
        conv_positions = spec.feature_count - spec.conv_kernel + 1
        self.params: dict[str, Array] = {
            "W_dbn1": rng.normal(0, scale(spec.feature_count), (spec.feature_count, spec.dbn_hidden)),
            "b_dbn1": np.zeros(spec.dbn_hidden),
            "W_dbn2": rng.normal(0, scale(spec.dbn_hidden), (spec.dbn_hidden, spec.fusion_width)),
            "b_dbn2": np.zeros(spec.fusion_width),
            "K_conv": rng.normal(0, scale(spec.conv_kernel), (spec.conv_filters, spec.conv_kernel)),
            "b_conv": np.zeros(spec.conv_filters),
            "W_cnn": rng.normal(0, scale(spec.conv_filters), (spec.conv_filters, spec.fusion_width)),
            "b_cnn": np.zeros(spec.fusion_width),
            "W_gate": rng.normal(0, scale(2 * spec.fusion_width), (2 * spec.fusion_width, spec.fusion_width)),
            "b_gate": np.zeros(spec.fusion_width),
            "W_out": rng.normal(0, scale(spec.fusion_width), (spec.fusion_width, spec.class_count)),
            "b_out": np.zeros(spec.class_count),
        }
        self._conv_positions = conv_positions

    def topology_signature(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((name, self.params[name].shape) for name in self.PARAMETER_ORDER)

    def parameter_count(self) -> int:
        return int(sum(self.params[name].size for name in self.PARAMETER_ORDER))

    def parameter_vector(self) -> Array:
        return np.concatenate([self.params[name].ravel() for name in self.PARAMETER_ORDER])

    def load_parameter_vector(self, vector: Array) -> None:
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size != self.parameter_count():
            raise ValueError("parameter vector has an incompatible length")
        offset = 0
        for name in self.PARAMETER_ORDER:
            size = self.params[name].size
            self.params[name] = values[offset: offset + size].reshape(self.params[name].shape).copy()
            offset += size

    def _masked(self, x: Array, mask: Array) -> Array:
        features = np.asarray(x, dtype=np.float64)
        gate = np.asarray(mask, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.spec.feature_count:
            raise ValueError("x does not match the canonical feature schema")
        if gate.shape != (self.spec.feature_count,) or not np.all((gate == 0) | (gate == 1)):
            raise ValueError("all clients must use one compatible binary mask")
        return features * gate[None, :]

    def _forward(self, x: Array, mask: Array) -> tuple[Array, dict[str, Array]]:
        z = self._masked(x, mask)
        a1 = np.tanh(z @ self.params["W_dbn1"] + self.params["b_dbn1"])
        h_d = np.tanh(a1 @ self.params["W_dbn2"] + self.params["b_dbn2"])

        windows = np.lib.stride_tricks.sliding_window_view(z, self.spec.conv_kernel, axis=1)
        conv_pre = np.einsum("npk,fk->npf", windows, self.params["K_conv"]) + self.params["b_conv"]
        conv = np.maximum(conv_pre, 0.0)
        argmax = conv.argmax(axis=1)
        pooled = conv.max(axis=1)
        h_c = np.tanh(pooled @ self.params["W_cnn"] + self.params["b_cnn"])

        joined = np.concatenate([h_d, h_c], axis=1)
        fusion_gate = _sigmoid(joined @ self.params["W_gate"] + self.params["b_gate"])
        fused = fusion_gate * h_d + (1.0 - fusion_gate) * h_c
        logits = fused @ self.params["W_out"] + self.params["b_out"]
        probabilities = _softmax(logits)
        cache = {
            "z": z, "a1": a1, "h_d": h_d, "windows": windows, "conv_pre": conv_pre,
            "argmax": argmax, "pooled": pooled, "h_c": h_c, "joined": joined,
            "gate": fusion_gate, "fused": fused, "probabilities": probabilities,
        }
        return probabilities, cache

    def predict_proba(self, x: Array, mask: Array) -> Array:
        return self._forward(x, mask)[0]

    def predict(self, x: Array, mask: Array) -> Array:
        return self.predict_proba(x, mask).argmax(axis=1)

    def loss(self, x: Array, y: Array, mask: Array, *, class_weights: Array | None = None, weight_decay: float = 0.0) -> float:
        labels = np.asarray(y, dtype=np.int64)
        probabilities = self.predict_proba(x, mask)
        if labels.shape != (probabilities.shape[0],) or np.any((labels < 0) | (labels >= self.spec.class_count)):
            raise ValueError("y contains incompatible class labels")
        weights = np.ones(self.spec.class_count) if class_weights is None else np.asarray(class_weights, dtype=np.float64)
        if weights.shape != (self.spec.class_count,) or np.any(weights <= 0):
            raise ValueError("class_weights must be a positive class_count vector")
        ce = -np.mean(weights[labels] * np.log(probabilities[np.arange(labels.size), labels] + 1e-12))
        l2 = weight_decay * sum(float(np.sum(self.params[n] ** 2)) for n in self.PARAMETER_ORDER if n.startswith(("W_", "K_")))
        return float(ce + l2)

    def train_batch(
        self,
        x: Array,
        y: Array,
        mask: Array,
        *,
        learning_rate: float,
        class_weights: Array | None = None,
        weight_decay: float = 0.0,
    ) -> float:
        """Run one full supervised update through both paths and the fusion gate."""

        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        labels = np.asarray(y, dtype=np.int64)
        probabilities, c = self._forward(x, mask)
        n = labels.size
        weights = np.ones(self.spec.class_count) if class_weights is None else np.asarray(class_weights, dtype=np.float64)
        if labels.shape != (probabilities.shape[0],) or weights.shape != (self.spec.class_count,):
            raise ValueError("incompatible labels or class weights")
        sample_weights = weights[labels]
        dlogits = probabilities.copy()
        dlogits[np.arange(n), labels] -= 1.0
        dlogits *= sample_weights[:, None] / n

        gradients: dict[str, Array] = {}
        gradients["W_out"] = c["fused"].T @ dlogits + 2 * weight_decay * self.params["W_out"]
        gradients["b_out"] = dlogits.sum(axis=0)
        dfused = dlogits @ self.params["W_out"].T
        dgate = dfused * (c["h_d"] - c["h_c"])
        dh_d = dfused * c["gate"]
        dh_c = dfused * (1.0 - c["gate"])
        dgate_pre = dgate * c["gate"] * (1.0 - c["gate"])
        gradients["W_gate"] = c["joined"].T @ dgate_pre + 2 * weight_decay * self.params["W_gate"]
        gradients["b_gate"] = dgate_pre.sum(axis=0)
        djoined = dgate_pre @ self.params["W_gate"].T
        dh_d += djoined[:, : self.spec.fusion_width]
        dh_c += djoined[:, self.spec.fusion_width :]

        dh_d_pre = dh_d * (1.0 - c["h_d"] ** 2)
        gradients["W_dbn2"] = c["a1"].T @ dh_d_pre + 2 * weight_decay * self.params["W_dbn2"]
        gradients["b_dbn2"] = dh_d_pre.sum(axis=0)
        da1 = dh_d_pre @ self.params["W_dbn2"].T
        da1_pre = da1 * (1.0 - c["a1"] ** 2)
        gradients["W_dbn1"] = c["z"].T @ da1_pre + 2 * weight_decay * self.params["W_dbn1"]
        gradients["b_dbn1"] = da1_pre.sum(axis=0)

        dh_c_pre = dh_c * (1.0 - c["h_c"] ** 2)
        gradients["W_cnn"] = c["pooled"].T @ dh_c_pre + 2 * weight_decay * self.params["W_cnn"]
        gradients["b_cnn"] = dh_c_pre.sum(axis=0)
        dpooled = dh_c_pre @ self.params["W_cnn"].T
        dconv = np.zeros_like(c["conv_pre"])
        rows = np.arange(n)[:, None]
        filters = np.arange(self.spec.conv_filters)[None, :]
        dconv[rows, c["argmax"], filters] = dpooled
        dconv *= c["conv_pre"] > 0.0
        gradients["K_conv"] = np.einsum("npf,npk->fk", dconv, c["windows"]) + 2 * weight_decay * self.params["K_conv"]
        gradients["b_conv"] = dconv.sum(axis=(0, 1))

        for name in self.PARAMETER_ORDER:
            self.params[name] -= learning_rate * gradients[name]
        return self.loss(x, labels, mask, class_weights=weights, weight_decay=weight_decay)

    def rbm_pretrain_first_layer(self, x: Array, mask: Array, *, epochs: int = 2, learning_rate: float = 0.01, seed: int = 0) -> None:
        """CD-1 initialization of the first DBN layer on training data only."""

        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        visible = self._masked(x, mask)
        # Bernoulli RBM pretraining requires [0,1] inputs; scaling is local and
        # used only for initialization, not as a replacement for preprocessing.
        low = visible.min(axis=0)
        span = visible.max(axis=0) - low
        v0 = (visible - low) / np.where(span > 1e-12, span, 1.0)
        rng = np.random.default_rng(seed)
        w = self.params["W_dbn1"].copy()
        hb = self.params["b_dbn1"].copy()
        vb = np.zeros(self.spec.feature_count)
        for _ in range(epochs):
            ph0 = _sigmoid(v0 @ w + hb)
            h0 = (rng.random(ph0.shape) < ph0).astype(np.float64)
            pv1 = _sigmoid(h0 @ w.T + vb)
            ph1 = _sigmoid(pv1 @ w + hb)
            w += learning_rate * ((v0.T @ ph0 - pv1.T @ ph1) / v0.shape[0])
            vb += learning_rate * (v0 - pv1).mean(axis=0)
            hb += learning_rate * (ph0 - ph1).mean(axis=0)
        self.params["W_dbn1"] = w
        self.params["b_dbn1"] = hb


def clipped_class_weights(labels: Array, class_count: int, *, lower: float = 0.25, upper: float = 4.0) -> Array:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or np.any((y < 0) | (y >= class_count)):
        raise ValueError("labels are incompatible with class_count")
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    inverse = y.size / (class_count * np.maximum(counts, 1.0))
    return np.clip(inverse, lower, upper)
