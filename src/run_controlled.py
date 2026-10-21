from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from rare_class_consensus import aggregate_prevalence_mask, aggregate_rare_class_mask, client_summary


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controlled"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [104729, 130363, 155921, 196613, 216091, 262147, 300007, 350377, 399989, 448451,
         500009, 550007, 600011, 650011, 700001, 750019, 800011, 850009, 900001, 950009]
CLASS_NAMES = ["C0-major", "C1-major", "C2-mid", "C3-rare", "C4-rarest"]
SIGNAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def make_fixture(seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Severe-imbalance fixture with a known rare-only discriminative signal."""
    rng = np.random.default_rng(seed)
    feature_count = 24
    train_counts = np.asarray([2200, 1300, 500, 100, 25])
    means = np.zeros((5, feature_count))
    for c, (first, second) in enumerate(SIGNAL_PAIRS):
        means[c, first], means[c, second] = 3.0, 2.2
    # The rarest class resembles C3 unless its rare-only features 8–9 survive.
    means[4, 6:8] = means[3, 6:8]

    def draw(c: int, n: int) -> np.ndarray:
        z = rng.normal(size=(n, feature_count))
        z[:, 16:20] += rng.normal(size=(n, 1))
        return z + means[c]

    client_x = [[] for _ in range(5)]
    client_y = [[] for _ in range(5)]
    for c, n in enumerate(train_counts):
        x_c = draw(c, int(n))
        allocation = rng.dirichlet(np.full(5, 0.7))
        counts = rng.multinomial(int(n), allocation)
        cursor = 0
        for k, count in enumerate(counts):
            if count:
                client_x[k].append(x_c[cursor:cursor + count])
                client_y[k].append(np.full(count, c, dtype=int))
            cursor += count
    clients = []
    for xs, ys in zip(client_x, client_y):
        clients.append((np.vstack(xs), np.concatenate(ys)))

    test_x = np.vstack([draw(c, 400) for c in range(5)])
    test_y = np.repeat(np.arange(5), 400)
    return clients, test_x, test_y


def nearest_centroid(clients, test_x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    train_x = np.vstack([x for x, _ in clients])
    train_y = np.concatenate([y for _, y in clients])
    scaler = StandardScaler().fit(train_x[:, mask])
    x_train = scaler.transform(train_x[:, mask])
    x_test = scaler.transform(test_x[:, mask])
    centroids = np.asarray([x_train[train_y == c].mean(axis=0) for c in range(5)])
    distance = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distance.argmin(axis=1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())


def evaluate(seed: int) -> dict:
    clients, test_x, test_y = make_fixture(seed)
    summaries = [client_summary(x, y, classes=5, candidate_budget=8, bootstraps=12,
                                seed=seed + 1009 * k) for k, (x, y) in enumerate(clients)]
    baseline = aggregate_prevalence_mask(summaries, feature_budget=8)
    proposed, diagnostics = aggregate_rare_class_mask(summaries, feature_budget=8)
    result = {"seed": seed, "masks": {}, "metrics": {}}
    for name, mask in (("prevalence_consensus", baseline), ("rasc_fcm", proposed)):
        pred = nearest_centroid(clients, test_x, mask)
        recalls = recall_score(test_y, pred, labels=np.arange(5), average=None, zero_division=0)
        result["masks"][name] = np.flatnonzero(mask).tolist()
        result["metrics"][name] = {
            "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
            "rarest_recall": float(recalls[4]),
            "macro_recall": float(recalls.mean()),
            # Coverage is the run-level probability that at least one member of
            # the known redundant rare-signal pair survives the fixed budget.
            "rarest_signal_retention": float(mask[[8, 9]].any()),
            "rarest_signal_fraction": float(mask[[8, 9]].mean()),
            "all_signal_retention": float(mask[:10].mean()),
        }
    result["diagnostics"] = {
        "class_support": diagnostics["class_support"].tolist(),
        "class_weights": diagnostics["class_weights"].tolist(),
    }
    return result


def summarize(runs: list[dict]) -> dict:
    summary = {}
    for method in ("prevalence_consensus", "rasc_fcm"):
        summary[method] = {}
        for metric in runs[0]["metrics"][method]:
            values = np.asarray([r["metrics"][method][metric] for r in runs])
            summary[method][metric] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
        masks = [np.isin(np.arange(24), r["masks"][method]) for r in runs]
        pairs = [jaccard(a, b) for a, b in itertools.combinations(masks, 2)]
        summary[method]["mask_jaccard"] = {"mean": float(np.mean(pairs)), "sample_sd": float(np.std(pairs, ddof=1))}
        summary[method]["selection_frequency"] = np.mean(masks, axis=0).tolist()

    diff = {}
    for metric in ("macro_f1", "rarest_recall", "rarest_signal_retention"):
        values = np.asarray([r["metrics"]["rasc_fcm"][metric] - r["metrics"]["prevalence_consensus"][metric] for r in runs])
        diff[metric] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    criteria = {
        "rarest_signal_retention_at_least_0_80": summary["rasc_fcm"]["rarest_signal_retention"]["mean"] >= 0.80,
        "rarest_recall_gain_at_least_0_15": diff["rarest_recall"]["mean"] >= 0.15,
        "macro_f1_loss_not_below_minus_0_02": diff["macro_f1"]["mean"] >= -0.02,
        "mask_jaccard_at_least_0_70": summary["rasc_fcm"]["mask_jaccard"]["mean"] >= 0.70,
    }
    return {
        "status": "controlled_mechanism_gate_not_official_dataset_effectiveness",
        "fixture": {"classes": CLASS_NAMES, "features": 24, "train_class_counts": [2200, 1300, 500, 100, 25],
                    "balanced_test_per_class": 400, "clients": 5, "feature_budget": 8, "seeds": SEEDS,
                    "known_rarest_signal_features": [8, 9]},
        "methods": summary, "paired_difference_rasc_minus_prevalence": diff,
        "acceptance_criteria": criteria, "gate_passed": all(criteria.values()), "runs": runs,
    }


def write_outputs(record: dict) -> None:
    (OUT / "step5_controlled_gate.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (OUT / "step5_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "metric", "mean", "sample_sd"])
        for method in ("prevalence_consensus", "rasc_fcm"):
            for metric, values in record["methods"][method].items():
                if isinstance(values, dict) and "mean" in values:
                    writer.writerow([method, metric, values["mean"], values["sample_sd"]])


def figures(record: dict) -> None:
    labels = ["Rare-signal\nretention", "Rarest-class\nrecall", "Macro F1", "Mask Jaccard"]
    keys = ["rarest_signal_retention", "rarest_recall", "macro_f1", "mask_jaccard"]
    x = np.arange(4); width = .36
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    for offset, method, label, color in [(-width/2, "prevalence_consensus", "Prevalence consensus", "#7597b7"),
                                          (width/2, "rasc_fcm", "RASC-FCM", "#c46643")]:
        means = [record["methods"][method][k]["mean"] for k in keys]
        sds = [record["methods"][method][k]["sample_sd"] for k in keys]
        bars = ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.12), ylabel="Score",
           title="Controlled severe-imbalance mechanism gate (20 seeds)")
    ax.grid(axis="y", alpha=.22); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "figure_step5_gate_outcomes.png", dpi=240); plt.close(fig)

    f0 = record["methods"]["prevalence_consensus"]["selection_frequency"]
    f1 = record["methods"]["rasc_fcm"]["selection_frequency"]
    xx = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(xx, f0, "o-", label="Prevalence consensus", color="#7597b7")
    ax.plot(xx, f1, "s-", label="RASC-FCM", color="#c46643")
    ax.axvspan(7.5, 9.5, color="#d5a021", alpha=.18, label="Rarest-class signal")
    ax.set(xlabel="Synthetic feature index", ylabel="Selection frequency", ylim=(-.04, 1.06),
           title="Mask selection frequency across controlled seeds")
    ax.set_xticks(xx); ax.grid(alpha=.2); ax.legend(frameon=False, ncol=3); fig.tight_layout()
    fig.savefig(OUT / "figure_step5_selection_frequency.png", dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.5)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.5); ax.axis("off")
    boxes = [
        (0.35, 1.25, 2.1, 3.0, "Client-side\ntraining data only", "Bootstrap mask frequency\nClipped per-class effect\nClass support"),
        (3.05, 1.25, 2.1, 3.0, "Secure sum interface", "Aggregate numerators\nand denominators\nNo raw rows or local masks"),
        (5.75, 1.25, 2.1, 3.0, "RASC-FCM server", "Capped inverse-support weights\nPer-class reserved slots\nStable composite ranking"),
        (8.45, 1.25, 2.35, 3.0, "Shared fixed mask", "Same feature support\nSame input dimension\nCompatible model tensors"),
    ]
    colors = ["#e8f0f7", "#edf3ea", "#fff0e8", "#eee9f5"]
    for (x, y, w, h, title, body), color in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor=color, edgecolor="#34495e", linewidth=1.4))
        ax.text(x+w/2, y+h-.55, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x+w/2, y+1.25, body, ha="center", va="center", fontsize=9.4, linespacing=1.35)
    for x0, x1 in ((2.47, 3.02), (5.17, 5.72), (7.87, 8.42)):
        ax.add_patch(FancyArrowPatch((x0, 2.75), (x1, 2.75), arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.5, color="#34495e"))
    ax.text(5.6, .55, "Controlled gate validates rare-signal preservation; official-dataset confirmation remains required",
            ha="center", va="center", fontsize=10, color="#7f4f24", fontweight="bold")
    ax.set_title("Rare-Class-Aware Stable Consensus Federated Masking (RASC-FCM)", fontsize=14, fontweight="bold", pad=8)
    fig.tight_layout(); fig.savefig(OUT / "figure_step5_rasc_architecture.png", dpi=240); plt.close(fig)


if __name__ == "__main__":
    runs = [evaluate(seed) for seed in SEEDS]
    record = summarize(runs)
    write_outputs(record)
    print(json.dumps({"gate_passed": record["gate_passed"], "criteria": record["acceptance_criteria"],
                      "methods": {m: {k: v for k, v in record["methods"][m].items() if isinstance(v, dict)}
                                  for m in record["methods"]}}, indent=2))
