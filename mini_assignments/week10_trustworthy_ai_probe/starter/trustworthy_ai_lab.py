"""Hands-on probes for three trustworthy-AI pillars: fairness, explainability, privacy.

CPU-only, no GPU, no API keys. Uses only numpy and scikit-learn (both preinstalled on
free-tier Colab).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------------
# Pillar 1: Fairness -- recompute a headline number from raw per-cell data.
# ---------------------------------------------------------------------------

def load_subgroup_csv(path_or_text: str) -> list[dict]:
    """Load a fairness subgroup CSV (condition, lighting, subgroup, n_samples,
    detections_correct) from a file path or raw CSV text."""
    text = path_or_text
    try:
        with open(path_or_text, encoding="utf-8") as f:
            text = f.read()
    except (FileNotFoundError, OSError):
        pass
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        row["n_samples"] = int(row["n_samples"])
        row["detections_correct"] = int(row["detections_correct"])
        rows.append(row)
    return rows


def weighted_accuracy(rows: list[dict]) -> float:
    """The single headline number a report would quote: total correct / total samples."""
    total_n = sum(r["n_samples"] for r in rows)
    total_correct = sum(r["detections_correct"] for r in rows)
    if total_n == 0:
        raise ValueError("no samples")
    return total_correct / total_n


def per_cell_accuracy(rows: list[dict]) -> list[dict]:
    """Per-cell accuracy -- the numbers a headline accuracy can hide."""
    out = []
    for r in rows:
        acc = r["detections_correct"] / r["n_samples"] if r["n_samples"] else 0.0
        out.append({**r, "accuracy": acc})
    return out


def worst_cell(rows: list[dict]) -> dict:
    """The cell a weighted-mean headline is most likely to be hiding."""
    cells = per_cell_accuracy(rows)
    return min(cells, key=lambda r: r["accuracy"])


# ---------------------------------------------------------------------------
# Pillar 2: Explainability -- a local explanation is not proof of global reasoning.
# ---------------------------------------------------------------------------

def make_explainability_demo_data(seed: int = 0, n: int = 400):
    """Two-feature synthetic data. The TRUE label depends only on feature A (signal).
    Feature B is pure noise but happens to correlate with the label for a specific,
    narrow region of A -- exactly the kind of local pattern a post-hoc explanation
    can mistake for the model's real reasoning."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3, 3, n)
    b = rng.normal(0, 1, n)
    y = (a > 0).astype(int)
    # In a narrow band of `a`, make `b` spuriously predictive by construction --
    # this is the region a local explanation is likely to be probed on.
    band = (a > -1.2) & (a < 1.2)
    b[band] = np.where(y[band] == 1, b[band] + 7.0, b[band] - 7.0)
    X = np.column_stack([a, b])
    return X, y


def train_explainability_demo_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


def local_explanation(model, x_row: np.ndarray, X_background: np.ndarray, n_samples: int = 200, seed: int = 0) -> dict:
    """A minimal, from-scratch local explanation: perturb one feature at a time
    (holding the others at the instance's own value) and measure how much the
    model's predicted probability moves. This is the same idea LIME/SHAP formalize --
    a *local* sensitivity measurement, not a description of what the model does
    everywhere."""
    rng = np.random.default_rng(seed)
    base_prob = model.predict_proba(x_row.reshape(1, -1))[0, 1]
    importances = {}
    for feat_idx, feat_name in enumerate(["feature_a", "feature_b"]):
        perturbed = np.tile(x_row, (n_samples, 1))
        perturbed[:, feat_idx] = rng.choice(X_background[:, feat_idx], size=n_samples, replace=True)
        probs = model.predict_proba(perturbed)[:, 1]
        importances[feat_name] = float(np.mean(np.abs(probs - base_prob)))
    return {"base_prob": float(base_prob), "local_importance": importances}


def global_feature_weight(model) -> dict:
    """The model's actual (global) linear coefficients, for comparison against the
    local explanation above."""
    coef = model.coef_[0]
    return {"feature_a": float(coef[0]), "feature_b": float(coef[1])}


# ---------------------------------------------------------------------------
# Pillar 3: Privacy -- a trained model can leak whether a point was in training data.
# ---------------------------------------------------------------------------

def make_membership_inference_demo_data(seed: int = 0, n_train: int = 40, n_test: int = 300):
    """A small, high-capacity model trained on very few, noisy points is the classic
    setup for a real (if toy) membership-inference gap: it memorizes its small
    training set more confidently than it generalizes to held-out points."""
    rng = np.random.default_rng(seed)
    n_features = 25
    X_train = rng.normal(0, 1, (n_train, n_features))
    y_train = (X_train[:, 0] + 1.5 * rng.normal(0, 1, n_train) > 0).astype(int)
    X_test = rng.normal(0, 1, (n_test, n_features))
    y_test = (X_test[:, 0] + 1.5 * rng.normal(0, 1, n_test) > 0).astype(int)
    return X_train, y_train, X_test, y_test


def train_membership_inference_demo_model(X_train, y_train):
    # Deliberately high-capacity relative to n_train=40, unregularized (alpha=0) --
    # this is what makes the model memorize rather than generalize, which is the
    # whole point of the demo, not a mistake to "fix".
    model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=3000, alpha=0.0, random_state=0)
    model.fit(X_train, y_train)
    return model


def membership_confidence_gap(model, X_train, X_test) -> dict:
    """Mean predicted-probability confidence on training points vs. held-out points.
    A real membership-inference attack does exactly this: query a model twice on
    a point-you-suspect-was-training-data and a point-you-know-was-not, and use the
    confidence gap to guess which set it came from."""
    train_conf = model.predict_proba(X_train).max(axis=1)
    test_conf = model.predict_proba(X_test).max(axis=1)
    return {
        "mean_train_confidence": float(train_conf.mean()),
        "mean_test_confidence": float(test_conf.mean()),
        "gap": float(train_conf.mean() - test_conf.mean()),
    }
