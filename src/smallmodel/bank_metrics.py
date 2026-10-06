"""Calibration, selective prediction and paired uncertainty for intent experiments."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss


def temperature_scale(logits, labels):
    result = minimize_scalar(
        lambda t: log_loss(labels, softmax(logits / t, axis=1), labels=np.arange(logits.shape[1])),
        bounds=(0.2, 5),
        method="bounded",
    )
    return float(result.x)


def selective_threshold(probabilities, labels, target=0.95, minimum=50):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    candidates = []
    for threshold in np.unique(confidence):
        mask = confidence >= threshold
        if mask.sum() >= minimum and correct[mask].mean() >= target:
            candidates.append(float(threshold))
    return min(candidates) if candidates else 1.01


def metrics(probabilities, labels, threshold=0):
    labels = np.asarray(labels)
    predictions, confidence = probabilities.argmax(1), probabilities.max(1)
    correct = predictions == labels
    selected = confidence >= threshold
    ece = 0.0
    for low in np.linspace(0, 0.9, 10):
        mask = (confidence > low) & (confidence <= low + 0.1 + 1e-9)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "nll": log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])),
        "ece_10_bins": float(ece),
        "coverage": float(selected.mean()),
        "selective_accuracy": float(correct[selected].mean()) if selected.any() else None,
        "threshold": threshold,
    }


def paired_accuracy_ci(first, second, labels, seed=17, samples=2000):
    differences = (first.argmax(1) == labels).astype(float) - (second.argmax(1) == labels)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(differences, len(differences)).mean()) for _ in range(samples)]
    return {
        "difference": float(differences.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "unit": "test example, conditional on trained checkpoints",
        "samples": samples,
    }
