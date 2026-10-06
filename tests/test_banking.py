import pytest
import numpy as np
import torch
from smallmodel.bank_metrics import (
    metrics,
    paired_accuracy_ci,
    selective_threshold,
    temperature_scale,
)
from smallmodel.compression import IntentStudent, distillation_loss, quantize_student


def test_confidence_threshold_and_paired_uncertainty():
    logits = np.array([[4, 0], [0, 4], [1, 0], [1, 0]], dtype=float)
    labels = np.array([0, 1, 1, 0])
    t = temperature_scale(logits, labels)
    assert 0.2 <= t <= 5
    probs = np.array([[0.99, 0.01], [0.01, 0.99], [0.6, 0.4], [0.6, 0.4]])
    threshold = selective_threshold(probs, labels, target=0.95, minimum=2)
    assert threshold == 0.99
    result = metrics(probs, labels, threshold)
    assert result["coverage"] == 0.5 and result["selective_accuracy"] == 1
    assert paired_accuracy_ci(probs, probs, labels)["ci95"] == [0, 0]


def test_distillation_gradients_do_not_flow_into_teacher():
    model = IntentStudent(vocabulary=50, width=8, classes=3)
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = (ids != 0).long()
    teacher = torch.randn(2, 3)
    loss = distillation_loss(model(ids, mask), torch.tensor([0, 2]), teacher)
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert teacher.grad is None


def test_quantized_model_executes_integer_modules():
    model = IntentStudent(vocabulary=50, width=8, classes=3).eval()
    quantized = quantize_student(model)
    assert "quantized" in type(quantized.embedding).__module__
    assert "quantized" in type(quantized.head[0]).__module__
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        assert torch.max(torch.abs(model(ids, mask) - quantized(ids, mask))).item() < 0.1


def test_empty_selection_is_reported_without_nan():
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = metrics(p, np.array([0, 1]), 1.01)
    assert result["coverage"] == 0 and result["selective_accuracy"] is None


def test_artifact_corruption_rejected_before_model_allocation(tmp_path):
    import hashlib
    import json
    from smallmodel.inference import IntentEngine

    path = tmp_path / "distilled_student.safetensors"
    path.write_bytes(b"corrupt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"sha256": {"distilled_student.safetensors": hashlib.sha256(b"original").hexdigest()}}
        )
    )
    with pytest.raises(ValueError, match="integrity"):
        IntentEngine(tmp_path)
