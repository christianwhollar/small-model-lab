"""An order-insensitive student with actual CPU quantized embedding/linear kernels."""

import copy
import torch
from torch import nn


class IntentStudent(nn.Module):
    def __init__(self, vocabulary=30522, width=48, classes=77):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, classes))

    def forward(self, input_ids, attention_mask, **_):
        mask = attention_mask.unsqueeze(-1).to(torch.float32)
        pooled = (self.embedding(input_ids) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)


def quantize_student(model):
    """PyTorch eager dynamic int8, not float dequantization inside a custom layer."""
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        float_qparams_weight_only_qconfig,
        quantize_dynamic,
    )

    engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
    torch.backends.quantized.engine = engine
    result = quantize_dynamic(
        copy.deepcopy(model).cpu().eval(),
        {
            nn.Embedding: float_qparams_weight_only_qconfig,
            nn.Linear: default_dynamic_qconfig,
        },
        inplace=False,
    )
    return result


def distillation_loss(student, labels, teacher=None, temperature=2.0, alpha=0.5):
    hard = nn.functional.cross_entropy(student, labels)
    if teacher is None:
        return hard
    soft = (
        nn.functional.kl_div(
            nn.functional.log_softmax(student / temperature, dim=-1),
            nn.functional.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return alpha * soft + (1 - alpha) * hard
