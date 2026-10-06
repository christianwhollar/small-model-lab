import torch
from smallmodel.data import batch, records
from smallmodel.model import Int8Linear, TinyLM, adapt, quantize
from smallmodel.experiment import task_logits, train

torch.set_num_threads(2)


def test_templates_and_ids_are_held_out():
    training, testing = records("train", 100), records("test", 100)
    assert not {r.id for r in training} & {r.id for r in testing}
    assert not {r.text for r in training} & {r.text for r in testing}


def test_causal_mask_prevents_future_leakage():
    torch.manual_seed(17)
    model = TinyLM(width=32, layers=1).eval()
    original = torch.tensor([[3, 4, 5, 6]])
    changed = torch.tensor([[3, 4, 10, 11]])
    with torch.no_grad():
        assert torch.allclose(model(original)[:, :2], model(changed)[:, :2], atol=1e-6)


def test_lora_starts_equivalent_and_freezes_base():
    torch.manual_seed(17)
    base = TinyLM(width=32, layers=1)
    adapted = adapt(base)
    inputs, _, _ = batch(records("train", 4))
    assert torch.allclose(base(inputs), adapted(inputs), atol=1e-6)
    before = adapted.token.weight.detach().clone()
    train(adapted, records("train", 32), 5, "finetune")
    assert torch.equal(before, adapted.token.weight)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in adapted.parameters()
        if p.requires_grad
    )


def test_quantization_error_and_storage():
    torch.manual_seed(17)
    linear = torch.nn.Linear(64, 64)
    quantized = Int8Linear(linear)
    x = torch.randn(8, 64)
    assert (linear(x) - quantized(x)).abs().max() < 0.02
    assert quantized.weight_int8.element_size() == 1
    model = quantize(TinyLM(width=32, layers=1))
    logits, labels = task_logits(model, records("test", 4))
    assert logits.shape[0] == len(labels) and torch.isfinite(logits).all()


def test_training_reduces_supervised_loss():
    torch.manual_seed(17)
    model = TinyLM(width=32, layers=1)
    curve = train(model, records("train", 64), 25, "finetune")
    assert curve[-1]["loss"] < curve[0]["loss"]
