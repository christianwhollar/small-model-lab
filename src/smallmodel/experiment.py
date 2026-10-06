import argparse
import copy
import io
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import INDEX, LABELS, batch, fingerprint, records
from .model import TinyLM, adapt, quantize


def task_logits(model, items):
    inputs, lengths, labels = batch(items)
    return model(inputs)[torch.arange(len(items)), lengths - 1], labels


def train(model, items, steps, mode, teacher=None, seed=17):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.004)
    losses = []
    model.train()
    for step in range(steps):
        sample = rng.sample(items, min(32, len(items)))
        inputs, lengths, labels = batch(sample)
        logits = model(inputs)
        if mode == "pretrain":
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                inputs[:, 1:].reshape(-1),
                ignore_index=0,
            )
        else:
            last = logits[torch.arange(len(sample)), lengths - 1]
            loss = F.cross_entropy(last, labels)
            if teacher is not None:
                with torch.no_grad():
                    targets, _ = task_logits(teacher, sample)
                temperature = 2.0
                divergence = (
                    F.kl_div(
                        F.log_softmax(last / temperature, dim=-1),
                        F.softmax(targets / temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * temperature**2
                )
                loss = 0.5 * loss + 0.5 * divergence
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    return losses


def evaluate(model, items):
    model.eval()
    with torch.no_grad():
        logits, labels = task_logits(model, items)
        prediction = logits.argmax(-1)
        confidence = F.softmax(logits, dim=-1).max(-1).values
        nll = float(F.cross_entropy(logits, labels))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    example, _, _ = batch(items[:1])
    with torch.no_grad():
        for _ in range(5):
            model(example)
        times = []
        for _ in range(30):
            start = time.perf_counter()
            model(example)
            times.append((time.perf_counter() - start) * 1000)
    chosen = confidence >= 0.8
    correct = prediction == labels
    return {
        "accuracy": float(correct.float().mean()),
        "target_token_nll": nll,
        "coverage_at_0_8": float(chosen.float().mean()),
        "accuracy_at_0_8": float(correct[chosen].float().mean()) if chosen.any() else None,
        "state_dict_bytes": len(buffer.getvalue()),
        "p50_forward_ms": float(np.median(times)),
        "p95_forward_ms": float(np.quantile(times, 0.95)),
        "per_class_accuracy": {
            label: float(correct[labels == INDEX[label]].float().mean()) for label in LABELS
        },
        "predictions": [
            {
                "id": r.id,
                "truth": r.label,
                "predicted": next(
                    (label for label in LABELS if INDEX[label] == int(prediction[i])),
                    "invalid_token",
                ),
                "confidence": float(confidence[i]),
            }
            for i, r in enumerate(items)
        ],
    }


def experiment(steps=150, seed=17):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train_data, val_data, test_data = (
        records("train", 320, seed),
        records("validation", 80, seed),
        records("test", 160, seed),
    )
    started = time.perf_counter()
    base = TinyLM()
    pretraining = train(base, train_data, steps, "pretrain", seed=seed)
    teacher = adapt(base)
    adapting = train(teacher, train_data, steps, "finetune", seed=seed)
    full_teacher = copy.deepcopy(base)
    full_tuning = train(full_teacher, train_data, steps, "finetune", seed=seed)
    validation = {"lora": evaluate(teacher, val_data), "full": evaluate(full_teacher, val_data)}
    selected = max(
        validation,
        key=lambda name: (validation[name]["accuracy"], -validation[name]["target_token_nll"]),
    )
    distillation_teacher = {"lora": teacher, "full": full_teacher}[selected]
    torch.manual_seed(seed + 1)
    student = TinyLM(width=32, layers=1)
    supervised = copy.deepcopy(student)
    supervision = train(supervised, train_data, steps, "finetune", seed=seed)
    distilling = train(
        student, train_data, steps, "distill", teacher=distillation_teacher, seed=seed
    )
    compressed = quantize(student)
    training_seconds = time.perf_counter() - started
    return {
        "protocol": "synthetic-ticket-language-v2",
        "seed": seed,
        "steps_per_stage": steps,
        "scope": "Tiny causal language model on a closed synthetic vocabulary. Scores do not establish general language or financial-domain competence. Test field orders are absent from training. Quantization compresses stored weights; execution dequantizes to float32 and may be slower.",
        "environment": {"platform": platform.platform(), "torch": torch.__version__, "threads": 2},
        "data": {
            name: {"count": len(items), "sha256": fingerprint(items)}
            for name, items in (
                ("train", train_data),
                ("validation", val_data),
                ("test", test_data),
            )
        },
        "parameters": {
            "base": sum(p.numel() for p in base.parameters()),
            "adapter_trainable": sum(p.numel() for p in teacher.parameters() if p.requires_grad),
            "student": sum(p.numel() for p in student.parameters()),
        },
        "training_wall_seconds": training_seconds,
        "learning_curves": {
            "pretraining": pretraining,
            "lora": adapting,
            "full_finetune": full_tuning,
            "supervised_student": supervision,
            "distillation": distilling,
        },
        "validation_teachers": validation,
        "teacher_selected_on_validation": selected,
        "test": {
            "base": evaluate(base, test_data),
            "lora_teacher": evaluate(teacher, test_data),
            "full_teacher": evaluate(full_teacher, test_data),
            "supervised_student": evaluate(supervised, test_data),
            "student": evaluate(student, test_data),
            "int8_student": evaluate(compressed, test_data),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="runtime/study.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a fresh output path")
    result = experiment(args.steps, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "training_seconds": result["training_wall_seconds"],
                "test": {
                    name: {k: v for k, v in row.items() if k != "predictions"}
                    for name, row in result["test"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
