"""Reproducible BANKING77 study: lexical baseline, pretrained adaptation and compression.

Test metrics are computed only after selecting checkpoints and confidence thresholds on
validation data. This is an exploratory public benchmark, not a blind evaluation.
"""

import argparse
import copy
import hashlib
import io
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from .banking_data import load_splits
from .bank_metrics import metrics, paired_accuracy_ci, selective_threshold, temperature_scale
from .compression import IntentStudent, distillation_loss, quantize_student

MODEL_ID = "prajjwal1/bert-tiny"
MODEL_REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def logits_of(model, ids, mask):
    output = model(input_ids=ids, attention_mask=mask)
    return output.logits if hasattr(output, "logits") else output


def encode(tokenizer, rows):
    values = tokenizer(
        [r["text"] for r in rows],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    return values["input_ids"], values["attention_mask"], torch.tensor([r["label"] for r in rows])


def predict(model, data, device, batch_size=128):
    model.eval().to(device)
    rows = []
    with torch.inference_mode():
        for ids, mask, _ in DataLoader(TensorDataset(*data), batch_size=batch_size):
            rows.append(logits_of(model, ids.to(device), mask.to(device)).cpu().numpy())
    return np.concatenate(rows)


def fit(model, train, validation, device, epochs, lr, seed, teacher=None):
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(seed)
    tensors = (*train, torch.tensor(teacher)) if teacher is not None else train
    loader = DataLoader(TensorDataset(*tensors), batch_size=64, shuffle=True, generator=generator)
    history, best, state = [], -1, None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            ids, mask, labels = [x.to(device) for x in batch[:3]]
            logits = logits_of(model, ids, mask)
            loss = distillation_loss(
                logits, labels, batch[3].to(device) if teacher is not None else None
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += loss.item() * len(ids)
        validation_logits = predict(model, validation, device)
        accuracy = float((validation_logits.argmax(1) == validation[2].numpy()).mean())
        entry = {"epoch": epoch + 1, "loss": total / len(train[0]), "validation_accuracy": accuracy}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if accuracy > best:
            best = accuracy
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return {
        "history": history,
        "selected_epoch": max(history, key=lambda r: r["validation_accuracy"])["epoch"],
        "train_seconds": time.perf_counter() - started,
    }


def serialized_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def latency(model, data, repetitions=100):
    model.cpu().eval()
    durations = []
    with torch.inference_mode():
        for i in range(repetitions + 10):
            index = i % len(data[0])
            before = time.perf_counter()
            logits_of(model, data[0][index : index + 1], data[1][index : index + 1])
            if i >= 10:
                durations.append((time.perf_counter() - before) * 1000)
    return {
        "batch_size": 1,
        "samples": repetitions,
        "warmup": 10,
        "p50_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "includes_tokenization": False,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def evaluate(model, data, device, labels, name, report_dir, threshold=None, temperature=None):
    val = predict(model, data["validation"], device)
    if temperature is None:
        temperature = temperature_scale(val, data["validation"][2].numpy())
    val_probs = softmax(val / temperature, axis=1)
    if threshold is None:
        threshold = selective_threshold(val_probs, data["validation"][2].numpy())
    test_logits = predict(model, data["test"], device)
    probs = softmax(test_logits / temperature, axis=1)
    np.savez_compressed(
        report_dir / f"{name}-predictions.npz", probabilities=probs, labels=data["test"][2].numpy()
    )
    return {
        "test": metrics(probs, data["test"][2].numpy(), threshold),
        "validation": metrics(val_probs, data["validation"][2].numpy(), threshold),
        "temperature": temperature,
        "serialized_bytes": serialized_size(model),
        "latency": latency(model, data["test"]),
        "confusions": confusions(probs, data["test"][2].numpy(), labels),
    }, probs


def confusions(probabilities, truth, labels):
    from collections import Counter

    pairs = Counter(
        (labels[int(a)], labels[int(b)]) for a, b in zip(truth, probabilities.argmax(1)) if a != b
    )
    return [{"truth": a, "prediction": b, "count": n} for (a, b), n in pairs.most_common(12)]


def run(output, seed=17, device="cpu", epochs=10, student_epochs=20, cache="runtime/data"):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    splits, labels, audit = load_splits(cache, seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.save_pretrained(artifacts / "tokenizer")
    data = {name: encode(tokenizer, rows) for name, rows in splits.items()}
    config = {
        "seed": seed,
        "epochs": epochs,
        "student_epochs": student_epochs,
        "device": device,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": 64,
        "teacher_lr": 0.0003,
        "lora_lr": 0.001,
        "student_lr": 0.003,
        "distillation_temperature": 2,
        "distillation_alpha": 0.5,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_targets": ["query", "value"],
        "selection": "maximum validation accuracy; earliest tie",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "study": "BANKING77 exploratory v1",
    }
    # Save protocol before any training or test evaluation.
    (output / "protocol.json").write_text(json.dumps({"config": config, "data": audit}, indent=2))
    report = {
        "config": config,
        "data": audit,
        "labels": labels,
        "models": {},
        "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
    }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([r["text"] for r in splits["train"]])
    baseline = LogisticRegression(C=4, max_iter=300).fit(matrix, data["train"][2].numpy())
    baseline_probs = baseline.predict_proba(
        vectorizer.transform([r["text"] for r in splits["test"]])
    )
    report["models"]["tfidf_logistic"] = {
        "test": metrics(baseline_probs, data["test"][2].numpy()),
        "description": "word unigrams/bigrams, C=4 fixed before evaluation",
    }
    np.savez_compressed(
        output / "tfidf_logistic-predictions.npz",
        probabilities=baseline_probs,
        labels=data["test"][2].numpy(),
    )

    def fresh():
        seed_everything(seed)
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, num_labels=len(labels), attn_implementation="eager"
        )

    teacher = fresh()
    print("Training full pretrained encoder", flush=True)
    training = fit(teacher, data["train"], data["validation"], device, epochs, 0.0003, seed)
    teacher.cpu().save_pretrained(artifacts / "teacher")
    result, teacher_probs = evaluate(teacher, data, device, labels, "full_finetune", output)
    report["models"]["full_finetune"] = {
        **result,
        **training,
        "parameters": sum(p.numel() for p in teacher.parameters()),
    }
    # The fixed full-tuning teacher supplies only training logits to distillation.
    teacher_train = predict(teacher, data["train"], device)
    teacher.cpu()

    print("Training LoRA encoder", flush=True)
    adapter = get_peft_model(
        fresh(),
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
        ),
    )
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    training = fit(adapter, data["train"], data["validation"], device, epochs, 0.001, seed)
    adapter.cpu().save_pretrained(artifacts / "lora")
    merged = adapter.merge_and_unload()
    result, _ = evaluate(merged, data, device, labels, "lora", output)
    report["models"]["lora"] = {
        **result,
        **training,
        "trainable_parameters": trainable,
        "parameters": sum(p.numel() for p in merged.parameters()),
    }
    del merged, adapter

    seed_everything(seed)
    initial = IntentStudent(classes=len(labels))
    student_predictions = {}
    for name, targets in [("supervised_student", None), ("distilled_student", teacher_train)]:
        print(f"Training {name}", flush=True)
        seed_everything(seed)
        student = copy.deepcopy(initial)
        training = fit(
            student, data["train"], data["validation"], device, student_epochs, 0.003, seed, targets
        )
        student.cpu()
        from safetensors.torch import save_file

        save_file(student.state_dict(), str(artifacts / f"{name}.safetensors"))
        result, probs = evaluate(student, data, device, labels, name, output)
        report["models"][name] = {
            **result,
            **training,
            "parameters": sum(p.numel() for p in student.parameters()),
        }
        student_predictions[name] = probs
        if targets is not None:
            quantized = quantize_student(student)
            qresult, qprobs = evaluate(
                quantized,
                data,
                "cpu",
                labels,
                "int8_student",
                output,
                result["test"]["threshold"],
                result["temperature"],
            )
            report["models"]["int8_student"] = {
                **qresult,
                "quantized_engine": torch.backends.quantized.engine,
                "kernels": [str(type(quantized.embedding)), str(type(quantized.head[0]))],
            }
            report["quantization_agreement"] = float((qprobs.argmax(1) == probs.argmax(1)).mean())
    report["paired_accuracy"] = {
        "distilled_minus_supervised": paired_accuracy_ci(
            student_predictions["distilled_student"],
            student_predictions["supervised_student"],
            data["test"][2].numpy(),
        ),
        "teacher_minus_lexical": paired_accuracy_ci(
            teacher_probs, baseline_probs, data["test"][2].numpy()
        ),
    }
    metadata = {
        "labels": labels,
        "architecture": {"vocabulary": 30522, "width": 48, "classes": len(labels)},
        "temperature": report["models"]["distilled_student"]["temperature"],
        "threshold": report["models"]["distilled_student"]["test"]["threshold"],
        "dataset": audit,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({n: r["test"] for n, r in report["models"].items()}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--cache", default="runtime/data")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.student_epochs <= 100:
        parser.error("Epoch counts must be between 1 and 100")
    run(args.output, args.seed, args.device, args.epochs, args.student_epochs, args.cache)


if __name__ == "__main__":
    main()
