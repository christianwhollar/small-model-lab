"""Safe-tensor inference for the released BANKING77 student bundle."""

import hashlib
import json
from pathlib import Path
import threading
import time

import torch
from .compression import IntentStudent, quantize_student


class IntentEngine:
    def __init__(self, artifact, quantized=False):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        self.path = Path(artifact)
        manifest = json.loads((self.path / "manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            file = (self.path / name).resolve()
            if (
                not file.is_relative_to(self.path.resolve())
                or hashlib.sha256(file.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Artifact integrity verification failed")
        required = {
            "metadata.json",
            "distilled_student.safetensors",
            "tokenizer/tokenizer_config.json",
        }
        if not required <= manifest["sha256"].keys():
            raise ValueError("Manifest is missing required model components")
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        architecture = self.metadata["architecture"]
        if architecture != {"vocabulary": 30522, "width": 48, "classes": 77}:
            raise ValueError("Unsupported model architecture")
        self.model = IntentStudent(**architecture)
        self.model.load_state_dict(load_file(str(self.path / "distilled_student.safetensors")))
        torch.set_num_threads(4)
        self.model = quantize_student(self.model) if quantized else self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path / "tokenizer", local_files_only=True
        )
        self.quantized, self.lock = quantized, threading.Lock()
        self.fingerprint = manifest["sha256"]["distilled_student.safetensors"]

    def predict(self, texts):
        if not texts or len(texts) > 32 or any(not t.strip() or len(t) > 2000 for t in texts):
            raise ValueError("Provide 1..32 nonempty texts, each at most 2000 characters")
        started = time.perf_counter()
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with self.lock, torch.inference_mode():
            logits = self.model(**tokens)
            probs = torch.softmax(logits / self.metadata["temperature"], dim=-1)
            values, indices = probs.topk(3, dim=-1)
        results = []
        for confidence, predictions in zip(values.tolist(), indices.tolist()):
            abstained = confidence[0] < self.metadata["threshold"]
            results.append(
                {
                    "decision": "human_review" if abstained else "suggest_intent",
                    "abstained": abstained,
                    "suggestions": [
                        {"intent": self.metadata["labels"][i], "confidence": p}
                        for i, p in zip(predictions, confidence)
                    ],
                }
            )
        return {
            "results": results,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "includes_tokenization": True,
            "model_sha256": self.fingerprint,
            "quantized": self.quantized,
            "threshold": self.metadata["threshold"],
        }
