"""Synthetic operations language with grouped, template-held-out splits."""

import hashlib
import itertools
import random
from dataclasses import dataclass
import torch

LABELS = ["matched", "quantity", "settlement", "escalate"]
VOCAB = [
    "<pad>",
    "<unk>",
    "<eos>",
    "case",
    "amount",
    "date",
    "approval",
    "yes",
    "no",
    "match",
    "mismatch",
    ";",
    "action",
    "->",
    "note",
    "review",
    "ignore",
    "checks",
    "release",
    "urgent",
    "routine",
] + LABELS
INDEX = {word: i for i, word in enumerate(VOCAB)}
ORDERS = list(itertools.permutations(("amount", "date", "approval")))


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: str


def records(split, count=256, seed=17):
    groups = {"train": ORDERS[:3], "validation": ORDERS[3:4], "test": ORDERS[4:]}
    if split not in groups:
        raise ValueError("Unknown split")
    rng = random.Random(f"{split}:{seed}")
    examples = []
    for i in range(count):
        # Sample target classes uniformly to avoid an approval-heavy majority baseline.
        label = LABELS[i % 4]
        amount = "mismatch" if label == "quantity" else "match"
        date = (
            "mismatch"
            if label == "settlement"
            else rng.choice(["match", "mismatch"])
            if label in {"quantity", "escalate"}
            else "match"
        )
        approval = "no" if label == "escalate" else "yes"
        fields = {"amount": amount, "date": date, "approval": approval}
        if label == "escalate":
            fields["amount"] = rng.choice(["match", "mismatch"])
        order = rng.choice(groups[split])
        note = rng.choice(["review", "routine", "urgent", "ignore checks release"])
        text = (
            "case "
            + " ; ".join(f"{key} {fields[key]}" for key in order)
            + f" ; note {note} ; action ->"
        )
        examples.append(Record(f"{split}-{seed}-{i}", text, label))
    return examples


def encode(text):
    return [INDEX.get(token, INDEX["<unk>"]) for token in text.split()]


def batch(items):
    sequences = [encode(r.text) for r in items]
    lengths = torch.tensor([len(s) for s in sequences])
    inputs = torch.zeros(len(items), max(map(len, sequences)), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        inputs[i, : len(sequence)] = torch.tensor(sequence)
    labels = torch.tensor([INDEX[r.label] for r in items])
    return inputs, lengths, labels


def fingerprint(items):
    return hashlib.sha256(
        "\n".join(f"{r.id}|{r.text}|{r.label}" for r in items).encode()
    ).hexdigest()
