"""Pinned BANKING77 inputs and a leakage-audited, stratified development split."""

import csv
import hashlib
import io
import re
from pathlib import Path

import httpx
import numpy as np
from sklearn.model_selection import train_test_split

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE = (
    f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
)
CHECKSUMS = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def normalized(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def download(cache, split):
    path = Path(cache) / f"banking77-{split}.csv"
    if not path.exists():
        response = httpx.get(f"{BASE}/{split}.csv", follow_redirects=True, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKSUMS[split]:
        raise ValueError(f"BANKING77 {split} checksum mismatch")
    return list(csv.DictReader(io.StringIO(raw.decode())))


def load_splits(cache, seed=17):
    train_rows, test_rows = download(cache, "train"), download(cache, "test")
    labels = sorted({r["category"] for r in train_rows})
    label_ids = {label: i for i, label in enumerate(labels)}
    # Use test text only to remove exact normalized duplicates; never use test labels
    # for fitting, checkpoint selection, calibration, or hyperparameter selection.
    test_texts = {normalized(r["text"]) for r in test_rows}
    seen, unique, removed = set(), [], 0
    for i, row in enumerate(train_rows):
        text = normalized(row["text"])
        if text in seen or text in test_texts:
            removed += 1
            continue
        seen.add(text)
        unique.append({**row, "id": f"train-{i}", "label": label_ids[row["category"]]})
    indices = np.arange(len(unique))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=seed,
        stratify=[row["label"] for row in unique],
    )
    test = [
        {**r, "id": f"test-{i}", "label": label_ids[r["category"]]} for i, r in enumerate(test_rows)
    ]
    splits = {
        "train": [unique[i] for i in train_idx],
        "validation": [unique[i] for i in val_idx],
        "test": test,
    }
    audit = {
        "dataset": "BANKING77",
        "revision": REVISION,
        "sha256": CHECKSUMS,
        "license": "CC-BY-4.0",
        "normalized_duplicates_removed_from_train": removed,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_seed": seed,
        "split_hashes": {
            name: hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()
            for name, rows in splits.items()
        },
    }
    return splits, labels, audit
