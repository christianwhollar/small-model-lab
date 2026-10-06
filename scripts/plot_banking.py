"""Render accuracy and compression from all recorded BANKING77 seeds."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
runs = [
    json.loads((root / f"reports/banking77/seed{s}/report.json").read_text()) for s in [17, 29, 41]
]
names = list(runs[0]["models"])
values = np.array([[r["models"][n]["test"]["accuracy"] for n in names] for r in runs])
labels = [
    "TF-IDF",
    "Full tuning",
    "LoRA",
    "Supervised\nstudent",
    "Distilled\nstudent",
    "INT8\nstudent",
]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
x = np.arange(len(names))
axes[0].bar(
    x,
    values.mean(0),
    yerr=values.std(0, ddof=1),
    capsize=4,
    color=["#79919f"] * 4 + ["#147a78", "#529a8c"],
)
axes[0].set_xticks(x, labels)
axes[0].set(ylim=(0, 1), ylabel="Test accuracy", title="Mean ± sample SD · 3 training/split seeds")
models = runs[0]["models"]
subset = ["full_finetune", "distilled_student", "int8_student"]
axes[1].bar(
    np.arange(3),
    [models[n]["serialized_bytes"] / 1e6 for n in subset],
    color=["#79919f", "#147a78", "#529a8c"],
)
axes[1].set_xticks(np.arange(3), ["Full encoder", "Distilled student", "INT8 student"])
axes[1].set(ylabel="Serialized weight size (MB)", title="Release seed 17 · storage, not speed")
for ax in axes:
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
fig.suptitle("BANKING77: 77 intents, 3,080 official test queries", fontsize=15)
for suffix in ["png", "svg"]:
    fig.savefig(root / f"reports/banking77-study.{suffix}", dpi=180)
