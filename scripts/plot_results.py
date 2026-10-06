"""Regenerate the figure from the three fixed v2 runs."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
runs = [json.loads((root / f"reports/seed{s}-v2.json").read_text()) for s in (17, 29, 41)]
names = ["lora_teacher", "full_teacher", "supervised_student", "student", "int8_student"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
x = np.arange(len(names))
labels = [
    "LoRA\nteacher",
    "Full\nteacher",
    "Supervised\nstudent",
    "Distilled\nstudent",
    "Int8\nstudent",
]
for i, run in enumerate(runs):
    axes[0].plot(
        x,
        [run["test"][name]["accuracy"] for name in names],
        marker="o",
        label=f"Seed {run['seed']}",
        alpha=0.8,
    )
axes[0].set(
    xticks=x,
    xticklabels=labels,
    ylabel="Action-token accuracy",
    ylim=(0, 1.05),
    title="Unseen field orders; substantial seed variation",
)
axes[0].legend(frameon=False)
axes[1].bar(
    labels, [runs[0]["test"][name]["state_dict_bytes"] / 1024 for name in names], color="#2b6777"
)
axes[1].set(
    ylabel="Serialized state dictionary (KiB)",
    title="Storage reduction, not an integer-kernel speedup",
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "reports/model-comparison.svg")

fig.savefig(root / "reports/model-comparison.png", dpi=150)
