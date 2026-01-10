from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: Dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # With a single epoch, lines can look "empty" because a line needs 2+ points.
    # Always show markers so 1-epoch smoke tests still render visibly.
    plot_kws = {"marker": "o", "markersize": 3, "linewidth": 1.5}

    axes[0].plot(epochs, history.get("train_loss", []), label="train", **plot_kws)
    axes[0].plot(epochs, history.get("val_loss", []), label="val", **plot_kws)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc", []), label="train acc", **plot_kws)
    axes[1].plot(epochs, history.get("val_acc", []), label="val acc", **plot_kws)
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history.get("val_macro_f1", []), label="val macroF1", **plot_kws)
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if len(epochs) == 1:
        # Make the single point easier to see by widening x-limits slightly.
        axes[0].set_xlim(0.5, 1.5)
        axes[1].set_xlim(0.5, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    out_path: str | Path,
    normalize: bool = True,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = np.clip(mat.sum(axis=1, keepdims=True), a_min=1.0, a_max=None)
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


