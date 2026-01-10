from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, classification_report_dict
from .utils import ensure_dir, save_json


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    log_every: int = 50
    save_best_on: str = "macro_f1"  # or "val_loss"


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(float(loss.item()))
        accs.append(_accuracy_from_logits(logits, yb))

        preds = torch.argmax(logits, dim=1)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return float(np.mean(losses)), float(np.mean(accs)), y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: TrainConfig,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    accs: List[float] = []

    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    for step, (xb, yb) in enumerate(pbar, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        accs.append(_accuracy_from_logits(logits, yb))

        if step % max(1, cfg.log_every) == 0:
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{np.mean(accs):.3f}")

    return float(np.mean(losses)), float(np.mean(accs))


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    class_names: List[str],
    class_weights: torch.Tensor | None,
    cfg: TrainConfig,
) -> Dict:
    out = ensure_dir(output_dir)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_score = -math.inf
    best_path = out / "best.pt"
    last_path = out / "last.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
        )
        val_loss, val_acc, y_true, y_pred = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )
        m = compute_metrics(y_true, y_pred, num_classes=len(class_names))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(m.macro_f1)

        # Save "last" every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "class_names": class_names,
            },
            last_path,
        )

        if cfg.save_best_on == "val_loss":
            score = -val_loss
        else:
            score = m.macro_f1

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "class_names": class_names,
                },
                best_path,
            )

        # Persist epoch metrics
        save_json(out / "val_metrics_epoch.json", {"epoch": epoch, **m.__dict__})
        save_json(out / "val_classification_report.json", classification_report_dict(y_true, y_pred, class_names))

        print(
            f"epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} macroF1 {m.macro_f1:.3f}"
        )

    return {"history": history, "best_path": str(best_path), "last_path": str(last_path)}


