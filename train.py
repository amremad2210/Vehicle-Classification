from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import DataConfig, build_dataloaders, compute_class_weights_from_counts
from src.models import CustomCNN
from src.plots import plot_confusion_matrix, plot_history
from src.train import TrainConfig, fit
from src.utils import get_device, now_run_id, save_json, seed_everything, ensure_dir
from src.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a custom CNN for vehicle classification.")
    p.add_argument("--data-root", type=str, default="data/splits", help="Folder containing train/val/test splits.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (CUDA only).")
    p.add_argument("--no-weighted-sampler", action="store_true")
    p.add_argument("--no-weighted-loss", action="store_true")
    p.add_argument("--no-minority-heavy-aug", action="store_true")
    p.add_argument("--minority-quantile", type=float, default=0.35)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--run-name", type=str, default="", help="Optional run name; default is timestamp.")
    p.add_argument("--outputs", type=str, default="outputs", help="Base outputs directory.")
    return p.parse_args()


@torch.no_grad()
def evaluate_and_save_test(model: torch.nn.Module, test_loader, device, out_dir: Path, class_names):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(yb.detach().cpu().tolist())

    m = compute_metrics(y_true, y_pred, num_classes=len(class_names))
    save_json(out_dir / "test_metrics.json", m.__dict__)
    plot_confusion_matrix(m.confusion_matrix, class_names, out_dir / "confusion_matrix_test.png", normalize=True)
    return m


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = get_device(force_cpu=args.force_cpu)
    print(f"device: {device}")

    data_cfg = DataConfig(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_weighted_loss=not args.no_weighted_loss,
        minority_heavy_aug=not args.no_minority_heavy_aug,
        minority_quantile=args.minority_quantile,
    )

    train_loader, val_loader, test_loader, info = build_dataloaders(data_cfg)

    run_id = args.run_name.strip() or now_run_id("cnn")
    out_dir = ensure_dir(Path(args.outputs) / run_id)

    save_json(out_dir / "data_info.json", info)
    save_json(out_dir / "config.json", {"data": data_cfg.__dict__, "train": {"epochs": args.epochs, "lr": args.lr}})

    class_names = info["class_names"]
    counts = info["train_class_counts"]
    class_weights = None
    if data_cfg.use_weighted_loss:
        class_weights = compute_class_weights_from_counts(torch.tensor(counts).numpy())

    model = CustomCNN(num_classes=len(class_names)).to(device)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        log_every=50,
        save_best_on="macro_f1",
    )

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=out_dir,
        class_names=class_names,
        class_weights=class_weights,
        cfg=train_cfg,
    )

    plot_history(result["history"], out_dir / "training_curves.png")

    # Load best model for test metrics
    ckpt = torch.load(result["best_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate_and_save_test(model, test_loader, device, out_dir, class_names)
    print(f"test acc {test_m.accuracy:.3f} macroF1 {test_m.macro_f1:.3f}")


if __name__ == "__main__":
    main()


