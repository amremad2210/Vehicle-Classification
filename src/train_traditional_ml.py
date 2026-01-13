"""Train and evaluate traditional ML classifiers on saved feature sets.

This script is intended to work with the feature-selection outputs produced by
`notebooks/2_feature_selection.ipynb`, e.g.:
  data/features/features_train_fstat.csv
  data/features/features_val_fstat.csv
  data/features/features_test_fstat.csv

It supports running a single (feature_method, model) combo, or sweeping multiple
models/methods and writing a summary CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "traditional_ml"


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]


def _load_class_names() -> list[str] | None:
    class_names_path = METADATA_DIR / "class_names.json"
    if not class_names_path.exists():
        return None

    with class_names_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    try:
        # keys are strings like "0", "1", ...
        items = sorted(((int(k), v) for k, v in mapping.items()), key=lambda kv: kv[0])
        return [v for _, v in items]
    except Exception:
        # Fall back to whatever ordering JSON gives.
        return list(mapping.values())


def _read_features_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}")

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column in {path}")

    y = df["label"].to_numpy()

    # Feature CSVs may or may not include image_path depending on which notebook wrote them.
    drop_cols = [c for c in ("label", "image_path") if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    X = feature_df.to_numpy(dtype=np.float64, copy=False)
    feature_columns = feature_df.columns.tolist()

    # Safety: guard against NaN/Inf in case older CSVs are used.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_columns


def load_dataset(method: str) -> DatasetSplit:
    """Load train/val/test CSVs for a given feature method.

    method options:
      - raw: features_train.csv / features_val.csv / features_test.csv
      - fstat, mutual_info, random_forest, pca: corresponding suffix CSVs
    """

    suffix = "" if method == "raw" else f"_{method}"

    X_train, y_train, cols_train = _read_features_csv(FEATURES_DIR / f"features_train{suffix}.csv")
    X_val, y_val, cols_val = _read_features_csv(FEATURES_DIR / f"features_val{suffix}.csv")
    X_test, y_test, cols_test = _read_features_csv(FEATURES_DIR / f"features_test{suffix}.csv")

    # Columns should match across splits; if not, we still proceed but warn by intersecting.
    if cols_train != cols_val or cols_train != cols_test:
        common = [c for c in cols_train if (c in set(cols_val) and c in set(cols_test))]
        if len(common) == 0:
            raise ValueError(
                "Feature columns do not match across splits and have no intersection. "
                "Re-generate features so train/val/test use identical columns."
            )

        def align(df_path: Path, common_cols: list[str]) -> np.ndarray:
            df = pd.read_csv(df_path)
            drop_cols = [c for c in ("label", "image_path") if c in df.columns]
            feat_df = df.drop(columns=drop_cols)
            feat_df = feat_df[common_cols]
            return np.nan_to_num(feat_df.to_numpy(dtype=np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        X_train = align(FEATURES_DIR / f"features_train{suffix}.csv", common)
        X_val = align(FEATURES_DIR / f"features_val{suffix}.csv", common)
        X_test = align(FEATURES_DIR / f"features_test{suffix}.csv", common)
        cols_train = common

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_columns=cols_train,
    )


def available_methods() -> list[str]:
    return ["fstat", "mutual_info", "random_forest", "pca", "raw"]


def available_models() -> list[str]:
    return ["dummy", "logreg", "linear_svm", "rbf_svm", "knn", "rf", "gnb"]


def make_estimator(model: str, seed: int) -> ClassifierMixin:
    if model == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=seed)

    if model == "logreg":
        # Strong baseline for standardized features.
        return LogisticRegression(
            max_iter=5000,
            solver="saga",
            n_jobs=-1,
            multi_class="auto",
            random_state=seed,
        )

    if model == "linear_svm":
        # Fast and effective on high-dimensional standardized features.
        return LinearSVC(C=1.0, random_state=seed)

    if model == "rbf_svm":
        # More flexible but can be slower on large datasets.
        return SVC(C=10.0, kernel="rbf", gamma="scale", random_state=seed)

    if model == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")

    if model == "rf":
        # Reasonably-regularized defaults (you can tune further).
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if model == "gnb":
        return GaussianNB()

    raise ValueError(f"Unknown model: {model}")


def evaluate(model: ClassifierMixin, ds: DatasetSplit, class_names: list[str] | None) -> dict[str, Any]:
    model.fit(ds.X_train, ds.y_train)

    preds_train = model.predict(ds.X_train)
    preds_val = model.predict(ds.X_val)
    preds_test = model.predict(ds.X_test)

    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }

    report = {
        "train": metrics(ds.y_train, preds_train),
        "val": metrics(ds.y_val, preds_val),
        "test": metrics(ds.y_test, preds_test),
        "test_classification_report": classification_report(
            ds.y_test,
            preds_test,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(ds.y_test, preds_test).tolist(),
    }

    return report


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_one(method: str, model_name: str, seed: int) -> dict[str, Any]:
    ds = load_dataset(method)
    class_names = _load_class_names()
    model = make_estimator(model_name, seed)

    report = evaluate(model, ds, class_names)

    out = {
        "method": method,
        "model": model_name,
        "seed": seed,
        "n_train": int(ds.X_train.shape[0]),
        "n_val": int(ds.X_val.shape[0]),
        "n_test": int(ds.X_test.shape[0]),
        "n_features": int(ds.X_train.shape[1]),
        "metrics": report,
    }
    return out


def run_sweep(methods: list[str], models: list[str], seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        ds = load_dataset(method)
        class_names = _load_class_names()

        for model_name in models:
            model = make_estimator(model_name, seed)
            report = evaluate(model, ds, class_names)
            rows.append(
                {
                    "method": method,
                    "model": model_name,
                    "n_features": int(ds.X_train.shape[1]),
                    "train_acc": report["train"]["accuracy"],
                    "val_acc": report["val"]["accuracy"],
                    "test_acc": report["test"]["accuracy"],
                    "train_f1_macro": report["train"]["f1_macro"],
                    "val_f1_macro": report["val"]["f1_macro"],
                    "test_f1_macro": report["test"]["f1_macro"],
                }
            )

    return pd.DataFrame(rows).sort_values(["val_acc", "test_acc"], ascending=False).reset_index(drop=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate classical ML models on saved feature sets")

    p.add_argument(
        "--method",
        default="fstat",
        choices=available_methods() + ["all"],
        help="Which feature set to use: fstat/mutual_info/random_forest/pca/raw, or 'all'",
    )
    p.add_argument(
        "--model",
        default="linear_svm",
        choices=available_models() + ["all"],
        help="Which classifier to train, or 'all'",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--outdir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write results (default: outputs/traditional_ml)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.method == "all" or args.model == "all":
        methods = available_methods() if args.method == "all" else [args.method]
        models = available_models() if args.model == "all" else [args.model]

        df = run_sweep(methods=methods, models=models, seed=args.seed)
        out_csv = outdir / f"sweep_{timestamp}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved sweep summary to: {out_csv}")
        print(df.head(20).to_string(index=False))
        return

    result = run_one(method=args.method, model_name=args.model, seed=args.seed)
    out_json = outdir / f"result_{args.method}_{args.model}_{timestamp}.json"
    _write_json(out_json, result)

    metrics = result["metrics"]
    print("=" * 70)
    print(f"Method: {result['method']} | Model: {result['model']} | Features: {result['n_features']}")
    print("=" * 70)
    print(
        f"Train acc={metrics['train']['accuracy']:.4f} | Val acc={metrics['val']['accuracy']:.4f} | Test acc={metrics['test']['accuracy']:.4f}"
    )
    print(
        f"Train F1(macro)={metrics['train']['f1_macro']:.4f} | Val F1(macro)={metrics['val']['f1_macro']:.4f} | Test F1(macro)={metrics['test']['f1_macro']:.4f}"
    )
    print(f"Saved full report to: {out_json}")


if __name__ == "__main__":
    main()
