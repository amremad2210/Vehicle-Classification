from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    confusion_matrix: List[List[int]]


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Metrics:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=list(range(num_classes))))

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    return Metrics(
        accuracy=acc,
        macro_f1=macro,
        per_class_precision=[float(x) for x in p],
        per_class_recall=[float(x) for x in r],
        per_class_f1=[float(x) for x in f1],
        confusion_matrix=cm.astype(int).tolist(),
    )


def classification_report_dict(
    y_true: List[int], y_pred: List[int], target_names: List[str]
) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


