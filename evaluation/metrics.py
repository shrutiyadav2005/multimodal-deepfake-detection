"""
Evaluation metrics computation module.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_scores: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Computes standard forensic classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0: Real, 1: Fake).
        y_pred: Binary predictions (0 or 1).
        y_scores: Optional probability scores for class 1 (Fake).

    Returns:
        Dict containing accuracy, precision, recall, f1, confusion matrix, and ROC-AUC.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if len(y_true_arr) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "sample_count": 0,
        }

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()

    roc_auc = 0.0
    roc_data = None
    if y_scores is not None and len(np.unique(y_true_arr)) > 1:
        try:
            y_scores_arr = np.array(y_scores)
            roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
            fpr, tpr, thresh = roc_curve(y_true_arr, y_scores_arr)
            roc_data = {
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr],
            }
        except Exception:
            roc_auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "sample_count": len(y_true_arr),
    }
