from __future__ import annotations
from typing import Dict, Any

import numpy as np
import secretflow as sf
from secretflow.stats.biclassification_eval import BiClassificationEval


def eval_biclassification(test_y, y_score, bucket_size: int = 20) -> Dict[str, Any]:
    evaluator = BiClassificationEval(y_true=test_y, y_score=y_score, bucket_size=bucket_size)
    r = sf.reveal(evaluator.get_all_reports())
    return {
        "positive_samples": r.summary_report.positive_samples,
        "negative_samples": r.summary_report.negative_samples,
        "auc": r.summary_report.auc,
        "ks": r.summary_report.ks,
        "f1_score": r.summary_report.f1_score,
    }


def confusion_at_threshold(bob_pyu, test_y, y_score, threshold: float):
    """
    Compute confusion matrix stats at a given threshold on Bob.
    Return a dict with accuracy, recalls, f1, confusion_matrix (list).
    """
    def _bob_eval(y_true, y_pred_proba, thr):
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

        y_pred = (y_pred_proba >= thr)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True
        )
        return {
            "accuracy": float(acc),
            "negative_recall": float(report["Negative"]["recall"]),
            "positive_recall": float(report["Positive"]["recall"]),
            "f1_score": float(report["Positive"]["f1-score"]),
            "confusion_matrix": cm.tolist(),
        }

    result = sf.reveal(bob_pyu(_bob_eval)(test_y.partitions[bob_pyu].data,
                                          y_score.partitions[bob_pyu].data,
                                          threshold))
    return result