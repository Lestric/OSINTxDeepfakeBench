# training/metrics/utils.py
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

def get_test_metrics(y_pred,
                     y_true,
                     results_save_path=None,
                     img_names=None,
                     **kwargs):
    """
    Berechnet Accuracy, AUC, EER. Robust gegen Single-Class und NaN/Inf.
    Zusätzliche, unbekannte Argumente (z. B. aus test.py) werden ignoriert.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    # NaN/Inf in Vorhersagen herausfiltern
    finite = np.isfinite(y_pred)
    if not finite.all():
        y_pred = y_pred[finite]
        y_true = y_true[finite]

    # Schutz: leere Arrays oder nur eine Klasse -> ROC/EER nicht definiert
    uniq = np.unique(y_true)
    if (y_pred.size == 0) or (uniq.size < 2):
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int))) if y_pred.size else float('nan')
        return {
            "acc": acc,
            "auc": float('nan'),
            "eer": float('nan'),
            "note": "Single-class oder leere Vorhersagen; ROC/AUC/EER nicht definiert."
        }

    # Standardfall
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred, drop_intermediate=False)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float(fpr[idx])
        auc = float(roc_auc_score(y_true, y_pred))
    except Exception as e:
        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        return {
            "acc": acc,
            "auc": float('nan'),
            "eer": float('nan'),
            "note": f"ROC/EER fehlgeschlagen: {e}"
        }

    acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    return {"acc": acc, "auc": auc, "eer": eer}

