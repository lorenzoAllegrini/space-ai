import os
import random
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, matthews_corrcoef, average_precision_score)
from pyod.utils.data import precision_n_scores

def evaluate_metrics(y_test, y_pred, y_proba=None, digits=3):
    res = {
        "Accuracy": round(accuracy_score(y_test, y_pred), digits),
        "Precision": round(precision_score(y_test, y_pred), digits),
        "Recall": round(recall_score(y_test, y_pred), digits),
        "F1": round(f1_score(y_test, y_pred), digits),
        "MCC": round(matthews_corrcoef(y_test, y_pred), digits),
    }
    if y_proba is not None:
        res["AUC_PR"] = round(average_precision_score(y_test, y_proba), digits)
        res["AUC_ROC"] = round(roc_auc_score(y_test, y_proba), digits)
        res["PREC_N_SCORES"] = precision_n_scores(y_test, y_proba).round(digits)
    return res

def set_seed_numpy(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)