import os
import random
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, matthews_corrcoef, average_precision_score)
from pyod.utils.data import precision_n_scores

def evaluate_metrics(y_test, y_pred, y_proba=None, digits=3):
    res = {"Accuracy": round(accuracy_score(y_test, y_pred), digits),
           "Precision": precision_score(y_test, y_pred).round(digits),
           "Recall": recall_score(y_test, y_pred).round(digits),
           "F1": f1_score(y_test, y_pred).round(digits),
           "MCC": round(matthews_corrcoef(y_test, y_pred), ndigits=digits)}
    if y_proba is not None:
        res["AUC_PR"] = average_precision_score(y_test, y_proba).round(digits)
        res["AUC_ROC"] = roc_auc_score(y_test, y_proba).round(digits)
        res["PREC_N_SCORES"] = precision_n_scores(y_test, y_proba).round(digits)
    return res

def set_seed_numpy(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)