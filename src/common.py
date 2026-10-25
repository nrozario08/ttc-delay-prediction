"""Shared bits: loading the modeling table, the time-based split, and the metrics."""
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DB_PATH = os.path.join("data", "ttc.db")

CAT_FEATURES = ["weekday", "station", "line", "direction", "code"]
NUM_FEATURES = ["hour", "month", "is_weekend", "is_rush_hour",
                "line_incidents_prev_hour", "system_incidents_prev_hour"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "major_delay"

# split by time, so the test set really is "the future" from the model's point of view
TRAIN_LAST_YEAR = 2023     # train: 2019-2023
VAL_YEAR = 2024            # validation: 2024 (used to pick settings and the alert threshold)
                           # test: 2025 through Aug 2026


def load_data():
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM model_data", con)
    train = df[df["year"] <= TRAIN_LAST_YEAR].copy()
    val = df[df["year"] == VAL_YEAR].copy()
    test = df[df["year"] > VAL_YEAR].copy()
    return train, val, test


def score_all(y_true, p):
    """The numbers reported for every model."""
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "top10_capture": float(capture_at_top(y_true, p, 0.10)),
    }


def capture_at_top(y_true, p, share):
    """Share of all major delays that sit inside the riskiest `share` of incidents."""
    y_true = np.asarray(y_true)
    k = int(len(p) * share)
    top = np.argsort(-np.asarray(p))[:k]
    return y_true[top].sum() / y_true.sum()
