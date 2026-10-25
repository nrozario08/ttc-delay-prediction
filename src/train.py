"""Train and compare models that predict whether a logged incident becomes a major delay.

    python src/train.py

Writes reports/metrics.json, two figures in reports/figures/ and the best model
to models/best_model.joblib.
"""
import itertools
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CAT_FEATURES, FEATURES, NUM_FEATURES, TARGET, load_data, score_all

SEED = 42
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"


def one_hot():
    # rare stations / codes get lumped into one "infrequent" column
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False)


def linear_preprocessor():
    # for logistic regression: hour is treated as a category so it can learn the rush hour bumps
    return ColumnTransformer([
        ("cat", one_hot(), CAT_FEATURES + ["hour"]),
        ("num", StandardScaler(), ["month", "is_weekend", "is_rush_hour"]),
    ])


def tree_preprocessor():
    return ColumnTransformer([
        ("cat", one_hot(), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])


class CodeRateBaseline:
    """Predicts the historical major-delay rate for each delay code and nothing else.
    Shows how much of the problem the reported cause alone already explains."""

    def fit(self, X, y):
        rates = y.groupby(X["code"]).agg(["mean", "count"])
        self.rates_ = rates["mean"].to_dict()
        self.default_ = float(y.mean())
        return self

    def predict_proba(self, X):
        return X["code"].map(self.rates_).fillna(self.default_).to_numpy()


def best_f1_threshold(y_true, p):
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(thresholds[np.argmax(f1)])


def main():
    train, val, test = load_data()
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xva, yva = val[FEATURES], val[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"major delay rate: train {ytr.mean():.1%}, val {yva.mean():.1%}, test {yte.mean():.1%}")

    results = {}      # name -> {"val": {...}, "test": {...}}
    fitted = {}       # name -> object with predict_proba on a feature frame

    def proba(model, X):
        p = model.predict_proba(X)
        return p if p.ndim == 1 else p[:, 1]

    def record(name, model):
        fitted[name] = model
        results[name] = {"val": score_all(yva, proba(model, Xva)),
                         "test": score_all(yte, proba(model, Xte))}
        r = results[name]
        print(f"{name:<28} val PR-AUC {r['val']['pr_auc']:.3f} | test ROC {r['test']['roc_auc']:.3f}  "
              f"PR {r['test']['pr_auc']:.3f}  top10% captures {r['test']['top10_capture']:.0%}")

    # --- baselines ---
    record("Always the base rate", DummyClassifier(strategy="prior").fit(Xtr, ytr))
    record("Delay code rate only", CodeRateBaseline().fit(Xtr, ytr))

    # --- models ---
    record("Logistic regression", Pipeline([
        ("prep", linear_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, C=0.5)),
    ]).fit(Xtr, ytr))

    record("Random forest", Pipeline([
        ("prep", tree_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED)),
    ]).fit(Xtr, ytr))

    # small grid for gradient boosting, picked on the validation year only
    best_val, best_params = -1, None
    for lr, leaves, l2 in itertools.product([0.05, 0.1], [15, 31], [0.0, 1.0]):
        gb = Pipeline([
            ("prep", tree_preprocessor()),
            ("model", HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
                                                     max_iter=300, early_stopping=True, random_state=SEED)),
        ]).fit(Xtr, ytr)
        score = score_all(yva, proba(gb, Xva))["pr_auc"]
        print(f"  gb lr={lr} leaves={leaves} l2={l2}: val PR-AUC {score:.4f}")
        if score > best_val:
            best_val, best_params = score, dict(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2)
    print("best gradient boosting settings:", best_params)

    record("Gradient boosting", Pipeline([
        ("prep", tree_preprocessor()),
        ("model", HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=SEED, **best_params)),
    ]).fit(Xtr, ytr))

    # --- pick the best by validation PR-AUC and set an alert threshold on validation ---
    best_name = max((n for n in results if n not in ("Always the base rate", "Delay code rate only")),
                    key=lambda n: results[n]["val"]["pr_auc"])
    best = fitted[best_name]
    threshold = best_f1_threshold(yva, proba(best, Xva))
    pte = proba(best, Xte)
    flagged = pte >= threshold
    alert = {
        "model": best_name,
        "threshold": threshold,
        "precision": float(precision_score(yte, flagged)),
        "recall": float(recall_score(yte, flagged)),
        "f1": float(f1_score(yte, flagged)),
        "share_flagged": float(flagged.mean()),
        "base_rate_test": float(yte.mean()),
    }
    print(f"\nbest model: {best_name}, threshold {threshold:.3f} (chosen on validation)")
    print(f"test: precision {alert['precision']:.2f}, recall {alert['recall']:.2f}, "
          f"flags {alert['share_flagged']:.1%} of incidents")

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump({"split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                   "results": results, "alert_rule": alert, "gb_params": best_params}, f, indent=2)
    joblib.dump({"model": best, "name": best_name, "threshold": threshold}, "models/best_model.joblib")

    plot_pr_curves(fitted, Xte, yte, proba)
    plot_calibration(best, best_name, Xte, yte, proba)


def plot_pr_curves(fitted, X, y, proba):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Delay code rate only": GREY, "Logistic regression": GREEN,
              "Random forest": ORANGE, "Gradient boosting": BLUE}
    for name, color in colors.items():
        precision, recall, _ = precision_recall_curve(y, proba(fitted[name], X))
        ax.plot(recall, precision, color=color, lw=2, label=name)
    ax.axhline(y.mean(), color="#c3c2b7", ls="--", lw=1.5, label=f"Base rate ({y.mean():.1%})")
    ax.set_xlabel("Recall (share of major delays caught)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the test period (2025 to Aug 2026)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/figures/precision_recall.png", dpi=150)
    plt.close(fig)


def plot_calibration(model, name, X, y, proba):
    p = proba(model, X)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, max(mean_pred.max(), frac_pos.max())], [0, max(mean_pred.max(), frac_pos.max())],
            ls="--", color="#c3c2b7", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color=BLUE, lw=2, label=name)
    ax.set_xlabel("Predicted probability of a major delay")
    ax.set_ylabel("Actual share that were major delays")
    ax.set_title("Calibration on the test period")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/figures/calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
