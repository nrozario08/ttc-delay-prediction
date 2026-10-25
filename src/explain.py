"""Look inside the best model: which inputs matter, and where does it do well or badly?

    python src/explain.py       (run src/train.py first)

Writes figures to reports/figures/ and reports/explain.json.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from common import FEATURES, TARGET, capture_at_top, load_data

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"

# nicer names for the chart
LABELS = {
    "code": "Delay code (reported cause)",
    "station": "Station",
    "hour": "Hour of day",
    "line": "Line",
    "direction": "Direction",
    "weekday": "Day of week",
    "month": "Month",
    "is_weekend": "Weekend flag",
    "is_rush_hour": "Rush hour flag",
    "line_incidents_prev_hour": "Incidents on line, last hour",
    "system_incidents_prev_hour": "Incidents on system, last hour",
}


def main():
    saved = joblib.load("models/best_model.joblib")
    model = saved["model"]
    _, _, test = load_data()
    X, y = test[FEATURES], test[TARGET]

    # permutation importance: shuffle one column at a time and see how much the PR-AUC drops
    result = permutation_importance(model, X, y, scoring="average_precision", n_repeats=5,
                                    random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "drop": result.importances_mean, "sd": result.importances_std})
    imp = imp.sort_values("drop", ascending=True)
    print(imp.sort_values("drop", ascending=False).round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh([LABELS[f] for f in imp["feature"]], imp["drop"], xerr=imp["sd"], color=BLUE,
            error_kw={"ecolor": GREY, "lw": 1})
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
    ax.set_title("What the model relies on")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/figures/feature_importance.png", dpi=150)
    plt.close(fig)

    test = test.assign(p=model.predict_proba(X)[:, 1])
    by_hour(test)
    by_code(test)
    by_line(test)

    # Incidents with no direction logged almost never become delays (they are mostly events that
    # never held up a train), so the model gets easy credit for spotting them. Score the harder
    # subset on its own: incidents where a direction was recorded.
    with_dir = test[test["direction"] != "NONE"]
    subset = {
        "n": int(len(with_dir)),
        "base_rate": float(with_dir[TARGET].mean()),
        "roc_auc": float(roc_auc_score(with_dir[TARGET], with_dir["p"])),
        "pr_auc": float(average_precision_score(with_dir[TARGET], with_dir["p"])),
        "capture_top10": float(capture_at_top(with_dir[TARGET], with_dir["p"], 0.10)),
    }
    print("direction recorded only:", {k: round(v, 3) for k, v in subset.items()})

    summary = {
        "direction_recorded_only": subset,
        "importance": imp.sort_values("drop", ascending=False).round(5).to_dict("records"),
        "capture_top5": capture_at_top(y, test["p"], 0.05),
        "capture_top10": capture_at_top(y, test["p"], 0.10),
        "capture_top20": capture_at_top(y, test["p"], 0.20),
    }
    with open("reports/explain.json", "w") as f:
        json.dump(summary, f, indent=2)
    print({k: round(v, 3) for k, v in summary.items() if k.startswith("capture")})


def by_hour(test):
    g = test.groupby("hour").agg(actual=(TARGET, "mean"), predicted=("p", "mean"), n=(TARGET, "size"))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(g.index, g["actual"] * 100, color=BLUE, lw=2, marker="o", ms=4, label="Actual")
    ax.plot(g.index, g["predicted"] * 100, color=ORANGE, lw=2, marker="o", ms=4, label="Model prediction")
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
    ax.set_xticklabels(["12am", "3am", "6am", "9am", "12pm", "3pm", "6pm", "9pm"])
    ax.set_ylabel("% of incidents that became major delays")
    ax.set_title("Major-delay rate by hour, test period")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/figures/risk_by_hour.png", dpi=150)
    plt.close(fig)


def by_code(test, top_n=12):
    codes = test["code"].value_counts().head(top_n).index
    g = (test[test["code"].isin(codes)]
         .groupby("code").agg(actual=(TARGET, "mean"), predicted=("p", "mean"), n=(TARGET, "size"))
         .sort_values("actual"))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pos = np.arange(len(g))
    ax.barh(pos + 0.2, g["actual"] * 100, height=0.4, color=BLUE, label="Actual")
    ax.barh(pos - 0.2, g["predicted"] * 100, height=0.4, color=ORANGE, label="Model prediction")
    ax.set_yticks(pos)
    ax.set_yticklabels(g.index)
    ax.set_xlabel("% of incidents that became major delays")
    ax.set_title(f"The {top_n} most common delay codes, test period")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("reports/figures/risk_by_code.png", dpi=150)
    plt.close(fig)


def by_line(test):
    g = (test.groupby("line").agg(actual=(TARGET, "mean"), predicted=("p", "mean"), n=(TARGET, "size"))
         .query("n >= 500"))
    print(g.round(3))


if __name__ == "__main__":
    main()
