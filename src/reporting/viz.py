"""All charts (matplotlib).  The dashboard shows them with st.pyplot and report.py saves the same figures as PNG / HTML."""
import os

import matplotlib

if os.environ.get("FRAUDDEX_SHOW") != "1":            # the terminal demo sets FRAUDDEX_SHOW=1 to open chart windows
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, confusion_matrix, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_curve)

INK, MUTED = "#15223A", "#5A6678"
FAM = {"Linear": "#3A5C8F", "Distance": "#6E9BDB", "Probabilistic": "#8FA3BF", "Tree": "#0C877F", "Tree ensemble": "#0C877F",
       "Boosting": "#6A4CC4", "Neural": "#B9770E", "Rules": "#CF4A3B", "Ensemble": "#15223A", "Hybrid": "#15223A", "Unsupervised": "#9AA6B8"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11})


def _fig(w=7, h=4):
    f, ax = plt.subplots(figsize=(w, h))
    return f, ax


def leaderboard(lb: pd.DataFrame):
    d = lb.sort_values("test_ap")
    f, ax = plt.subplots(1, 2, figsize=(11, max(4, 0.32 * len(d))), sharey=True)
    for a, col, ttl in [(ax[0], "test_ap", "Average precision (higher is better)"), (ax[1], "test_f2", "F2 score (recall weighted)")]:
        a.barh(d.model, d[col], color=[FAM.get(x, MUTED) for x in d.family])
        a.set_title(ttl); a.set_xlim(0, 1)
        for y, v in enumerate(d[col]):
            a.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    f.tight_layout()
    return f


def curves(test: pd.DataFrame, cols: dict):
    """cols: {label: column name in test}"""
    f, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    y = test.y.to_numpy()
    for lab, c in cols.items():
        fp, tp, _ = roc_curve(y, test[c]); ax[0].plot(fp, tp, label=f"{lab} (AUC {auc(fp, tp):.2f})")
        p, r, _ = precision_recall_curve(y, test[c]); ax[1].plot(r, p, label=f"{lab}")
    ax[0].plot([0, 1], [0, 1], "--", color=MUTED, lw=1); ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("Recall"); ax[0].set_title("ROC curve")
    ax[1].axhline(y.mean(), ls="--", color=MUTED, lw=1); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall curve")
    ax[0].legend(fontsize=8, loc="lower right"); ax[1].legend(fontsize=8)
    f.tight_layout()
    return f


def confusion(y, flagged, title="Confusion matrix (test set)"):
    cm = confusion_matrix(y, flagged, labels=[0, 1])
    f, ax = _fig(4.2, 3.8)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"]); ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_title(title); ax.spines[:].set_visible(False)
    f.tight_layout()
    return f


def threshold_curve(y, score, thr):
    ts = np.linspace(0.02, 0.98, 60)
    f, ax = _fig(7, 3.8)
    ax.plot(ts, [precision_score(y, score >= t, zero_division=0) for t in ts], label="Precision")
    ax.plot(ts, [recall_score(y, score >= t) for t in ts], label="Recall")
    ax.plot(ts, [fbeta_score(y, score >= t, beta=2, zero_division=0) for t in ts], label="F2", color=INK, lw=2)
    ax.axvline(thr, ls="--", color="#CF4A3B"); ax.text(thr + 0.01, 0.03, f"chosen {thr:.2f}", color="#CF4A3B", fontsize=8)
    ax.set_xlabel("Decision threshold"); ax.set_title("What the threshold trades off"); ax.legend(fontsize=8)
    f.tight_layout()
    return f


def score_dist(y, score, thr):
    f, ax = _fig(7, 3.8)
    bins = np.linspace(0, 1, 31)
    ax.hist(score[y == 0], bins, alpha=.75, label="Legitimate", color="#3A5C8F")
    ax.hist(score[y == 1], bins, alpha=.8, label="Fraud", color="#CF4A3B")
    ax.axvline(thr, ls="--", color=INK); ax.set_yscale("log"); ax.set_xlabel("Final risk score"); ax.set_ylabel("Cases (log scale)")
    ax.set_title("Risk scores: fraud versus legitimate"); ax.legend(fontsize=8)
    f.tight_layout()
    return f


def importance(imp: pd.DataFrame, n=15):
    d = imp.head(n).iloc[::-1]
    f, ax = _fig(6.5, max(3, 0.3 * len(d) + 1))
    ax.barh(d.feature, d.importance, color="#6A4CC4"); ax.set_title("Most important features"); ax.set_xlabel("Drop in average precision when shuffled")
    f.tight_layout()
    return f


def rule_lift(stats: pd.DataFrame, n=18):
    d = stats[stats.fired >= 3].dropna(subset=["lift"]).sort_values("lift").tail(n)
    f, ax = _fig(8, max(3, 0.3 * len(d) + 1))
    ax.barh([f"{r.rule} {r['name'][:32]}" for _, r in d.iterrows()], d.lift, color="#B9770E")
    ax.axvline(1, ls="--", color=MUTED); ax.set_xlabel("Lift (1 = no better than chance)")
    ax.set_title("Which rules pay off")
    f.tight_layout()
    return f


def calibration(y, p):
    fp, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
    f, ax = _fig(4.5, 4)
    ax.plot([0, 1], [0, 1], "--", color=MUTED); ax.plot(mp, fp, "o-", color=INK)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraud rate"); ax.set_title("Calibration")
    f.tight_layout()
    return f


# ------------------------------------------------------------------ dashboard charts
def risk_hist(risk, thr):
    f, ax = _fig(6.5, 3.4)
    ax.hist(risk, bins=np.linspace(0, 1, 26), color="#3A5C8F"); ax.axvline(thr, ls="--", color="#CF4A3B")
    ax.set_yscale("log"); ax.set_xlabel("Final risk"); ax.set_ylabel("Count (log)"); ax.set_title("Distribution of risk scores")
    f.tight_layout()
    return f


def top_bar(df, label_col, value_col, thr=None, n=20, title="Highest-risk cases"):
    d = df.sort_values(value_col, ascending=False).head(n).iloc[::-1]
    f, ax = _fig(6.5, max(3, 0.28 * len(d) + 1))
    ax.barh(d[label_col].astype(str), d[value_col], color=["#CF4A3B" if (thr is not None and v >= thr) else "#8FA3BF" for v in d[value_col]])
    ax.set_title(title); ax.set_xlim(0, 1)
    f.tight_layout()
    return f


def rule_freq(flags: pd.DataFrame, rules, n=20):
    names = {r["id"]: r["name"] for r in rules}
    s = flags.sum().sort_values().tail(n)
    f, ax = _fig(7, max(3, 0.28 * len(s) + 1))
    ax.barh([f"{i} {names[i][:34]}" for i in s.index], s.values, color="#B9770E"); ax.set_title("Rules fired most often"); ax.set_xlabel("Cases")
    f.tight_layout()
    return f


def category_bar(flags: pd.DataFrame, rules):
    cats = {r["id"]: r.get("category", "") for r in rules}
    s = flags.sum().groupby(lambda c: cats[c]).sum().sort_values()
    f, ax = _fig(6.5, max(2.8, 0.4 * len(s) + 1))
    ax.barh(s.index, s.values, color="#0C877F"); ax.set_title("Fraud patterns by category"); ax.set_xlabel("Rule hits")
    f.tight_layout()
    return f


def scatter(F, risk, x, y, thr):
    f, ax = _fig(6.5, 4.4)
    sc = ax.scatter(F[x].clip(lower=1e-3), F[y].clip(lower=1e-3), c=risk, cmap="coolwarm", s=14, alpha=.8, vmin=0, vmax=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}, coloured by risk")
    f.colorbar(sc, ax=ax, label="Final risk")
    f.tight_layout()
    return f


def explain(expl: pd.DataFrame):
    d = expl.iloc[::-1]
    f, ax = _fig(6.5, max(2.8, 0.42 * len(d) + 1))
    ax.barh([f"{r.feature}  ({r.value:,.3g} vs typical {r.typical:,.3g})" for _, r in d.iterrows()], d.effect,
            color=["#CF4A3B" if e > 0 else "#3A5C8F" for e in d.effect])
    ax.set_title("Why this case scores high (effect on fraud probability)"); ax.set_xlabel("Probability points added by this feature")
    f.tight_layout()
    return f


def peer(expl_all: pd.DataFrame, n=12):
    """expl_all: rows with feature, percentile.  Shows where the case sits against the training population."""
    d = expl_all.sort_values("percentile").tail(n)
    f, ax = _fig(6.5, max(2.8, 0.3 * len(d) + 1))
    ax.barh(d.feature, d.percentile, color="#6A4CC4"); ax.axvline(95, ls="--", color="#CF4A3B"); ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile among training providers"); ax.set_title("Where this case stands against peers")
    f.tight_layout()
    return f


def monthly(s: pd.Series, title="Claims per month"):
    f, ax = _fig(6.5, 3.2)
    ax.plot(s.index.astype(str), s.values, marker="o", color="#3A5C8F"); ax.set_title(title)
    ax.tick_params(axis="x", rotation=60); ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    f.tight_layout()
    return f


# ------------------------------------------------------------------ terminal demo extras
def class_balance(items):
    """items: {label: (n_legitimate, n_fraud)}"""
    f, ax = _fig(6.5, 3.2)
    names = list(items)
    fr = [items[k][1] / max(sum(items[k]), 1) * 100 for k in names]
    ax.barh(names, fr, color="#CF4A3B")
    for i, v in enumerate(fr):
        ax.text(v + 0.2, i, f"{v:.1f}% fraud  ({items[names[i]][1]:,} of {sum(items[names[i]]):,})", va="center", fontsize=9)
    ax.set_xlim(0, max(fr) * 1.8 + 1); ax.set_xlabel("Share of fraud in the training data (%)"); ax.set_title("Fraud is rare: the class-imbalance problem")
    f.tight_layout()
    return f


def risk_compare(rows):
    """rows: [(title, risk, threshold, flagged)]"""
    f, ax = _fig(8, max(2.6, 0.55 * len(rows) + 1))
    t = [r[0][:58] for r in rows][::-1]
    ax.barh(t, [r[1] for r in rows][::-1], color=["#CF4A3B" if r[3] else "#0C877F" for r in rows][::-1])
    ax.axvline(rows[0][2], ls="--", color=INK); ax.text(rows[0][2] + 0.01, -0.45, "review threshold", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Final risk"); ax.set_title("Risk of the example cases")
    f.tight_layout()
    return f


def timeline(df, title="Patient claims over time"):
    """df: service_date, claim_amount, final_risk"""
    d = df.copy(); d["dt"] = pd.to_datetime(d.service_date, errors="coerce"); d = d.dropna(subset=["dt"]).sort_values("dt")
    f, ax = _fig(7.5, 3.4)
    sc = ax.scatter(d.dt, d.claim_amount, c=d.final_risk.fillna(0), cmap="coolwarm", vmin=0, vmax=1, s=70, edgecolor=INK)
    ax.plot(d.dt, d.claim_amount, color=MUTED, lw=1, zorder=0)
    ax.set_ylabel("Claim amount"); ax.set_title(title); f.colorbar(sc, ax=ax, label="Final risk"); f.autofmt_xdate()
    f.tight_layout()
    return f


def image_pair(paths, titles):
    from PIL import Image
    f, ax = plt.subplots(1, len(paths), figsize=(5.2 * len(paths), 4.2))
    ax = np.atleast_1d(ax)
    for a, p, t in zip(ax, paths, titles):
        a.imshow(Image.open(p)); a.set_title(t); a.axis("off")
    f.tight_layout()
    return f