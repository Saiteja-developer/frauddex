"""Static review pack: PNG figures + one self-contained report.html (tables and charts) for every trained model."""
import base64
import io
import json

import matplotlib.pyplot as plt
import pandas as pd

from ..modeling import registry
from . import viz


def _b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return b.getvalue()


def build(model, res):
    d = registry.path(model.name); fd = d / "figures"; fd.mkdir(parents=True, exist_ok=True)
    lb, t = res["leaderboard"], res["test"]
    cols = {"FraudDex hybrid": "final", "ML only": "ml", "Rules only": "rule_score"}
    top = lb[lb.key.isin(model.models.keys())].sort_values("test_ap", ascending=False).key.head(2).tolist()
    for k in top:
        cols[lb[lb.key == k].model.iloc[0]] = f"p_{k}"
    figs = {
        "01_leaderboard": viz.leaderboard(lb), "02_curves": viz.curves(t, cols),
        "03_confusion": viz.confusion(t.y, t.flagged), "04_threshold": viz.threshold_curve(t.y.to_numpy(), t.final.to_numpy(), model.threshold),
        "05_score_distribution": viz.score_dist(t.y.to_numpy(), t.final.to_numpy(), model.threshold),
        "06_calibration": viz.calibration(t.y.to_numpy(), t.ml.to_numpy()),
    }
    if len(res["importance"]):
        figs["07_importance"] = viz.importance(res["importance"])
    if len(res["rule_stats"].dropna(subset=["lift"])):
        figs["08_rule_lift"] = viz.rule_lift(res["rule_stats"])
    png = {k: _b64(f) for k, f in figs.items()}
    for k, b in png.items():
        (fd / f"{k}.png").write_bytes(b)

    def tbl(df, fmt=3):
        return df.round(fmt).to_html(index=False, classes="t", border=0, na_rep="")
    show = ["model", "family", "papers", "cv_ap", "test_precision", "test_recall", "test_f2", "test_auc", "test_ap", "test_fpr"]
    m = model.metrics
    html = ["<html><head><meta charset='utf-8'><title>FraudDex report: %s</title><style>body{font-family:Segoe UI,Arial;max-width:1100px;margin:24px auto;color:#15223A}"
            "table.t{border-collapse:collapse;font-size:13px}table.t td,table.t th{border-bottom:1px solid #ddd;padding:4px 10px;text-align:right}"
            "table.t td:first-child,table.t th:first-child{text-align:left}img{max-width:100%%}.k{display:inline-block;margin:6px 18px 6px 0}"
            ".k b{font-size:22px;display:block}</style></head><body>" % model.name,
            f"<h1>FraudDex: {model.name}</h1><p>Level: {model.level}. Trained {model.created}. Champion: <b>{m['champion']}</b>. "
            f"Imbalance handling: {model.info.get('imbalance')}. Test set: {m['n_test']:,} cases ({m['fraud_rate']:.1%} fraud overall).</p>"]
    for lab, key in [("Precision", "precision"), ("Recall", "recall"), ("F2", "f2"), ("AUC", "auc"), ("Rules-only F2", "rules_only_f2")]:
        html.append(f"<span class='k'><b>{m[key]:.3f}</b>{lab}</span>")
    html.append("<h2>Model leaderboard (held-out test set)</h2>" + tbl(lb[show].sort_values("test_ap", ascending=False)))
    for k, b in png.items():
        html.append(f"<h3>{k[3:].replace('_', ' ').title()}</h3><img src='data:image/png;base64,{base64.b64encode(b).decode()}'>")
    html.append("<h2>Rule performance</h2>" + tbl(res["rule_stats"]))
    html.append("<h2>Feature importance</h2>" + tbl(res["importance"].head(20), 4))
    (d / "report.html").write_text("".join(html), encoding="utf-8")
    return d / "report.html"


def rebuild(name):
    """Regenerate figures + report.html for a saved model from the CSVs in its registry folder."""
    d = registry.path(name)
    model = registry.load(name)
    res = {"leaderboard": pd.read_csv(d / "leaderboard.csv"), "test": pd.read_csv(d / "test_predictions.csv"),
           "rule_stats": pd.read_csv(d / "rule_stats.csv"), "importance": pd.read_csv(d / "feature_importance.csv")}
    return build(model, res)