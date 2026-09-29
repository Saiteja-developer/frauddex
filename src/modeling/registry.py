"""Model registry: models/registry/<name>/ holds the model, its metrics, leaderboard and figures.  active.json says which
model is used for each level (provider / claim / custom)."""
import json
import shutil

import joblib
import pandas as pd

from ..config import REGISTRY


def _active_file():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    return REGISTRY / "active.json"


def path(name):
    return REGISTRY / name


def save(model, results, figures=None):
    d = path(model.name)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib", compress=3)
    results["leaderboard"].to_csv(d / "leaderboard.csv", index=False)
    results["test"].to_csv(d / "test_predictions.csv", index=False)
    results["rule_stats"].to_csv(d / "rule_stats.csv", index=False)
    results["importance"].to_csv(d / "feature_importance.csv", index=False)
    if "sample" in results:
        results["sample"].to_csv(d / "discovery_sample.csv.gz", index=False)     # used later to look for new fraud patterns
    meta = {"name": model.name, "level": model.level, "created": model.created, "threshold": model.threshold,
            "rule_weight": model.weight, "order": model.order, "n_features": len(model.features),
            "metrics": model.metrics, "info": {k: v for k, v in model.info.items() if k not in ("rules", "prep")}}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    a = json.loads(_active_file().read_text(encoding="utf-8")) if _active_file().exists() else {}
    a.setdefault(model.level, model.name)
    _active_file().write_text(json.dumps(a, indent=2), encoding="utf-8")


def load(name):
    return joblib.load(path(name) / "model.joblib")


def list_models() -> pd.DataFrame:
    rows = []
    if REGISTRY.exists():
        for m in sorted(REGISTRY.glob("*/meta.json")):
            j = json.loads(m.read_text(encoding="utf-8"))
            rows.append({"name": j["name"], "level": j["level"], "created": j["created"], "champion": j["metrics"].get("champion"),
                         "f2": j["metrics"].get("f2"), "recall": j["metrics"].get("recall"), "precision": j["metrics"].get("precision"),
                         "auc": j["metrics"].get("auc"), "n_features": j["n_features"]})
    return pd.DataFrame(rows)


def active():
    f = _active_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def set_active(level, name):
    a = active(); a[level] = name
    _active_file().write_text(json.dumps(a, indent=2), encoding="utf-8")


def delete(name):
    shutil.rmtree(path(name), ignore_errors=True)
    a = {k: v for k, v in active().items() if v != name}
    _active_file().write_text(json.dumps(a, indent=2), encoding="utf-8")


def discovery_sample(name):
    p = path(name) / "discovery_sample.csv.gz"
    return pd.read_csv(p) if p.exists() else None


def leaderboard(name):
    return pd.read_csv(path(name) / "leaderboard.csv")