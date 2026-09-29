"""Config-driven rule engine.
Rules live in config/rules_<level>.json, so an analyst can add, edit, disable or re-weight them in developer mode
without touching Python.  A condition reads   feature  operator  value   where value is a number or pNN (a percentile
of the training population).  Several conditions on one rule are combined with AND."""
import copy
import json
import operator
import re

import numpy as np
import pandas as pd

from ..config import CONFIGS
from .default_rules import DEFAULTS

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne}
_COND = re.compile(r"^\s*([A-Za-z_][\w]*)\s*(>=|<=|==|!=|>|<)\s*(p?)(-?\d+(?:\.\d+)?)\s*$")


# ---------------------------------------------------------------- text <-> conditions
def cond_to_text(c):
    v = f"p{c['value']:g}" if c["mode"] == "pctl" else f"{c['value']:g}"
    return f"{c['feature']} {c['op']} {v}"


def rule_text(rule):
    return " AND ".join(cond_to_text(c) for c in rule["conditions"])


def parse_text(text):
    """'amt_sum > p95 AND n_claims > 30'  ->  list of condition dicts.  Raises ValueError on bad syntax."""
    parts = [p for p in re.split(r"\s+AND\s+", text.strip(), flags=re.I) if p.strip()]
    if not parts:
        raise ValueError("Condition is empty")
    out = []
    for p in parts:
        m = _COND.match(p)
        if not m:
            raise ValueError(f"Cannot read '{p}'. Use: feature > 5   or   feature > p95   (join with AND)")
        f, op, pct, val = m.groups()
        val = float(val)
        if pct and not 0 <= val <= 100:
            raise ValueError(f"Percentile must be between 0 and 100 in '{p}'")
        out.append({"feature": f, "op": op, "mode": "pctl" if pct else "abs", "value": val})
    return out


# ---------------------------------------------------------------- load / save
def rules_path(level):
    return CONFIGS / f"rules_{level}.json"


def load_rules(level):
    p = rules_path(level)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return copy.deepcopy(DEFAULTS.get(level, []))


def save_rules(level, rules):
    CONFIGS.mkdir(exist_ok=True)
    rules_path(level).write_text(json.dumps(rules, indent=2), encoding="utf-8")


def reset_rules(level):
    save_rules(level, copy.deepcopy(DEFAULTS.get(level, [])))
    return load_rules(level)


def validate(rules, features):
    """Returns a list of human-readable problems (empty list = fine)."""
    problems, seen = [], set()
    for r in rules:
        if r["id"] in seen:
            problems.append(f"{r['id']}: duplicate id")
        seen.add(r["id"])
        if not r.get("conditions"):
            problems.append(f"{r['id']}: no conditions")
        for c in r.get("conditions", []):
            if c["feature"] not in features:
                problems.append(f"{r['id']}: unknown feature '{c['feature']}'")
            if c["op"] not in OPS:
                problems.append(f"{r['id']}: bad operator '{c['op']}'")
    return problems


# ---------------------------------------------------------------- thresholds and evaluation
def make_ref_quantiles(X: pd.DataFrame):
    """101 percentile points per feature, learned on training data only (labels are never used)."""
    return {f: np.percentile(X[f].to_numpy(dtype=float), np.arange(101)).tolist() for f in X.columns}


def threshold(cond, ref):
    if cond["mode"] == "abs":
        return float(cond["value"])
    q = ref.get(cond["feature"])
    return float(np.interp(cond["value"], np.arange(101), q)) if q else np.nan


def percentile_of(value, q):
    return float(np.interp(value, q, np.arange(101))) if q else np.nan


def evaluate(rules, X: pd.DataFrame, ref):
    """0/1 matrix, one column per ENABLED rule.  Rules that mention a feature missing from X never fire."""
    out = {}
    for r in rules:
        if not r.get("enabled", True):
            continue
        ok = np.ones(len(X), dtype=bool)
        for c in r["conditions"]:
            if c["feature"] not in X.columns:
                ok[:] = False
                break
            ok &= OPS[c["op"]](X[c["feature"]].to_numpy(dtype=float), threshold(c, ref))
        out[r["id"]] = ok.astype(int)
    return pd.DataFrame(out, index=X.index)


def rule_score(flags: pd.DataFrame, rules, cap=4.0):
    """Weighted share of the rule budget used, 0..1.  A case whose fired rules add up to `cap` severity scores 1."""
    sev = {r["id"]: float(r.get("severity", 1.0)) for r in rules}
    if flags.shape[1] == 0:
        return pd.Series(0.0, index=flags.index)
    w = np.array([sev[c] for c in flags.columns])
    return pd.Series(np.clip(flags.to_numpy() @ w / cap, 0, 1), index=flags.index)


def hard_stops(flags: pd.DataFrame, rules):
    ids = [r["id"] for r in rules if r.get("hard_stop") and r["id"] in flags.columns]
    return flags[ids].any(axis=1) if ids else pd.Series(False, index=flags.index)


def fired_names(flags_row, rules):
    names = {r["id"]: r["name"] for r in rules}
    return ", ".join(f"{i} {names[i]}" for i, v in flags_row.items() if v)


def fired_categories(flags_row, rules):
    cats = {r["id"]: r.get("category", "") for r in rules}
    got = sorted({cats[i] for i, v in flags_row.items() if v})
    return ", ".join(got)


def rules_table(rules) -> pd.DataFrame:
    return pd.DataFrame([{"enabled": r.get("enabled", True), "id": r["id"], "name": r["name"], "category": r.get("category", ""),
                          "severity": r.get("severity", 1.0), "hard_stop": r.get("hard_stop", False),
                          "condition": rule_text(r), "description": r.get("description", "")} for r in rules])


def rules_from_table(df: pd.DataFrame):
    out = []
    for _, row in df.iterrows():
        if not str(row["id"]).strip():
            continue
        out.append({"id": str(row["id"]).strip(), "name": str(row["name"]), "category": "" if pd.isna(row["category"]) else str(row["category"]),
                    "description": "" if pd.isna(row.get("description")) else str(row.get("description")),
                    "severity": 1.0 if pd.isna(row["severity"]) else float(row["severity"]),
                    "enabled": True if pd.isna(row["enabled"]) else bool(row["enabled"]),
                    "hard_stop": False if pd.isna(row["hard_stop"]) else bool(row["hard_stop"]),
                    "conditions": parse_text(str(row["condition"]))})
    return out


def rule_stats(flags: pd.DataFrame, y, rules) -> pd.DataFrame:
    """How useful each rule was on labelled data: how often it fires, how many fired cases were fraud, and the lift."""
    y = np.asarray(y)
    base = y.mean() if len(y) else 0
    names = {r["id"]: r["name"] for r in rules}
    cats = {r["id"]: r.get("category", "") for r in rules}
    rows = []
    for c in flags.columns:
        f = flags[c].to_numpy().astype(bool)
        n = int(f.sum())
        prec = float(y[f].mean()) if n else np.nan
        rows.append({"rule": c, "name": names[c], "category": cats[c], "fired": n, "fired_rate": n / max(len(y), 1),
                     "precision": prec, "lift": prec / base if n and base else np.nan,
                     "recall": float(y[f].sum() / max(y.sum(), 1))})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def auto_rules(X: pd.DataFrame, y, n=10):
    """For datasets with no hand-written rules: turn the strongest single features into 'top 5% outlier' rules."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    scored = []
    for f in X.columns:
        v = X[f].to_numpy(dtype=float)
        if np.nanstd(v) == 0:
            continue
        try:
            auc = roc_auc_score(y, v)
        except ValueError:
            continue
        scored.append((abs(auc - 0.5), auc, f))
    rules = []
    for i, (_, auc, f) in enumerate(sorted(scored, reverse=True)[:n], 1):
        hi = auc >= 0.5
        rules.append({"id": f"A{i:02d}", "name": f"{'High' if hi else 'Low'} {f}", "category": "Auto (data-driven)",
                      "description": f"Auto-generated: {f} is in the {'top' if hi else 'bottom'} 5% of the training data.",
                      "severity": 1.0, "enabled": True, "hard_stop": False,
                      "conditions": [{"feature": f, "op": ">" if hi else "<", "mode": "pctl", "value": 95.0 if hi else 5.0}]})
    return rules