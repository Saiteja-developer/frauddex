"""Everything needed at test time lives in ONE picklable object (FraudModel): feature pipeline, models, blend,
fusion weight, threshold, and the rules that were active during training."""
import copy
from datetime import datetime

import numpy as np
import pandas as pd

from ..rules import engine as RE


class FeaturePipeline:
    """raw feature table -> model input.  Adds rule flags (base paper [16]) and anomaly scores ([14] [21])."""

    def __init__(self, features, ref_q, medians, rules_snapshot, use_rule_features, bank):
        self.features, self.ref_q, self.medians = list(features), ref_q, medians
        self.rules_snapshot, self.use_rule_features, self.bank = copy.deepcopy(rules_snapshot), use_rule_features, bank

    def clean(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.features).replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.clean(X)
        parts = [X]
        if self.use_rule_features and self.rules_snapshot:
            fl = RE.evaluate(self.rules_snapshot, X, self.ref_q)
            parts += [fl.add_prefix("rf_"), fl.sum(axis=1).rename("rf_count")]
        if self.bank is not None:
            parts.append(self.bank.transform(X))
        return pd.concat(parts, axis=1)


class FraudModel:
    def __init__(self, name, level, pipe, models, order, meta, weight, threshold, score_cap, info, metrics):
        self.name, self.level, self.pipe = name, level, pipe
        self.models, self.order, self.meta = models, order, meta
        self.weight, self.threshold, self.score_cap = weight, threshold, score_cap
        self.info, self.metrics = info, metrics
        self.created = datetime.now().strftime("%Y-%m-%d %H:%M")

    @property
    def features(self):
        return self.pipe.features

    def missing_features(self, X):
        return [f for f in self.features if f not in X.columns]

    def predict_ml(self, X: pd.DataFrame) -> np.ndarray:
        A = self.pipe.transform(X)
        P = np.column_stack([self.models[k].predict_proba(A)[:, 1] for k in self.order])
        return self.meta.predict_proba(P)[:, 1] if self.meta is not None else P[:, 0]

    def score(self, X: pd.DataFrame, rules=None) -> pd.DataFrame:
        """rules: the LIVE rule set (what the analyst edited).  The ML part keeps using the rules it was trained with."""
        rules = rules if rules is not None else self.info.get("rules", [])
        Xc = self.pipe.clean(X)
        ml = self.predict_ml(X)
        flags = RE.evaluate(rules, Xc, self.pipe.ref_q)
        rs = RE.rule_score(flags, rules, self.score_cap).to_numpy()
        hs = RE.hard_stops(flags, rules).to_numpy()
        final = (1 - self.weight) * ml + self.weight * rs
        final = np.where(hs, np.maximum(final, max(self.threshold, 0.9)), final)
        out = pd.DataFrame({"ml_probability": ml.round(4), "rule_score": rs.round(3), "final_risk": final.round(4),
                            "flagged": ((final >= self.threshold) | hs).astype(int), "hard_stop": hs.astype(int),
                            "n_rules": flags.sum(axis=1).to_numpy()}, index=X.index)
        out["rules_fired"] = [RE.fired_names(flags.iloc[i], rules) for i in range(len(flags))]
        out["categories"] = [RE.fired_categories(flags.iloc[i], rules) for i in range(len(flags))]
        out.attrs["flags"] = flags
        return out

    def explain(self, row: pd.DataFrame, k=6) -> pd.DataFrame:
        """Model-agnostic local explanation by occlusion: replace one feature at a time with its typical (median) value and
        see how much the fraud probability drops.  Positive effect = this feature pushed the risk UP."""
        row = self.pipe.clean(row.iloc[[0]])
        base = float(self.predict_ml(row)[0])
        feats = self.features
        M = pd.concat([row] * len(feats), ignore_index=True)
        for i, f in enumerate(feats):
            M.at[i, f] = self.pipe.medians.get(f, 0.0)
        eff = base - self.predict_ml(M)
        d = pd.DataFrame({"feature": feats, "value": row.iloc[0].to_numpy(), "typical": [self.pipe.medians.get(f, 0.0) for f in feats],
                          "effect": eff})
        d["percentile"] = [RE.percentile_of(v, self.pipe.ref_q.get(f)) for f, v in zip(d.feature, d.value)]
        d.attrs["base"] = base
        return d.sort_values("effect", ascending=False).head(k).reset_index(drop=True)