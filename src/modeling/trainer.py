"""One training routine for every dataset (provider files, claim file, or any CSV you upload).

Steps:  split -> rule flags + anomaly scores as extra features -> cross-validated model zoo -> blend the best models
        -> pick decision threshold and rule weight on OUT-OF-FOLD data -> score the untouched test set -> save."""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split

from . import model_zoo as Z
from . import registry
from ..rules import engine as RE
from .anomaly import AnomalyBank
from ..config import SEED
from .pipeline import FeaturePipeline, FraudModel

warnings.filterwarnings("ignore")


def _best_threshold(y, p, beta=2.0):
    ts = np.linspace(0.02, 0.98, 97)
    f = [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])


def _m(y, p, t):
    pred = (p >= t).astype(int)
    neg = max((y == 0).sum(), 1)
    return {"test_precision": precision_score(y, pred, zero_division=0), "test_recall": recall_score(y, pred),
            "test_f1": f1_score(y, pred), "test_f2": fbeta_score(y, pred, beta=2, zero_division=0),
            "test_auc": roc_auc_score(y, p), "test_ap": average_precision_score(y, p), "test_fpr": ((pred == 1) & (y == 0)).sum() / neg,
            "threshold": t}


def train_model(X, y, *, name, level, rules, groups=None, models=None, imbalance="class_weight", use_rule_features=True,
                use_anomaly=True, quick=False, cv=3, test_size=0.25, top_k=5, info=None, log=print):
    t_all = time.time()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    if y.sum() < 10:
        raise ValueError("Need at least 10 fraud (positive) examples to train.")
    # ---- split (by group if given, so one provider / customer never sits in both train and test)
    idx = np.arange(len(X))
    if groups is not None:
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=SEED).split(X, y, groups=np.asarray(groups)))
    else:
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    log(f"split: train {len(Xtr):,} ({ytr.mean():.1%} fraud)  test {len(Xte):,} ({yte.mean():.1%} fraud)")

    # ---- rules (thresholds from training data only) and anomaly detectors (label free)
    ref_q = RE.make_ref_quantiles(Xtr)
    medians = Xtr.median().to_dict()
    bad = RE.validate(rules, set(X.columns))
    if bad:
        log("rule warnings: " + "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
    bank = AnomalyBank().fit(Xtr) if use_anomaly else None
    pipe = FeaturePipeline(X.columns, ref_q, medians, rules, use_rule_features, bank)
    Atr, Ate = pipe.transform(Xtr), pipe.transform(Xte)
    log(f"model input: {Xtr.shape[1]} raw features + {Atr.shape[1] - Xtr.shape[1]} rule / anomaly features")

    # ---- cross-validated leaderboard
    keys = [k for k in (models or Z.DEFAULT_MODELS) if k in Z.available()]
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    folds = StratifiedKFold(cv, shuffle=True, random_state=SEED)
    fitted, oof, ptest, rows = {}, {}, {}, []
    for k in keys:
        t0 = time.time()
        try:
            est = Z.build(k, pw, imbalance, quick)
            o = cross_val_predict(est, Atr, ytr, cv=folds, method="predict_proba")[:, 1]
            est = Z.build(k, pw, imbalance, quick).fit(Atr, ytr)
            p = est.predict_proba(Ate)[:, 1]
        except Exception as e:  # one broken model must not stop the run
            log(f"  {Z.ZOO[k]['name']}: skipped ({type(e).__name__}: {e})")
            continue
        t = _best_threshold(ytr, o)
        fitted[k], oof[k], ptest[k] = est, o, p
        rows.append({"model": Z.ZOO[k]["name"], "key": k, "family": Z.ZOO[k]["family"], "papers": Z.ZOO[k]["papers"],
                     "cv_ap": average_precision_score(ytr, o), "cv_auc": roc_auc_score(ytr, o), **_m(yte.values, p, t),
                     "seconds": time.time() - t0})
        log(f"  {Z.ZOO[k]['name']:<42s} cv AP {rows[-1]['cv_ap']:.3f}   test AP {rows[-1]['test_ap']:.3f}   F2 {rows[-1]['test_f2']:.3f}   {rows[-1]['seconds']:.0f}s")
    if not fitted:
        raise RuntimeError("No model could be trained.")

    # ---- blend the best models (stacking on out-of-fold predictions)
    ranked = sorted(fitted, key=lambda k: -average_precision_score(ytr, oof[k]))
    order = ranked[:max(2, min(top_k, len(ranked)))] if len(ranked) > 1 else ranked
    meta, champion = None, ranked[0]
    if len(order) > 1:
        O = np.column_stack([oof[k] for k in order])
        meta_c = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        meta_oof = cross_val_predict(meta_c, O, ytr, cv=folds, method="predict_proba")[:, 1]
        if average_precision_score(ytr, meta_oof) >= average_precision_score(ytr, oof[ranked[0]]):
            meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(O, ytr)
            champ_oof = meta_oof
            champ_test = meta.predict_proba(np.column_stack([ptest[k] for k in order]))[:, 1]
            champion = "blend"
    if champion != "blend":
        order, champ_oof, champ_test = [champion], oof[champion], ptest[champion]
    champ_name = "Blend of " + ", ".join(Z.ZOO[k]["name"].split(" (")[0] for k in order) if champion == "blend" else Z.ZOO[champion]["name"]
    log(f"champion: {champ_name}")

    # ---- rule score, fusion weight and threshold (out-of-fold, no test data)
    fl_tr, fl_te = RE.evaluate(rules, Xtr, ref_q), RE.evaluate(rules, Xte, ref_q)
    cap = 4.0
    rs_tr, rs_te = RE.rule_score(fl_tr, rules, cap).to_numpy(), RE.rule_score(fl_te, rules, cap).to_numpy()
    best = (-1, 0.0, 0.5)
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        f = (1 - w) * champ_oof + w * rs_tr
        for t in np.linspace(0.05, 0.95, 91):
            s = fbeta_score(ytr, (f >= t).astype(int), beta=2, zero_division=0)
            if s > best[0] + 1e-9:
                best = (s, w, float(t))
    _, w, thr = best
    final_te = (1 - w) * champ_test + w * rs_te
    hs = RE.hard_stops(fl_te, rules).to_numpy()
    final_te = np.where(hs, np.maximum(final_te, max(thr, 0.9)), final_te)
    log(f"rule weight {w}, threshold {thr:.2f}")

    # ---- evaluation table on the untouched test set
    r_thr = _best_threshold(ytr, rs_tr)
    lb = pd.DataFrame(rows)
    extra = [{"model": "Rules only", "key": "rules", "family": "Rules", "papers": "[1] [16]", **_m(yte.values, rs_te, r_thr)},
             {"model": "Blend / champion (ML only)", "key": "champion_ml", "family": "Ensemble", "papers": "[18]",
              **_m(yte.values, champ_test, _best_threshold(ytr, champ_oof))},
             {"model": "FraudDex hybrid (ML + rules)", "key": "hybrid", "family": "Hybrid", "papers": "[16]", **_m(yte.values, final_te, thr)}]
    if bank is not None:
        for c in bank.NAMES:
            if c in Ate.columns:
                tt = float(np.quantile(Atr[c], 1 - max(ytr.mean() * 1.5, 0.02)))
                d = {"model": {"an_iforest": "Isolation Forest (unsupervised)", "an_lof": "Local Outlier Factor (unsupervised)",
                               "an_ecod": "ECOD-lite (unsupervised)", "an_ae": "Autoencoder (unsupervised)"}[c],
                     "key": c, "family": "Unsupervised", "papers": {"an_iforest": "[3] [9]", "an_lof": "[9]", "an_ecod": "[9] [14]", "an_ae": "[4] [7]"}[c]}
                pr = (Ate[c] >= tt).astype(int)
                d.update({"test_precision": precision_score(yte, pr, zero_division=0), "test_recall": recall_score(yte, pr), "test_f1": f1_score(yte, pr),
                          "test_f2": fbeta_score(yte, pr, beta=2, zero_division=0), "test_auc": roc_auc_score(yte, Ate[c]),
                          "test_ap": average_precision_score(yte, Ate[c]), "test_fpr": ((pr == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1), "threshold": tt})
                extra.append(d)
    lb = pd.concat([lb, pd.DataFrame(extra)], ignore_index=True)
    hy = extra[2]
    test = pd.DataFrame({"y": yte.values, "ml": champ_test, "rule_score": rs_te, "final": final_te, "flagged": (final_te >= thr).astype(int)},
                        index=Xte.index)
    for k in fitted:
        test[f"p_{k}"] = ptest[k]

    # ---- permutation importance on raw features (through the whole pipeline)
    tmp = FraudModel(name, level, pipe, fitted, order, meta, w, thr, cap, info or {}, {})
    imp = _importance(tmp, Xte, yte)
    stats = RE.rule_stats(fl_te, yte.values, rules)

    metrics = {"champion": champ_name, "precision": hy["test_precision"], "recall": hy["test_recall"], "f1": hy["test_f1"],
               "f2": hy["test_f2"], "auc": hy["test_auc"], "ap": hy["test_ap"], "fpr": hy["test_fpr"], "rules_only_f2": extra[0]["test_f2"],
               "n_train": len(Xtr), "n_test": len(Xte), "fraud_rate": float(y.mean()), "train_seconds": time.time() - t_all}
    info = dict(info or {}); info.update({"rules": rules, "imbalance": imbalance, "use_rule_features": use_rule_features,
                                          "use_anomaly": use_anomaly, "models_tried": keys, "cv": cv})
    model = FraudModel(name, level, pipe, fitted, order, meta, w, thr, cap, info, metrics)
    log(f"done in {time.time() - t_all:.0f}s   hybrid test: precision {hy['test_precision']:.3f}  recall {hy['test_recall']:.3f}  F2 {hy['test_f2']:.3f}  AUC {hy['test_auc']:.3f}")
    # keep a sample of the feature table (with the true answers and a "held out" mark) for the demo and for finding new rules later
    sample = X.assign(_y=y.to_numpy(), _test=X.index.isin(Xte.index).astype(int))
    sample = sample.sample(min(len(sample), 6000), random_state=SEED)
    return model, {"leaderboard": lb, "test": test, "rule_stats": stats, "importance": imp, "sample": sample}


def _importance(model, Xte, yte, max_rows=1500, max_feats=40, seed=SEED):
    rng = np.random.RandomState(seed)
    n = min(len(Xte), max_rows)
    ii = rng.choice(len(Xte), n, replace=False)
    Xs, ys = Xte.iloc[ii].reset_index(drop=True), yte.iloc[ii].to_numpy()
    if ys.sum() < 3:
        return pd.DataFrame({"feature": [], "importance": []})
    base = average_precision_score(ys, model.predict_ml(Xs))
    feats = list(Xs.columns)
    if len(feats) > max_feats:
        auc = {f: abs(roc_auc_score(ys, Xs[f]) - 0.5) for f in feats if Xs[f].nunique() > 1}
        feats = sorted(auc, key=auc.get, reverse=True)[:max_feats]
    out = []
    for f in feats:
        drops = []
        for _ in range(2):
            Xp = Xs.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - average_precision_score(ys, model.predict_ml(Xp)))
        out.append((f, float(np.mean(drops))))
    return pd.DataFrame(out, columns=["feature", "importance"]).sort_values("importance", ascending=False)