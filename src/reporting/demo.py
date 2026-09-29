"""Terminal demo: how the model works, step by step, with tables and charts.   python run.py demo

It reads what training saved in models/registry/<name>/ (leaderboard, curves, rule results, held-out cases), so it needs no data files.
Every chart is also saved as a PNG in test/outputs/demo/."""
import json

import pandas as pd

from ..modeling import registry
from ..rules import engine as RE
from . import terminal as T
from . import viz

LEVEL_TEXT = {"provider": "PROVIDER model: is a hospital / clinic billing in a fraudulent way? (learns from every claim a provider submitted)",
              "claim": "CLAIM model: is this single claim fraudulent? (learns from one claim at a time)"}


def _meta(name):
    return json.loads((registry.path(name) / "meta.json").read_text(encoding="utf-8"))


def _drivers(model, X):
    d = model.explain(X, k=4)
    return [f"{r.feature} = {r.value:,.3g} (typical {r.typical:,.3g}) adds {r.effect:.0%}" for r in d.itertuples() if r.effect > 0.005]


def _card(title, model, rules, X, truth, case_id, window, chart=None):
    sc = model.score(X, rules)
    r = sc.iloc[0]
    fl = sc.attrs["flags"].iloc[0]
    names = {x["id"]: x for x in rules}
    T.banner(f"{title}   (case {case_id})", "-")
    print(f" true answer : {'FRAUD' if truth else 'genuine'}          model says : {'FLAGGED' if r.flagged else 'not flagged'}")
    print(f" risk        : {T.bar(r.final_risk, mark=model.threshold)}   ('|' marks the review threshold {model.threshold:.2f})")
    print(f" made of     : model probability {r.ml_probability:.2f}   rule score {r.rule_score:.2f}   ({int(r.n_rules)} rule(s) fired)\n")
    fired = [names[i] for i, v in fl.items() if v]
    T.table(pd.DataFrame([{"rule": x["id"], "what it means": x["name"], "why it matters": x.get("description", "")} for x in fired]),
            "RULES THAT FIRED", max_rows=8, max_col=70)
    d = _drivers(model, X)
    if d:
        print(" WHAT PUSHED THE MODEL'S SCORE UP:")
        T.bullets(d)
        print()
    if chart:
        ex = model.explain(X, k=6)
        T.show(viz.explain(ex), f"{chart}_why", window)
    return r


def _demo_model(level, name, window, pause):
    meta, d = _meta(name), registry.path(name)
    lb, test = pd.read_csv(d / "leaderboard.csv"), pd.read_csv(d / "test_predictions.csv")
    stats, imp = pd.read_csv(d / "rule_stats.csv"), pd.read_csv(d / "feature_importance.csv")
    sample = registry.discovery_sample(name)
    model = registry.load(name)
    rules = RE.load_rules(level)
    m = meta["metrics"]
    T.banner(f"{level.upper()} MODEL  |  {name}\n{LEVEL_TEXT[level]}", "#")

    # ---------------------------------------------------------------- 1. data
    T.step(1, "THE DATA THE MODEL LEARNED FROM",
           "Real fraud is rare, so the model sees far more genuine cases than fraud. The table shows a few of the numbers (features) it works with. "
           "The chart shows how rare fraud is. Because of this we never judge a model by accuracy alone: a model that says 'genuine' every time would be right "
           f"{1 - m['fraud_rate']:.0%} of the time and catch nothing.")
    n_all = m["n_train"] + m["n_test"]
    print(f" {n_all:,} cases: {m['n_train']:,} used to train, {m['n_test']:,} kept aside to test (the model never saw these).  Fraud rate: {m['fraud_rate']:.1%}.\n")
    if sample is not None:
        show_cols = [c for c in sample.columns if not c.startswith("_")][:8] + ["_y"]
        T.table(sample[show_cols].rename(columns={"_y": "is_fraud"}).head(6), f"A FEW CASES (first 8 of {meta['n_features']} features)", digits=2)
    fr = round(m["fraud_rate"] * n_all)
    T.show(viz.class_balance({name: (n_all - fr, fr)}), f"{level}_1_class_balance", window)
    T.pause(pause)

    # ---------------------------------------------------------------- 2. rules
    T.step(2, "RULES: FRAUD PATTERNS WRITTEN AS PLAIN IF-STATEMENTS",
           "Each rule looks for one known fraud pattern. Rules are easy to explain and an analyst can edit them. "
           "The table shows how each rule did on the held-out cases: 'fired' = how often it triggered, 'precision' = how many of those were truly fraud, "
           "'lift' = how many times better than picking cases at random.")
    cats = pd.DataFrame(rules).groupby("category").size().rename("rules").reset_index()
    T.table(cats, f"{len(rules)} RULES BY FRAUD PATTERN", max_rows=12)
    ex = [r for r in rules if r["id"] in stats.sort_values("lift", ascending=False).rule.head(3).tolist()]
    print(" EXAMPLES (best three):")
    for r in ex:
        print(f"   {r['id']}  IF {RE.rule_text(r)}")
        T.say(f"-> {r['name']}: {r.get('description', '')}", 8)
    print()
    T.table(stats.sort_values("lift", ascending=False)[["rule", "name", "fired", "precision", "lift", "recall"]], "BEST RULES ON HELD-OUT CASES", max_rows=8, max_col=40)
    T.show(viz.rule_lift(stats), f"{level}_2_rule_lift", window)
    T.pause(pause)

    # ---------------------------------------------------------------- 3. models
    T.step(3, "MACHINE-LEARNING MODELS: A CONTEST",
           "Many different models from the research papers are trained on the same data and compared on the held-out cases. "
           "'AP' (average precision) says how well a model puts real fraud near the top of its ranking, and 'F2' rewards catching fraud (recall) more than avoiding false alarms. "
           "Unsupervised models get no answers during training; they only look for unusual cases.")
    cols = ["model", "family", "test_precision", "test_recall", "test_f2", "test_auc", "test_ap"]
    T.table(lb.sort_values("test_ap", ascending=False)[cols], "ALL MODELS, BEST FIRST", max_rows=20, max_col=42)
    T.show(viz.leaderboard(lb), f"{level}_3_model_contest", window)
    T.pause(pause)

    # ---------------------------------------------------------------- 4. combine
    T.step(4, "COMBINING RULES AND MODELS",
           "The best models are also tried as a blend (stacking). Whichever is better, the blend or the best single model, becomes the champion, "
           "and its probability is then combined with the rule score. "
           f"Champion here: {m['champion']}. Rule weight {model.weight:.1f} and review threshold {model.threshold:.2f} were chosen on training data only, never on the held-out cases.")
    best_single = lb[lb.key.isin(model.models.keys())].sort_values("test_ap", ascending=False).iloc[0]
    rows = [lb[lb.key == "rules"].iloc[0], best_single, lb[lb.key == "champion_ml"].iloc[0], lb[lb.key == "hybrid"].iloc[0]]
    T.table(pd.DataFrame(rows)[["model", "test_precision", "test_recall", "test_f2", "test_auc"]], "RULES ONLY vs MODELS vs COMBINED", max_col=44)
    cur = {"FraudDex combined": "final", "Model blend only": "ml", "Rules only": "rule_score"}
    T.show(viz.curves(test, cur), f"{level}_4_curves", window)
    print(" How to read the curves: the closer a line is to the top-left (ROC) or top-right (precision-recall), the better.\n")
    T.show(viz.score_dist(test.y.to_numpy(), test.final.to_numpy(), model.threshold), f"{level}_4_score_split", window)
    T.show(viz.threshold_curve(test.y.to_numpy(), test.final.to_numpy(), model.threshold), f"{level}_4_threshold", window)
    tp = int(((test.flagged == 1) & (test.y == 1)).sum()); fn = int(((test.flagged == 0) & (test.y == 1)).sum())
    fp = int(((test.flagged == 1) & (test.y == 0)).sum()); tn = int(((test.flagged == 0) & (test.y == 0)).sum())
    print(f" On the {len(test):,} held-out cases: {tp} frauds caught, {fn} missed, {fp} false alarms, {tn} correctly left alone.\n")
    T.show(viz.confusion(test.y, test.flagged), f"{level}_4_confusion", window)
    T.pause(pause)

    # ---------------------------------------------------------------- 5. why
    T.step(5, "WHAT THE MODEL PAYS ATTENTION TO",
           "Each feature is shuffled in turn. If the model gets much worse, that feature matters. This tells the analyst which numbers drive fraud decisions.")
    T.table(imp.head(10).rename(columns={"importance": "drop in AP when shuffled"}), "MOST IMPORTANT FEATURES", digits=4, max_col=40)
    T.show(viz.importance(imp), f"{level}_5_importance", window)
    T.pause(pause)

    # ---------------------------------------------------------------- 6. cases
    T.step(6, "FOUR REAL CASES THE MODEL NEVER SAW",
           "One fraud it caught, one genuine case it correctly left alone, one false alarm and one fraud it missed. The model is not perfect and it is more useful to see where it fails.")
    if sample is not None and int(sample._test.sum()) > 0:
        s = sample[sample._test == 1].reset_index(drop=True)
        X, y = s.drop(columns=["_y", "_test"]), s["_y"].astype(int)
        sc = model.score(X, rules)
        picks = []
        cand = sc.assign(y=y.values)
        for title, sub, asc in [("A FRAUD THAT WAS CAUGHT", cand[(cand.y == 1) & (cand.flagged == 1)], False),
                                ("A GENUINE CASE LEFT ALONE", cand[(cand.y == 0) & (cand.flagged == 0)], True),
                                ("A FALSE ALARM (genuine but flagged)", cand[(cand.y == 0) & (cand.flagged == 1)], False),
                                ("A FRAUD THAT WAS MISSED", cand[(cand.y == 1) & (cand.flagged == 0)], True)]:
            if len(sub):
                picks.append((title, int(sub.sort_values("final_risk", ascending=asc).index[0])))
        for i, (title, idx) in enumerate(picks):
            _card(title, model, rules, X.iloc[[idx]], int(y.iloc[idx]), idx, window, chart=f"{level}_6_case{i + 1}" if i == 0 else None)
    T.pause(pause)

    # ---------------------------------------------------------------- summary
    T.banner(f"SUMMARY: {level.upper()} MODEL", "-")
    T.table(pd.DataFrame([{"champion": m["champion"], "precision": m["precision"], "recall": m["recall"], "F2": m["f2"], "AUC": m["auc"],
                           "rules-only F2": m["rules_only_f2"]}]), "RESULT ON HELD-OUT CASES")
    print(f" Reports with every chart and table: models/registry/{name}/report.html\n")


def run(window=True, pause=False, only=None):
    act = {k: v for k, v in registry.active().items() if k in ("provider", "claim") and (registry.path(v) / "meta.json").exists()}
    if only:
        act = {k: v for k, v in act.items() if k == only}
    if not act:
        print("No trained model found. Add your data to data/ and run:  python run.py train")
        return
    T.banner("FRAUDDEX  PHASE 1: HOW THE MODEL WORKS", "#")
    T.say("data  ->  features  ->  rules + machine-learning models  ->  blended risk score  ->  explanation")
    if window:
        T.say("Charts open in windows: close each window to continue. They are also saved in test/outputs/demo/. Tables print here.")
    else:
        T.say("Charts are saved as pictures in test/outputs/demo/. Tables print here.")
    T.pause(pause)
    for level, name in act.items():
        _demo_model(level, name, window, pause)
    T.banner("END OF DEMO. Next phases: documents and images, explanations from a knowledge base, analyst dashboard with a case database.", "#")