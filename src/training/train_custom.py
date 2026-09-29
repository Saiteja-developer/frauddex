"""Train on ANY labelled CSV you bring.
    python -m src.training.train_custom --csv my.csv --label is_fraud --name my_model --drop approved_amount,status --group provider_id --id claim_id
Column types are detected automatically (numbers, dates, categories).  A starter rule set is generated from the data
(config/rules_<name>.json) and can be edited in developer mode."""
import argparse

import pandas as pd

from ..modeling import registry
from ..reporting import report
from ..rules import engine as RE
from ..features.prep import TabularPrep, suggest_ids, suggest_leakage, to_binary
from ..modeling.trainer import train_model


def run(df, label, name, drop=(), id_cols=(), group=None, quick=False, imbalance="class_weight", models=None, log=print):
    y = to_binary(df[label])
    prep = TabularPrep(label, drop, id_cols, group).fit(df)
    X = prep.transform(df)
    rules = RE.auto_rules(X, y, n=10)
    RE.save_rules(name, rules)
    log(f"features: {X.shape[1]}   auto rules: {len(rules)}")
    model, res = train_model(X, y, name=name, level="custom", rules=rules, groups=(df[group].to_numpy() if group else None),
                             quick=quick, imbalance=imbalance, models=models,
                             info={"builder": "tabular", "prep": prep, "label": label, "dropped": list(drop), "ids": list(id_cols), "group": group}, log=log)
    registry.save(model, res)
    report.build(model, res)
    return model, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--drop", default="", help="comma list of columns to exclude (leakage etc.)")
    ap.add_argument("--id", default="", help="comma list of ID columns")
    ap.add_argument("--group", default=None, help="column to split by, e.g. provider id")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--imbalance", default="class_weight", choices=["class_weight", "smote", "adasyn", "smoteenn", "none"])
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    drop = [c for c in a.drop.split(",") if c]
    if not drop:
        drop = suggest_leakage([c for c in df.columns if c != a.label])
        print("auto-dropped possible leakage columns:", drop)
    ids = [c for c in a.id.split(",") if c] or suggest_ids(list(df.columns))
    run(df, a.label, a.name, drop, ids, a.group, a.quick, a.imbalance)