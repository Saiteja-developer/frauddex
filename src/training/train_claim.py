"""Claim-level model from a claim file shaped like healthcare_fraud_detection.csv. Started by:  python run.py train
Approved_Amount and Claim_Status are dropped: they are decided after review and would leak the answer."""
import pandas as pd

from ..modeling import registry
from ..reporting import report
from ..rules import engine as RE
from ..features.claim_features import DROP_LEAKAGE, ClaimPrep
from ..modeling.trainer import train_model


def run(a: pd.DataFrame, name="claim_default", quick=False, imbalance="class_weight", models=None, log=print):
    prep = ClaimPrep().fit(a)
    X, y = prep.transform(a), a.Is_Fraud
    model, res = train_model(X, y, name=name, level="claim", rules=RE.load_rules("claim"), groups=a.Provider_ID.to_numpy(), quick=quick,
                             imbalance=imbalance, models=models,
                             info={"builder": "claim", "prep": prep, "dropped_for_leakage": DROP_LEAKAGE}, log=log)
    registry.save(model, res)
    report.build(model, res)
    return model, res