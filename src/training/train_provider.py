"""Provider-level model from the Kaggle Healthcare Provider Fraud files. Started by:  python run.py train
Reads CSVs only.  The database is never touched during training."""
import pandas as pd

from ..modeling import registry
from ..reporting import report
from ..rules import engine as RE
from ..features.provider_features import build_provider_features, prepare_ben, prepare_claims
from ..modeling.trainer import train_model


def load_provider_training(ip, op, ben, lab):
    F = build_provider_features(prepare_claims(ip, op), prepare_ben(ben))
    lab = lab.set_index("Provider")
    F = F.join(lab.PotentialFraud, how="inner")
    y = (F.pop("PotentialFraud") == "Yes").astype(int)
    return F, y


def run(ip, op, ben, lab, name="provider_default", quick=False, imbalance="class_weight", models=None, log=print):
    F, y = load_provider_training(ip, op, ben, lab)
    log(f"provider table {F.shape}, fraud rate {y.mean():.3f}")
    model, res = train_model(F, y, name=name, level="provider", rules=RE.load_rules("provider"), quick=quick,
                             imbalance=imbalance, models=models, info={"builder": "provider"}, log=log)
    registry.save(model, res)
    report.build(model, res)
    return model, res