"""Quick proof that the model engine works:   python scripts/selftest.py
Trains a few models on made-up data (about 30 seconds) and prints the result. Nothing is saved."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.modeling.trainer import train_model
from src.rules import engine as RE

rng = np.random.default_rng(0)
n = 1500
X = pd.DataFrame({"amount": rng.lognormal(6, 1, n), "days_to_claim": rng.integers(1, 30, n),
                  "visits": rng.poisson(3, n), "age": rng.integers(20, 90, n)})
score = (X.amount > X.amount.quantile(0.9)) * 1.5 + (X.days_to_claim < 4) * 1.5 + rng.normal(0, 0.7, n)
y = (score > 1.6).astype(int)
print(f"made-up data: {n} cases, {y.mean():.1%} fraud")

rules = RE.auto_rules(X, y, n=4)
model, res = train_model(X, y, name="selftest", level="custom", rules=rules,
                         models=["lr", "dt", "rf", "hgb", "xgb", "lgbm", "cat"], quick=True, log=lambda m: print("  ", m))
hy = res["leaderboard"].query("key == 'hybrid'").iloc[0]
print(f"\nSELF-TEST OK   hybrid: precision {hy.test_precision:.2f}  recall {hy.test_recall:.2f}  F2 {hy.test_f2:.2f}  AUC {hy.test_auc:.2f}")