"""Unsupervised anomaly detectors (papers [9] Isolation Forest / CBLOF / ECOD, [14] LODA / VAE, [4][7] autoencoders).
They never see labels.  Their scores are added to the supervised models as extra features (semi-supervised idea of [21])
and each one is also reported on its own in the leaderboard."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..config import SEED


def slog(x):
    """Signed log transform: tames heavy-tailed money columns without losing the sign."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class ECODLite:
    """Empirical-CDF outlier score: sum over features of -log(tail probability).  Fast, parameter free (idea of ECOD)."""

    def fit(self, Z):
        self.sorted_ = np.sort(Z, axis=0)
        self.n_ = len(Z)
        return self

    def score(self, Z):
        s = np.zeros(len(Z))
        for j in range(Z.shape[1]):
            cdf = np.searchsorted(self.sorted_[:, j], Z[:, j], side="right") / self.n_
            tail = np.clip(np.minimum(cdf, 1 - cdf + 1 / self.n_), 1 / (self.n_ + 1), 1)
            s += -np.log(tail)
        return s / Z.shape[1]


class AnomalyBank:
    NAMES = ["an_iforest", "an_lof", "an_ecod", "an_ae"]

    def __init__(self, detectors=("iforest", "lof", "ecod", "ae"), max_fit=8000):
        self.detectors, self.max_fit = list(detectors), max_fit

    def fit(self, X: pd.DataFrame):
        self.cols_ = list(X.columns)
        Z = self._z(X, fit=True)
        rng = np.random.RandomState(SEED)
        Zs = Z[rng.choice(len(Z), min(len(Z), self.max_fit), replace=False)]
        if "iforest" in self.detectors:
            self.iforest_ = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Z)
        if "lof" in self.detectors:
            self.lof_ = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Zs)
        if "ecod" in self.detectors:
            self.ecod_ = ECODLite().fit(Z)
        if "ae" in self.detectors:
            d = Z.shape[1]
            hid = (max(4, d // 2), max(2, d // 4), max(4, d // 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ae_ = MLPRegressor(hidden_layer_sizes=hid, activation="relu", max_iter=150, random_state=SEED, learning_rate_init=0.003).fit(Zs, Zs)
        return self

    def _z(self, X, fit=False):
        A = slog(X[self.cols_].to_numpy(dtype=float)) if not fit else slog(X.to_numpy(dtype=float))
        if fit:
            self.scaler_ = StandardScaler().fit(A)
        return np.clip(self.scaler_.transform(A), -8, 8)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._z(X)
        out = {}
        if "iforest" in self.detectors:
            out["an_iforest"] = -self.iforest_.score_samples(Z)
        if "lof" in self.detectors:
            out["an_lof"] = -self.lof_.score_samples(Z)
        if "ecod" in self.detectors:
            out["an_ecod"] = self.ecod_.score(Z)
        if "ae" in self.detectors:
            out["an_ae"] = np.log1p(((Z - self.ae_.predict(Z)) ** 2).mean(axis=1))
        return pd.DataFrame(out, index=X.index)