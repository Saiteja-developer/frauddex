"""Class-imbalance tools (papers [11] SMOTE-ENN, [16] ADASYN, [20] cost-sensitive learning).
Written with numpy / scikit-learn only, so it works with no extra install.  If `imbalanced-learn` is installed,
strategy 'adasyn' and 'smoteenn' use it; otherwise they fall back to SMOTE."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


class SimpleSMOTE:
    def __init__(self, ratio=0.5, k=5, seed=42):
        self.ratio, self.k, self.seed = ratio, k, seed

    def fit_resample(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        pos, neg = X[y == 1], X[y == 0]
        n_new = int(self.ratio * len(neg)) - len(pos)
        if len(pos) < 3 or n_new <= 0:
            return X, y
        sc = StandardScaler().fit(X)
        P = sc.transform(pos)
        k = min(self.k, len(P) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
        neigh = nn.kneighbors(P, return_distance=False)[:, 1:]
        rng = np.random.RandomState(self.seed)
        i = rng.randint(0, len(P), n_new)
        j = neigh[i, rng.randint(0, k, n_new)]
        lam = rng.rand(n_new, 1)
        synth = sc.inverse_transform(P[i] + lam * (P[j] - P[i]))
        return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def make_sampler(strategy, seed=42):
    if strategy in ("adasyn", "smoteenn"):
        try:
            if strategy == "adasyn":
                from imblearn.over_sampling import ADASYN
                return ADASYN(random_state=seed)
            from imblearn.combine import SMOTEENN
            return SMOTEENN(random_state=seed)
        except ImportError:
            pass
    return SimpleSMOTE(seed=seed)


class Resampled(ClassifierMixin, BaseEstimator):
    """Oversample the training part of every fit (also inside cross-validation, so validation folds stay untouched)."""

    def __init__(self, estimator=None, strategy="smote", seed=42):
        self.estimator, self.strategy, self.seed = estimator, strategy, seed

    def fit(self, X, y):
        Xr, yr = make_sampler(self.strategy, self.seed).fit_resample(np.asarray(X, float), np.asarray(y))
        self.est_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class Weighted(ClassifierMixin, BaseEstimator):
    """Give models without a class_weight option balanced sample weights (cost-sensitive learning)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.est_ = clone(self.estimator).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.est_.classes_
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)