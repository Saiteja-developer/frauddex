"""Model zoo.  Every model below is used in the papers surveyed in Review 2 (paper numbers from the reference list).
XGBoost / LightGBM / CatBoost appear automatically once they are pip-installed."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .anomaly import slog
from ..config import SEED
from .imbalance import Resampled, Weighted


def _scaled(est):
    """Heavy-tailed money columns hurt distance / linear models: signed-log, then standardise."""
    return make_pipeline(FunctionTransformer(slog), StandardScaler(), est)


def _lib(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _f_lr(cw, pw, q): return _scaled(LogisticRegression(max_iter=3000, C=0.3, class_weight=cw))
def _f_svm(cw, pw, q): return _scaled(CalibratedClassifierCV(LinearSVC(C=0.1, class_weight=cw, dual="auto", max_iter=5000), cv=3))
def _f_knn(cw, pw, q): return _scaled(KNeighborsClassifier(25, weights="distance"))
def _f_nb(cw, pw, q): return Weighted(GaussianNB()) if cw else GaussianNB()
def _f_dt(cw, pw, q): return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight=cw, random_state=SEED)
def _f_rf(cw, pw, q): return RandomForestClassifier(150 if q else 400, min_samples_leaf=3, class_weight=("balanced_subsample" if cw else None), n_jobs=-1, random_state=SEED)
def _f_et(cw, pw, q): return ExtraTreesClassifier(150 if q else 400, min_samples_leaf=3, class_weight=("balanced_subsample" if cw else None), n_jobs=-1, random_state=SEED)
def _f_ada(cw, pw, q):
    m = AdaBoostClassifier(n_estimators=100 if q else 200, learning_rate=0.5, random_state=SEED)
    return Weighted(m) if cw else m
def _f_gb(cw, pw, q):
    m = GradientBoostingClassifier(n_estimators=80 if q else 150, max_depth=3, subsample=0.8, random_state=SEED)
    return Weighted(m) if cw else m
def _f_hgb(cw, pw, q): return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150 if q else 300, l2_regularization=1.0, class_weight=cw, random_state=SEED)
def _f_mlp(cw, pw, q):
    m = _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300, early_stopping=True, random_state=SEED))
    return Resampled(m, "smote") if cw else m


def _f_xgb(cw, pw, q):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=150 if q else 300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=pw if cw else 1, eval_metric="logloss", n_jobs=-1, random_state=SEED)


def _f_lgbm(cw, pw, q):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=150 if q else 300, learning_rate=0.05, num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pw if cw else 1, n_jobs=-1, verbose=-1, random_state=SEED)


def _f_cat(cw, pw, q):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=150 if q else 300, depth=5, learning_rate=0.05, auto_class_weights="Balanced" if cw else None, verbose=0, random_seed=SEED)


ZOO = {
    "lr":   dict(name="Logistic Regression", family="Linear", papers="[3] [5]", make=_f_lr, needs=None),
    "svm":  dict(name="Linear SVM (calibrated)", family="Linear", papers="[2] [10]", make=_f_svm, needs=None),
    "knn":  dict(name="K-Nearest Neighbours", family="Distance", papers="[10]", make=_f_knn, needs=None),
    "nb":   dict(name="Naive Bayes", family="Probabilistic", papers="[10]", make=_f_nb, needs=None),
    "dt":   dict(name="Decision Tree", family="Tree", papers="[3] [10]", make=_f_dt, needs=None),
    "rf":   dict(name="Random Forest", family="Tree ensemble", papers="[3] [5] [24]", make=_f_rf, needs=None),
    "et":   dict(name="Extra Trees", family="Tree ensemble", papers="[26]", make=_f_et, needs=None),
    "ada":  dict(name="AdaBoost", family="Boosting", papers="[11] [26]", make=_f_ada, needs=None),
    "gb":   dict(name="Gradient Boosting", family="Boosting", papers="[10]", make=_f_gb, needs=None),
    "hgb":  dict(name="Hist Gradient Boosting (LightGBM-style)", family="Boosting", papers="[11]", make=_f_hgb, needs=None),
    "mlp":  dict(name="Neural Network (MLP)", family="Neural", papers="[4] [12]", make=_f_mlp, needs=None),
    "xgb":  dict(name="XGBoost", family="Boosting", papers="[3] [11] [16]", make=_f_xgb, needs="xgboost"),
    "lgbm": dict(name="LightGBM", family="Boosting", papers="[11] [18]", make=_f_lgbm, needs="lightgbm"),
    "cat":  dict(name="CatBoost", family="Boosting", papers="[18] [25]", make=_f_cat, needs="catboost"),
}
DEFAULT_MODELS = ["lr", "svm", "knn", "nb", "dt", "rf", "et", "ada", "gb", "hgb", "mlp", "xgb", "lgbm", "cat"]


def available():
    return [k for k, v in ZOO.items() if v["needs"] is None or _lib(v["needs"])]


def build(key, pos_weight, imbalance="class_weight", quick=False):
    """imbalance: 'class_weight' (cost-sensitive), 'smote' / 'adasyn' / 'smoteenn' (oversample), or 'none'."""
    cw = "balanced" if imbalance == "class_weight" else None
    est = ZOO[key]["make"](cw, pos_weight, quick)
    if imbalance in ("smote", "adasyn", "smoteenn") and not isinstance(est, Resampled):
        est = Resampled(est, imbalance)
    return est