"""Turn ANY labelled CSV into a numeric feature table (used when you upload a new dataset in the Model Lab)."""
import re
import warnings

import numpy as np
import pandas as pd

LEAK_HINT = re.compile(r"(approv|status|decision|paid|outcome|result|denied|reject|settle|adjud|reviewed|audit|investigat)", re.I)
ID_HINT = re.compile(r"(^id$|_id$|^id_|claimid|claim_id|uuid|guid)", re.I)


def suggest_leakage(cols):
    """Columns whose names suggest they are only known AFTER a reviewer has decided (they would leak the answer)."""
    return [c for c in cols if LEAK_HINT.search(c)]


def suggest_ids(cols):
    return [c for c in cols if ID_HINT.search(c)]


def to_binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    pos = {"1", "yes", "y", "true", "t", "fraud", "fraudulent", "positive"}
    return s.astype(str).str.strip().str.lower().isin(pos).astype(int)


class TabularPrep:
    def __init__(self, label=None, drop=(), id_cols=(), group=None):
        self.label, self.drop, self.id_cols, self.group = label, list(drop), list(id_cols), group

    def fit(self, df: pd.DataFrame):
        skip = set(self.drop) | set(self.id_cols) | {self.label} | ({self.group} if self.group else set())
        self.num, self.date, self.cat = [], [], []
        self.median, self.cat_map, self.freq = {}, {}, {}
        for c in df.columns:
            if c in skip:
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                self.num.append(c); self.median[c] = float(s.median()) if s.notna().any() else 0.0
            elif pd.api.types.is_bool_dtype(s):
                self.num.append(c); self.median[c] = 0.0
            else:
                d = self._as_date(s)
                if d is not None:
                    self.date.append(c)
                else:
                    self.cat.append(c)
                    vc = s.astype(str).value_counts()
                    self.cat_map[c] = {v: i for i, v in enumerate(sorted(vc.index)) } if len(vc) <= 100 else {}
                    self.freq[c] = (vc / len(s)).to_dict()
        self.date_min = {c: self._as_date(df[c]).min() for c in self.date}
        self.features = self.transform(df).columns.tolist()
        return self

    @staticmethod
    def _as_date(s):
        if not (s.dtype == object or str(s.dtype).startswith("str")):
            return None
        smp = s.dropna().head(300)
        if smp.empty:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = pd.to_datetime(smp, errors="coerce").notna().mean()
        if ok < 0.9:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, errors="coerce")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        miss = [c for c in self.num + self.date + self.cat if c not in df.columns]
        if miss:
            raise ValueError(f"Uploaded file is missing columns the model was trained on: {miss}")
        X = pd.DataFrame(index=df.index)
        for c in self.num:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(self.median[c])
        for c in self.date:
            d = self._as_date(df[c])
            X[f"{c}__month"] = d.dt.month.fillna(0)
            X[f"{c}__dow"] = d.dt.dayofweek.fillna(0)
            X[f"{c}__days"] = (d - self.date_min[c]).dt.days.fillna(0)
        for c in self.cat:
            s = df[c].astype(str)
            X[f"{c}__freq"] = s.map(self.freq[c]).fillna(0)
            if self.cat_map[c]:
                X[f"{c}__code"] = s.map(self.cat_map[c]).fillna(-1)
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)