"""Look inside data/ and decide what can be trained from it.  You just drop files there:

  Kaggle provider files   Train_Inpatientdata*, Train_Outpatientdata*, Train_Beneficiarydata*, Train-*.csv (labels)  -> provider model
  claim-level CSV         columns like Claim_Amount, Patient_Age, Provider_ID + an Is_Fraud label                    -> claim model
  any other labelled CSV  one column named is_fraud / fraud / label / target ...                                      -> its own model
  zip / pdf / image files are noticed here and handled in the next phase (extraction)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, SETTINGS

from ..features.claim_features import REQUIRED as CLAIM_REQUIRED
from ..features.prep import suggest_ids, suggest_leakage

SKIP_PARTS = {"_unpacked", "_extracted", "__MACOSX"}
DOC_EXT = {".zip", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Source:
    kind: str                       # provider | claim | custom | documents | unlabelled
    name: str
    files: list = field(default_factory=list)
    label: str = ""
    note: str = ""


def _label_col(cols):
    names = [n.lower() for n in SETTINGS.get("training", {}).get("label_names", [])]
    for c in cols:
        if c.lower() in names:
            return c
    return ""


def scan_data(root=DATA_DIR):
    root = Path(root)
    found, used = [], set()
    files = [f for f in sorted(root.rglob("*")) if f.is_file() and not (set(f.relative_to(root).parts) & SKIP_PARTS) and f.name.lower() != "readme.md"]
    for folder in sorted({f.parent for f in files}):
        here = [f for f in files if f.parent == folder]
        pick = lambda pat: next((f for f in here if re.search(pat, f.name, re.I) and f.suffix.lower() == ".csv"), None)
        ip, op, be, lab = pick(r"^Train_Inpatient"), pick(r"^Train_Outpatient"), pick(r"^Train_Beneficiary"), pick(r"^Train-")
        if ip and op and be and lab:
            found.append(Source("provider", "provider_default", [ip, op, be, lab], "PotentialFraud", "Kaggle provider files"))
            used.update([ip, op, be, lab])
    for f in files:
        if f in used or f.suffix.lower() != ".csv" or f.name.lower().startswith("test") or f.name.lower() in ("labels.csv", "history.csv"):
            continue
        try:
            cols = list(pd.read_csv(f, nrows=3).columns)
        except Exception as e:
            found.append(Source("unlabelled", f.stem, [f], "", f"could not read ({e})"))
            continue
        lab = _label_col(cols)
        if not lab:
            found.append(Source("unlabelled", f.stem, [f], "", "no label column (is_fraud / fraud / label / target), so it cannot be used for training"))
        elif set(CLAIM_REQUIRED) <= set(cols) and lab == "Is_Fraud":
            found.append(Source("claim", "claim_default", [f], lab, "claim-level file"))
        else:
            found.append(Source("custom", re.sub(r"\W+", "_", f.stem.lower()), [f], lab, "generic labelled CSV"))
    docs = [f for f in files if f.suffix.lower() in DOC_EXT]
    if docs:
        found.append(Source("documents", "documents_default", docs, "", f"{len(docs)} zip / pdf / image file(s): read in the extraction phase, not trained yet"))
    return found


def train_all(quick=False, log=print):
    from . import train_claim, train_custom, train_provider
    sources = scan_data()
    if not sources:
        log("data/ is empty. Add your training files and run again.")
        return []
    done = []
    for s in sources:
        log(f"\n=== {s.kind}: {s.name}  ({s.note}) ===")
        if s.kind == "provider":
            ip, op, be, lab = (pd.read_csv(f) for f in s.files)
            train_provider.run(ip, op, be, lab, name=s.name, quick=quick, log=log)
        elif s.kind == "claim":
            train_claim.run(pd.read_csv(s.files[0]), name=s.name, quick=quick, log=log)
        elif s.kind == "custom":
            df = pd.read_csv(s.files[0])
            others = [c for c in df.columns if c != s.label]
            train_custom.run(df, s.label, s.name, suggest_leakage(others), suggest_ids(list(df.columns)), None, quick, log=log)
        else:
            log(f"  skipped: {s.note}")
            continue
        done.append(s)
    return done