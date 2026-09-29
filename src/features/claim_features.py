"""Claim-level feature builder for files shaped like healthcare_fraud_detection.csv.
Post-decision columns (Approved_Amount, Claim_Status) are NEVER used: they only exist after a reviewer decides."""
import numpy as np
import pandas as pd

CAT_COLS = ["Diagnosis_Code", "Procedure_Code", "Insurance_Type", "Provider_Specialty", "Patient_State", "Visit_Type"]
DROP_LEAKAGE = ["Approved_Amount", "Claim_Status"]
NUMERIC = ["Patient_Age", "Claim_Amount", "Days_Between_Service_and_Claim", "Number_of_Claims_Per_Provider_Monthly",
           "Length_of_Stay", "Chronic_Condition_Flag", "Prior_Visits_12m"]
REQUIRED = ["Patient_Age", "Patient_Gender", "Claim_Amount", "Days_Between_Service_and_Claim",
            "Number_of_Claims_Per_Provider_Monthly", "Length_of_Stay", "Chronic_Condition_Flag", "Prior_Visits_12m",
            "Claim_Submission_Date"] + CAT_COLS


class ClaimPrep:
    """Learns category codes and group medians on training data, then encodes any later upload the same way."""

    def __init__(self):
        self.ref = None

    def fit(self, a: pd.DataFrame):
        self.ref = {"cats": {c: sorted(a[c].dropna().astype(str).unique()) for c in CAT_COLS},
                    "proc_med": a.groupby("Procedure_Code").Claim_Amount.median().to_dict(),
                    "dx_med": a.groupby("Diagnosis_Code").Claim_Amount.median().to_dict(),
                    "prior_med": float(a.Prior_Visits_12m.median()), "amt_med": float(a.Claim_Amount.median()),
                    "num_med": {c: float(a[c].median()) for c in NUMERIC if c in a.columns},
                    "cat_mode": {c: str(a[c].mode().iloc[0]) for c in CAT_COLS if c in a.columns}}
        return self

    def check(self, a: pd.DataFrame):
        miss = [c for c in REQUIRED if c not in a.columns]
        if miss:
            raise ValueError(f"Claim file is missing columns: {miss}")

    def transform(self, a: pd.DataFrame, tolerant=False) -> pd.DataFrame:
        """tolerant=True (used for PDFs / scans): missing columns are filled with typical training values instead of failing."""
        if tolerant:
            a = a.copy()
            for c in REQUIRED:
                if c not in a.columns:
                    a[c] = np.nan
            for c, m in self.ref.get("num_med", {}).items():
                a[c] = pd.to_numeric(a[c], errors="coerce").fillna(m)
            for c, m in self.ref.get("cat_mode", {}).items():
                a[c] = a[c].fillna(m)
            a["Patient_Gender"] = a["Patient_Gender"].fillna("Female")
        else:
            self.check(a)
        r = self.ref
        d = pd.to_datetime(a.Claim_Submission_Date, errors="coerce")
        X = pd.DataFrame(index=a.index)
        X["age"] = a.Patient_Age
        X["male"] = (a.Patient_Gender == "Male").astype(int)
        X["amt"] = a.Claim_Amount
        X["lag_days"] = a.Days_Between_Service_and_Claim
        X["lag_le3"] = (a.Days_Between_Service_and_Claim <= 3).astype(int)
        X["prov_monthly"] = a.Number_of_Claims_Per_Provider_Monthly
        X["los"] = a.Length_of_Stay
        X["chronic"] = a.Chronic_Condition_Flag
        X["prior_visits"] = a.Prior_Visits_12m.fillna(r["prior_med"])
        X["month"], X["dow"] = d.dt.month.fillna(0), d.dt.dayofweek.fillna(0)
        for c in CAT_COLS:
            X[c] = a[c].astype(str).map({v: i for i, v in enumerate(r["cats"][c])}).fillna(-1)
        X["amt_vs_proc"] = a.Claim_Amount / a.Procedure_Code.map(r["proc_med"]).fillna(r["amt_med"])
        X["amt_vs_dx"] = a.Claim_Amount / a.Diagnosis_Code.map(r["dx_med"]).fillna(r["amt_med"])
        X["amt_per_stay_day"] = a.Claim_Amount / (a.Length_of_Stay + 1)
        return X.replace([np.inf, -np.inf], np.nan).fillna(0)