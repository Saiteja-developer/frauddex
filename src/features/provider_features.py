"""Claims -> one feature row per provider. The same code runs in training and at test time.

Feature families (each maps to a fraud pattern the rule engine and the models look for):
  volume / spikes, amount anomalies, duplicate + cloned claims, identity (deceased / uncovered patients),
  clinical inconsistencies, physician workload, and graph features (shared physicians / patients between providers)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

DIAG = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
CLAIM_COLS = ["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", "InscClaimAmtReimbursed",
              "AttendingPhysician", "OperatingPhysician", "AdmissionDt", "DischargeDt", "DeductibleAmtPaid"] + DIAG + PROC + ["ip"]
BEN_COLS = ["BeneID", "DOB", "DOD", "RenalDiseaseIndicator", "State", "County", "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"]
CHRONIC = ["ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
           "ChronicCond_ObstrPulmonary", "ChronicCond_Depression", "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
           "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke"]


def prepare_claims(ip: pd.DataFrame, op: pd.DataFrame) -> pd.DataFrame:
    """Stack inpatient + outpatient claims into one table with an `ip` flag."""
    ip, op = ip.copy(), op.copy()
    ip["ip"], op["ip"] = 1, 0
    df = pd.concat([ip, op], ignore_index=True)
    for c in CLAIM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CLAIM_COLS]


def prepare_ben(ben: pd.DataFrame) -> pd.DataFrame:
    ben = ben.copy()
    ben["NumChronic"] = sum((ben[c] == 1).astype(int) for c in CHRONIC if c in ben.columns)
    for c in BEN_COLS:
        if c not in ben.columns:
            ben[c] = np.nan
    return ben[BEN_COLS + ["NumChronic"]]


def _graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Provider-provider graph built from shared physicians and shared patients (idea from the GNN paper [22]).
    deg_*    = number of other providers that share at least one physician / patient with this provider
    shared_* = share of this provider's physicians / patients that also work with another provider"""
    out = {}
    for col, nm in [("AttendingPhysician", "phys"), ("BeneID", "bene")]:
        d = df.dropna(subset=[col])[["Provider", col]].drop_duplicates()
        if d.empty:
            continue
        pi, ci = d.Provider.astype("category"), d[col].astype("category")
        M = sp.csr_matrix((np.ones(len(d)), (pi.cat.codes, ci.cat.codes)), shape=(len(pi.cat.categories), len(ci.cat.categories)))
        M = (M > 0).astype(float)
        P = (M @ M.T).tocsr()
        P.setdiag(0); P.eliminate_zeros()
        per_entity = np.asarray(M.sum(axis=0)).ravel()
        n_ent = np.asarray(M.sum(axis=1)).ravel()
        out[f"deg_{nm}"] = pd.Series(np.asarray((P > 0).sum(axis=1)).ravel(), index=pi.cat.categories)
        out[f"shared_{nm}_ratio"] = pd.Series(np.asarray(M @ (per_entity > 1).astype(float)).ravel() / np.maximum(n_ent, 1),
                                              index=pi.cat.categories)
    return pd.DataFrame(out)


def build_provider_features(claims: pd.DataFrame, ben: pd.DataFrame) -> pd.DataFrame:
    df = claims.merge(ben, on="BeneID", how="left")
    for c in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["amt"] = df.InscClaimAmtReimbursed
    df["days"] = (df.ClaimEndDt - df.ClaimStartDt).dt.days.clip(lower=0)
    df["stay"] = (df.DischargeDt - df.AdmissionDt).dt.days
    df["age"] = (df.ClaimStartDt - df.DOB).dt.days / 365.25
    df["dead"] = df.DOD.notna().astype(int)
    df["after_death"] = (df.ClaimStartDt > df.DOD).astype(int)
    df["ndiag"] = df[DIAG].notna().sum(axis=1)
    df["nproc"] = df[PROC].notna().sum(axis=1)
    df["renal"] = (df.RenalDiseaseIndicator.astype(str) == "Y").astype(int)
    df["same_phys"] = ((df.AttendingPhysician == df.OperatingPhysician) & df.AttendingPhysician.notna()).astype(int)
    df["dup"] = df.duplicated(["BeneID", "Provider", "ClaimStartDt", "amt"], keep=False).astype(int)
    df["bene_prov_n"] = df.groupby(["Provider", "BeneID"]).ClaimID.transform("count")
    df["month"] = df.ClaimStartDt.dt.to_period("M")
    df["week"] = df.ClaimStartDt.dt.to_period("W")
    df["round100"] = ((df.amt % 100) == 0).astype(int)
    dx1 = df["ClmDiagnosisCode_1"]
    df["clone"] = (df.duplicated(["Provider", "ClmDiagnosisCode_1", "ClmProcedureCode_1", "amt"], keep=False) & dx1.notna()).astype(int)
    df["zero_stay"] = ((df.ip == 1) & (df.stay == 0)).astype(int)
    df["nocov"] = (df.NoOfMonths_PartBCov == 0).astype(int)
    df["same_day"] = df.groupby(["Provider", "BeneID", "ClaimStartDt"]).ClaimID.transform("count")
    df["att_day"] = df.groupby(["AttendingPhysician", "ClaimStartDt"]).ClaimID.transform("count")
    df["amt_per_diag"] = df.amt / df.ndiag.clip(lower=1)

    g = df.groupby("Provider")
    F = pd.DataFrame({
        "n_claims": g.ClaimID.count(), "n_benes": g.BeneID.nunique(),
        "n_att": g.AttendingPhysician.nunique(), "n_opr": g.OperatingPhysician.nunique(),
        "ip_ratio": g.ip.mean(), "amt_mean": g.amt.mean(), "amt_med": g.amt.median(), "amt_std": g.amt.std(),
        "amt_max": g.amt.max(), "amt_sum": g.amt.sum(), "ded_mean": g.DeductibleAmtPaid.mean(),
        "days_mean": g.days.mean(), "stay_mean": g.stay.mean(), "age_mean": g.age.mean(),
        "ndiag_mean": g.ndiag.mean(), "nproc_mean": g.nproc.mean(), "chronic_mean": g.NumChronic.mean(),
        "renal_mean": g.renal.mean(), "dead_ratio": g.dead.mean(), "after_death": g.after_death.sum(),
        "dup_ratio": g.dup.mean(), "same_phys": g.same_phys.mean(), "claims_per_bene": g.bene_prov_n.mean(),
        "max_claims_one_bene": g.bene_prov_n.max(), "n_states": g.State.nunique(), "n_counties": g.County.nunique(),
        "round_amt_ratio": g.round100.mean(), "clone_ratio": g.clone.mean(),
        "zero_stay_ratio": g.zero_stay.sum() / g.ip.sum().clip(lower=1), "nocov_ratio": g.nocov.mean(),
        "same_day_repeat": g.same_day.mean(), "max_att_day": g.att_day.max(), "amt_per_diag": g.amt_per_diag.mean(),
        "ip_reimb_mean": g.IPAnnualReimbursementAmt.mean(), "op_reimb_mean": g.OPAnnualReimbursementAmt.mean(),
        "dx_diversity": g.ClmDiagnosisCode_1.nunique() / g.ClaimID.count(),
    })
    F["claims_per_bene_ratio"] = F.n_claims / F.n_benes
    F["claims_per_att"] = F.n_claims / F.n_att.clip(lower=1)
    F["amt_cv"] = F.amt_std / (F.amt_mean + 1)
    F["amt_max_over_med"] = F.amt_max / (F.amt_med + 1)
    mm = df.groupby(["Provider", "month"]).ClaimID.count().groupby("Provider")
    F["month_max"], F["month_mean"] = mm.max(), mm.mean()
    F["spike"] = F.month_max / F.month_mean
    F["burst_ratio"] = df.groupby(["Provider", "week"]).ClaimID.count().groupby("Provider").max() / F.n_claims
    F = F.join(_graph_features(df))
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)