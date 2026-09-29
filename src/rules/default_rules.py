"""Default rule library. Each rule = one fraud pattern. Analysts can edit all of this in developer mode.
Condition value 'p95' means "the 95th percentile of the training population"; a plain number is an absolute value.
`severity` is the rule's weight in the rule score; `hard_stop` rules always flag the case regardless of the ML score."""


def _r(rid, name, cat, desc, sev, conds, hard=False):
    return {"id": rid, "name": name, "category": cat, "description": desc, "severity": sev, "enabled": True,
            "hard_stop": hard, "conditions": conds}


def c(feature, op, value):
    if isinstance(value, str) and value.startswith("p"):
        return {"feature": feature, "op": op, "mode": "pctl", "value": float(value[1:])}
    return {"feature": feature, "op": op, "mode": "abs", "value": float(value)}


PROVIDER_RULES = [
    # ---- volume and spikes
    _r("P01", "Very high total reimbursement", "Volume and spikes", "Provider bills far more in total than almost all peers.", 1.0, [c("amt_sum", ">", "p95")]),
    _r("P02", "Extreme total reimbursement", "Volume and spikes", "Top 1% of all providers by money billed.", 1.5, [c("amt_sum", ">", "p99")]),
    _r("P03", "Very high claim volume", "Volume and spikes", "Claim count is far above normal capacity for a provider.", 1.0, [c("n_claims", ">", "p97")]),
    _r("P04", "Monthly billing spike", "Volume and spikes", "One month is much busier than the provider's usual month (sudden bust-out billing).", 1.0, [c("spike", ">", "p95"), c("n_claims", ">", "p50")]),
    _r("P05", "Short-window billing burst", "Volume and spikes", "A large share of all claims lands inside a single week.", 1.0, [c("burst_ratio", ">", "p90"), c("n_claims", ">", "p25")]),
    _r("P06", "Many claims per patient", "Volume and spikes", "Few patients generate many claims (repeat billing or ghost patients).", 1.0, [c("claims_per_bene_ratio", ">", "p95")]),
    _r("P07", "Single patient billed repeatedly", "Volume and spikes", "One patient appears an unusual number of times.", 1.0, [c("max_claims_one_bene", ">", "p99")]),
    # ---- amount anomalies and upcoding
    _r("P08", "High average claim", "Amount anomalies and upcoding", "Average claim is in the top 5% (possible upcoding to costlier codes).", 1.0, [c("amt_mean", ">", "p95")]),
    _r("P09", "Extreme single claim", "Amount anomalies and upcoding", "At least one claim is in the top 1% for size.", 1.0, [c("amt_max", ">", "p99")]),
    _r("P10", "Heavy-tail amounts", "Amount anomalies and upcoding", "Largest claim is many times the provider's typical claim.", 1.0, [c("amt_max_over_med", ">", "p95")]),
    _r("P11", "Highly variable amounts", "Amount anomalies and upcoding", "Claim amounts swing widely, unlike consistent legitimate billing.", 0.75, [c("amt_cv", ">", "p95")]),
    _r("P12", "High cost per diagnosis", "Amount anomalies and upcoding", "Lots of money per recorded diagnosis (weak clinical justification).", 1.0, [c("amt_per_diag", ">", "p95")]),
    _r("P13", "Round-number billing", "Amount anomalies and upcoding", "Amounts are often exact multiples of 100, which suggests estimated rather than real charges.", 0.75, [c("round_amt_ratio", ">=", "p95"), c("n_claims", ">", "p50")]),
    # ---- duplicates and cloned claims
    _r("P14", "Duplicate billing", "Duplicates and cloning", "Same patient, provider, date and amount billed more than once.", 1.25, [c("dup_ratio", ">", "p95")]),
    _r("P15", "Cloned claims", "Duplicates and cloning", "Many claims share the same diagnosis, procedure and amount (copy-paste billing).", 1.5, [c("clone_ratio", ">", "p95")]),
    _r("P16", "Same-day repeat billing", "Duplicates and cloning", "The same patient is billed several times on one day (unbundling).", 1.0, [c("same_day_repeat", ">", "p97")]),
    _r("P17", "Template diagnoses", "Duplicates and cloning", "Very few different diagnoses across many claims, as if filled from a template.", 1.0, [c("dx_diversity", "<", "p5"), c("n_claims", ">", "p50")]),
    # ---- identity and phantom billing
    _r("P18", "Billing after patient death", "Identity and phantom billing", "A claim starts after the patient's recorded date of death.", 2.0, [c("after_death", ">", 0)], hard=True),
    _r("P19", "Many deceased patients", "Identity and phantom billing", "High share of patients who are recorded as deceased.", 1.0, [c("dead_ratio", ">", "p95")]),
    _r("P20", "Patients without coverage", "Identity and phantom billing", "Claims for patients with no outpatient coverage months.", 1.25, [c("nocov_ratio", ">", 0.02)]),
    _r("P21", "Patients from many states", "Identity and phantom billing", "Patients come from an implausible number of states (recruited or stolen identities).", 1.0, [c("n_states", ">", "p95")]),
    _r("P22", "Patients from many counties", "Identity and phantom billing", "Wide geographic spread of patients for one provider.", 0.75, [c("n_counties", ">", "p95")]),
    _r("P23", "High-spend patient base", "Identity and phantom billing", "Patients already have unusually high annual inpatient reimbursement.", 0.75, [c("ip_reimb_mean", ">", "p95")]),
    # ---- clinical inconsistency
    _r("P24", "Admissions with zero-day stay", "Clinical inconsistency", "Inpatient claims where admit and discharge are the same day (outpatient care billed as inpatient).", 1.25, [c("zero_stay_ratio", ">", "p95")]),
    _r("P25", "Unusually high inpatient share", "Clinical inconsistency", "Provider bills inpatient far more often than peers.", 1.0, [c("ip_ratio", ">=", "p95"), c("n_claims", ">", "p50")]),
    _r("P26", "Unusually long stays", "Clinical inconsistency", "Average length of stay is in the top 5%.", 1.0, [c("stay_mean", ">", "p95")]),
    _r("P27", "Too many diagnoses per claim", "Clinical inconsistency", "Diagnosis padding to justify higher payment.", 1.0, [c("ndiag_mean", ">", "p95")]),
    _r("P28", "Too many procedures per claim", "Clinical inconsistency", "Procedure padding or unbundling.", 1.0, [c("nproc_mean", ">", "p95")]),
    _r("P29", "Attending also operating", "Clinical inconsistency", "Same physician listed in both roles far more often than usual.", 0.75, [c("same_phys", ">", "p95")]),
    _r("P30", "High chronic-condition burden", "Clinical inconsistency", "Patients are recorded as sicker than peers, which raises payment.", 0.75, [c("chronic_mean", ">", "p95")]),
    # ---- physician and collusion network
    _r("P31", "Physician overload", "Physician and collusion network", "One physician appears on many claims in a single day, which is not humanly possible.", 1.5, [c("max_att_day", ">", "p97")]),
    _r("P32", "Few physicians, many claims", "Physician and collusion network", "Claims per attending physician are far above normal.", 1.0, [c("claims_per_att", ">", "p95")]),
    _r("P33", "Broad physician sharing", "Physician and collusion network", "Provider shares physicians with many other providers.", 1.0, [c("deg_phys", ">", "p95")]),
    _r("P34", "Broad patient sharing", "Physician and collusion network", "Provider shares patients with very many other providers (referral or recruiting rings).", 1.0, [c("deg_bene", ">", "p95")]),
    _r("P35", "High share of shared physicians", "Physician and collusion network", "Most of the provider's physicians also work elsewhere.", 0.75, [c("shared_phys_ratio", ">=", "p95"), c("n_claims", ">", "p50")]),
    # ---- composite patterns (emerging schemes)
    _r("P36", "Referral-ring signature", "Emerging patterns", "Shares both patients and physicians with many providers at once.", 1.5, [c("deg_bene", ">", "p90"), c("deg_phys", ">", "p90")]),
    _r("P37", "Spike plus cloned claims", "Emerging patterns", "Sudden volume jump combined with copy-paste claims (automated or bot billing).", 2.0, [c("spike", ">", "p90"), c("clone_ratio", ">", "p90")]),
    _r("P38", "High billing plus round amounts", "Emerging patterns", "Large totals made of round-number charges.", 1.25, [c("amt_sum", ">", "p90"), c("round_amt_ratio", ">", "p90")]),
    _r("P39", "Inpatient inflation", "Emerging patterns", "Many inpatient claims that last zero days.", 1.5, [c("ip_ratio", ">", "p90"), c("zero_stay_ratio", ">", "p90")]),
    _r("P40", "Ghost-patient pattern", "Emerging patterns", "Many claims from a small patient pool, with heavy repeat billing.", 1.5, [c("claims_per_bene_ratio", ">", "p90"), c("n_benes", "<", "p50")]),
    _r("P41", "Overloaded physician at high volume", "Emerging patterns", "Physician overload inside a high-volume provider.", 1.5, [c("max_att_day", ">", "p90"), c("n_claims", ">", "p90")]),
    _r("P42", "Costly long stays", "Emerging patterns", "Long stays that are also expensive.", 1.25, [c("stay_mean", ">", "p90"), c("amt_mean", ">", "p90")]),
    _r("P43", "Deceased-patient exposure at volume", "Emerging patterns", "Deceased patients appear in a provider with meaningful volume.", 1.5, [c("dead_ratio", ">", "p90"), c("n_claims", ">", "p75")]),
    _r("P44", "Multi-state high spend", "Emerging patterns", "Wide patient geography together with very high billing.", 1.25, [c("n_states", ">", "p90"), c("amt_sum", ">", "p90")]),
    _r("P45", "Recruited-patient burst", "Emerging patterns", "Weekly burst of claims from patients who are shared with many providers.", 1.25, [c("burst_ratio", ">", "p75"), c("deg_bene", ">", "p75")]),
]

CLAIM_RULES = [
    _r("C01", "Very fast submission", "Timing", "Claim filed within 2 days of the service, a common pattern in rushed fraudulent submissions.", 1.5, [c("lag_days", "<=", 2)]),
    _r("C02", "Amount far above procedure norm", "Amount anomalies", "Claim is far larger than the typical claim for the same procedure.", 1.25, [c("amt_vs_proc", ">", "p95")]),
    _r("C03", "Amount far above diagnosis norm", "Amount anomalies", "Claim is far larger than the typical claim for the same diagnosis.", 1.25, [c("amt_vs_dx", ">", "p95")]),
    _r("C04", "Extreme claim amount", "Amount anomalies", "Claim amount is in the top 1%.", 1.0, [c("amt", ">", "p99")]),
    _r("C05", "High provider monthly volume", "Provider behaviour", "Provider submits an unusually high number of claims per month.", 1.0, [c("prov_monthly", ">", "p95")]),
    _r("C06", "Long stay", "Clinical", "Length of stay is in the top 5%.", 0.75, [c("los", ">", "p95")]),
    _r("C07", "Many prior visits", "Clinical", "Patient has an unusually high number of visits in the last 12 months.", 0.75, [c("prior_visits", ">", "p95")]),
    _r("C08", "Elderly and costly", "Clinical", "Older patient combined with a top-decile amount.", 0.75, [c("age", ">", "p90"), c("amt", ">", "p90")]),
    _r("C09", "Rushed and expensive", "Emerging patterns", "Fast submission and a large amount together.", 1.5, [c("lag_days", "<=", 3), c("amt", ">", "p75")]),
    _r("C10", "High-volume provider, high amount", "Emerging patterns", "Busy provider with claims well above the procedure norm.", 1.25, [c("prov_monthly", ">", "p90"), c("amt_vs_proc", ">", "p90")]),
    _r("C11", "No chronic condition but many visits", "Emerging patterns", "Frequent visits without a chronic condition to explain them.", 1.0, [c("chronic", "==", 0), c("prior_visits", ">", "p90")]),
    _r("C12", "Weekend submission", "Timing", "Claim submitted on a weekend.", 0.5, [c("dow", ">=", 5)]),
]

# Rules about a patient's / policy's / provider's PAST claims. They read numbers computed from the database (src/features/history.py),
# so they use fixed values, not percentiles of a training set.
HISTORY_RULES = [
    _r("H01", "Duplicate claim", "Repeat and duplicate claims", "Same patient, provider, procedure and date already exists in an earlier claim.", 2.0, [c("duplicate_claim", ">=", 1)]),
    _r("H02", "Another claim on the same day", "Repeat and duplicate claims", "The patient already has a claim for the same service date.", 1.25, [c("patient_same_day_claims", ">=", 1)]),
    _r("H03", "Frequent claimant", "Repeat and duplicate claims", "Four or more claims by this patient in the last 30 days.", 1.0, [c("patient_claims_30d", ">=", 4)]),
    _r("H04", "Claim soon after policy start", "Policy timing", "Service happened within 30 days of the policy starting, a classic pattern for pre-existing conditions or staged claims.", 1.25, [c("policy_age_days", ">=", 0), c("policy_age_days", "<=", 30)]),
    _r("H05", "Service before policy start", "Policy timing", "The service date is earlier than the policy start date, so the claim should not be covered.", 2.0, [c("policy_age_days", "<", 0)], hard=True),
    _r("H06", "Many providers in 90 days", "Repeat and duplicate claims", "Patient visited four or more different providers in 90 days (doctor shopping).", 1.0, [c("patient_distinct_providers_90d", ">=", 4)]),
    _r("H07", "Amount far above the patient's own history", "Amount against history", "Claim is at least three times the patient's usual claim.", 1.0, [c("amount_vs_patient_avg", ">=", 3), c("patient_claims_365d", ">=", 3)]),
    _r("H08", "Amount far above the provider's usual claim", "Amount against history", "Claim is at least three times this provider's usual claim.", 1.0, [c("amount_vs_provider_avg", ">=", 3), c("provider_claims_total", ">=", 10)]),
    _r("H09", "Provider volume surge", "Provider behaviour", "Provider claims this month are at least three times its normal monthly count.", 1.0, [c("provider_surge_ratio", ">=", 3), c("provider_claims_30d", ">=", 10)]),
    _r("H10", "Provider often flagged before", "Provider behaviour", "At least 40% of this provider's earlier claims were flagged.", 1.25, [c("provider_flag_rate", ">=", 0.4), c("provider_claims_total", ">=", 10)]),
    _r("H11", "Many claims on one policy", "Repeat and duplicate claims", "Five or more claims on the same policy in 90 days.", 1.0, [c("policy_claims_90d", ">=", 5)]),
    _r("H12", "Submitted before the service", "Timing", "The claim submission date is earlier than the service date.", 1.5, [c("submission_before_service", ">=", 1)]),
]

# Rules about the uploaded supporting documents (see extraction/ and src/cases/crosscheck.py).
DOCUMENT_RULES = [
    _r("D01", "Amount on document differs from amount entered", "Document checks", "The amount printed in the document is not the amount the analyst entered. Altered totals are one of the most common tampering methods.", 2.0, [c("doc_mismatch_amount", ">=", 1)]),
    _r("D02", "Patient or policy details differ", "Document checks", "Patient ID, name or policy number in the document does not match what was entered.", 1.5, [c("doc_mismatch_identity", ">=", 1)]),
    _r("D03", "Dates differ between document and form", "Document checks", "Service or claim date in the document is not the date entered.", 1.0, [c("doc_mismatch_dates", ">=", 1)]),
    _r("D04", "Document shows editing traces", "Document checks", "File metadata shows it was saved by an editing tool or modified after creation.", 1.25, [c("doc_edit_flag", ">=", 1)]),
    _r("D05", "Same document used in another case", "Document checks", "This document (same picture or same text) already appears in a different claim. Re-used paperwork is a common fraud method.", 2.0, [c("doc_reused", ">=", 1)]),
    _r("D06", "AI model doubts the document", "Document checks", "The vision model reported authenticity concerns or several abnormalities.", 1.5, [c("doc_ai_concern", ">=", 0.5)]),
    _r("D07", "Document hard to read", "Document checks", "Low quality or low OCR confidence, so values could not be verified.", 0.5, [c("doc_low_quality", ">=", 1)]),
    _r("D08", "No supporting document", "Document checks", "The claim has no supporting paperwork attached.", 0.4, [c("doc_missing", ">=", 1)]),
]

DEFAULTS = {"provider": PROVIDER_RULES, "claim": CLAIM_RULES, "history": HISTORY_RULES, "document": DOCUMENT_RULES}
