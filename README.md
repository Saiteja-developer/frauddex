# FraudDex: Smart Healthcare Insurance Fraud Detection

A hybrid fraud detection system for health insurance claims. Editable rules and a contest of machine-learning models work together, and every result comes with a plain-language reason.

Major project, GRIET (Gokaraju Rangaraju Institute of Engineering and Technology), Batch A9. Team: `<add names>`. Guide: `<add name>`.

## Status

| Phase | What it adds | State |
|---|---|---|
| 1 | Models, rules, training on the files in `data/`, terminal demo with tables and graphs | Done |
| 2 | Database, analyst workflow (patient ID, name, policy number, claim reports), dashboard, memory of past claims | Planned |
| 3 | Reading zip, PDF and scanned files | Planned |
| 4 | AI image check, RAG explanations, rule proposals approved by an analyst | Planned |

## How it works

![Architecture](docs/frauddex_architecture_updated.png)

![Flow](docs/frauddex_flow_updated.png)

1. Fraud patterns are written as plain rules (for example `amt_sum > p95`), 57 so far, editable without touching code.
2. Fourteen machine-learning models from the surveyed papers, plus four anomaly detectors, are compared on cases they never saw. The best are blended into a champion.
3. The model probability and the rule score are combined into one risk score.
4. Fields decided after review (approved amount, claim status) are removed from the inputs, because they would leak the answer.

## Quick start (Windows PowerShell)

```
python -m pip install -r requirements.txt
python make_folders.py
```

Put your own training files in `data/` (see `data/README.md`), then:

```
python run.py check              what is installed
python run.py scan               what the system found in data/
python run.py train --quick      train (a few minutes); drop --quick for the full run
python run.py demo --pause       walk through how the models and rules work
```

Reports with every table and chart are saved to `models/registry/<name>/report.html`.

## Data

The datasets are not in this repository. Download them yourself and follow their terms of use:

- Healthcare Provider Fraud Detection Analysis (Kaggle): the Train and Train_* files go in `data/provider/`.
- Healthcare Fraud Detection Dataset, claim-level (Kaggle): `healthcare_fraud_detection.csv` goes in `data/claims/`.

## Results (held-out cases the model never saw)

| Model | Precision | Recall | F2 | AUC |
|---|---|---|---|---|
| Claim model | 0.419 | 0.886 | 0.725 | 0.968 |
| Provider model | `<paste yours>` | `<paste yours>` | `<paste yours>` | `<paste yours>` |

## Limitations

- The claim-level dataset is synthetic and one feature (days between service and claim) does most of the work. The provider model is the more realistic result.
- Precision is around 0.4 to 0.5, so an analyst still reviews the flagged cases. This is a decision-support tool, not an automatic accept or reject system.
- Do not use real patient data with this project unless you have permission and appropriate safeguards.

## Project layout

```
config/      settings and editable rules
data/        your training files (not in Git)
src/         features, rules, models, training, reporting
models/      trained models (not in Git)
test/        analyst uploads, knowledge base, outputs (mostly not in Git)
scripts/     environment check and self-test
run.py       the one command you use
```

## Licence

See `LICENSE`.