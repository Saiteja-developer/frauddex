"""Run once from the project folder:   python make_folders.py
Creates the FraudDex folders and the empty __init__.py files that Python needs."""
from pathlib import Path

FOLDERS = ["config", "data", "scripts", "models/registry", "test/outputs",
           "src/features", "src/rules", "src/modeling", "src/training", "src/reporting"]
PACKAGES = ["src", "src/features", "src/rules", "src/modeling", "src/training", "src/reporting"]

for f in FOLDERS:
    Path(f).mkdir(parents=True, exist_ok=True)
for p in PACKAGES:
    (Path(p) / "__init__.py").touch()

Path("data/README.md").write_text(
    "Put your TRAINING files in this folder (any sub-folders are fine):\n"
    "  - Kaggle provider files: Train_Inpatientdata*.csv, Train_Outpatientdata*.csv, Train_Beneficiarydata*.csv, Train-*.csv\n"
    "  - claim-level CSV with an Is_Fraud column (healthcare_fraud_detection.csv)\n"
    "  - any other CSV with a label column named is_fraud / fraud / label / target\n"
    "Zip files, PDFs and images are added in a later phase.\n")

print("Folders created:")
for f in FOLDERS:
    print("  ", f)
print("Now copy the files one by one into the places shown in the chat.")