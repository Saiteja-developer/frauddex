"""python run.py check   ->  shows what is installed and what each missing piece would unlock."""
import importlib
import shutil
import sys


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main():
    print(f"Python {sys.version.split()[0]}\n")
    groups = [
        ("REQUIRED for this phase (training + terminal demo)", [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                                                              ("matplotlib", "matplotlib"), ("joblib", "joblib")]),
        ("OPTIONAL more models", [("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("imblearn", "imbalanced-learn")]),
        ("LATER PHASES (not needed yet)", [("streamlit", "streamlit (analyst dashboard)"), ("pdfplumber", "pdfplumber (read PDFs)"), ("pypdf", "pypdf (PDF fallback)"),
                                           ("PIL", "pillow (read images)"), ("pytesseract", "pytesseract (read scans)"),
                                           ("cv2", "opencv-python-headless"), ("openpyxl", "openpyxl (Excel files)")]),
    ]
    for title, items in groups:
        print(title)
        for mod, pipname in items:
            print(f"  {'OK     ' if has(mod) else 'missing'}  {pipname}")
        print()
    print("Programs (later phases)")
    print(f"  {'OK     ' if shutil.which('tesseract') else 'missing'}  tesseract (needed by pytesseract for scans)")
    print("\nInstall a missing library with:  python -m pip install <name>")
    print("Only the REQUIRED group is needed for this phase.")


if __name__ == "__main__":
    main()