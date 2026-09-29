"""Paths and settings.  Change behaviour in config/settings.json, not here.

Folders
  config/      settings, rules, field-name dictionary
  data/        YOU add training files here: csv, zip (of pdf / csv / images), pdf, images
  extraction/  turns any file into content (text, fields, image analysis)
  src/         features, rules, models, retrieval (RAG), database, case workflow
  models/      trained models
  test/        test time: analyst case uploads, knowledge base for retrieval, outputs, the database
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TEST_DIR = ROOT / "test"

UNPACKED_DIR = DATA_DIR / "_unpacked"        # made automatically from zip files in data/
EXTRACTED_DIR = DATA_DIR / "_extracted"      # tables made from the files in data/

CASES_DIR = TEST_DIR / "cases"               # files uploaded with each analyst case
RETRIEVAL_DIR = TEST_DIR / "retrieval"       # knowledge base documents (RAG)
PROCESSED_DIR = TEST_DIR / "processed"
OUTPUTS_DIR = TEST_DIR / "outputs"
FORENSICS_DIR = OUTPUTS_DIR / "forensics"    # heat-maps made by the image checks
SAMPLES_DIR = TEST_DIR / "samples"
DB_PATH = Path(os.environ.get("FRAUDDEX_DB", TEST_DIR / "frauddex.db"))   # test-time only

REGISTRY = MODELS_DIR / "registry"
RAG_DIR = MODELS_DIR / "rag"
OCR_MODELS_DIR = MODELS_DIR / "ocr"
CONFIGS = CONFIG_DIR


def load_settings():
    p = CONFIG_DIR / "settings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


SETTINGS = load_settings()
SEED = SETTINGS.get("seed", 42)
DEV_PASS = os.environ.get(SETTINGS.get("developer_passcode_env", "FRAUDDEX_DEV_PASS"), SETTINGS.get("developer_passcode_default", "dev"))


def ensure_dirs():
    for d in (DATA_DIR, CASES_DIR, RETRIEVAL_DIR, PROCESSED_DIR, OUTPUTS_DIR, FORENSICS_DIR, SAMPLES_DIR, REGISTRY, RAG_DIR, OCR_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)