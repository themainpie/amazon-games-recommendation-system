import pandas as pd
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_PATH / "data/raw"
PROCESSED_DATA_DIR = BASE_PATH / "data/processed"
INTERMEDIATE_DATA_DIR = BASE_PATH / "data/intermediate"


def load_review():
    return pd.read_json(RAW_DATA_DIR / "Video_Games.json", lines=True)


def load_meta():
    return pd.read_json(RAW_DATA_DIR / "meta_Video_Games.json", lines=True)