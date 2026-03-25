import pandas as pd
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = BASE_PATH / "data/raw"
PROCESSED_DATA_DIR = BASE_PATH / "data/processed"
INTERMEDIATE_DATA_DIR = BASE_PATH / "data/intermediate"


def load_review():
    df = pd.read_json(RAW_DATA_DIR / "Video_Games.json", lines=True)
    if "image" in df.columns:
        df = df.drop(columns=["image"])
    return df


def load_meta():
    df = pd.read_json(RAW_DATA_DIR / "meta_Video_Games.json", lines=True)
    cols_to_drop = ["imageURL", "imageURLHighRes"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df