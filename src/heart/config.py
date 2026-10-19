"""Central configuration: paths, column definitions and constants.

Keeping the schema in ONE place means the training pipeline, the API
validation layer and the tests can never drift apart.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_REPO_ID = 45  # id used by the `ucimlrepo` fallback

RANDOM_STATE = 42
TARGET = "target"

# Column order of processed.cleveland.data (no header in the raw file).
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plausible clinical ranges, used for API validation and data sanity checks.
VALID_RANGES = {
    "age": (18, 120),
    "trestbps": (60, 260),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0.0, 10.0),
    "ca": (0, 4),
}

# Allowed values of the categorical features (UCI data dictionary).
VALID_CATEGORIES = {
    "sex": [0, 1],  # 0 = female, 1 = male
    "cp": [1, 2, 3, 4],  # chest pain type
    "fbs": [0, 1],  # fasting blood sugar > 120 mg/dl
    "restecg": [0, 1, 2],  # resting ECG result
    "exang": [0, 1],  # exercise induced angina
    "slope": [1, 2, 3],  # slope of peak exercise ST segment
    "thal": [3, 6, 7],  # 3 normal, 6 fixed defect, 7 reversible defect
}
