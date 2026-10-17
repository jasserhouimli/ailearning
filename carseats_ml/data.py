"""Loading and feature preparation for the Carseats dataset.

The dataset is the classic ``Carseats`` table shipped with the ISLR book: 400
store observations with a numeric ``Sales`` response and several store-level
predictors (price, advertising, shelving location, ...).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_ENV_VAR = "CARSEATS_CSV"
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "carseats.csv"

SALES_COLUMN = "Sales"
TARGET_COLUMN = "High"
HIGH_SALES_THRESHOLD = 8.0
RANDOM_STATE = 0

#: Columns that only carry the source file's row index.
_ROW_INDEX_COLUMNS = {"rownames", "Unnamed: 0"}


def dataset_path() -> Path:
    """Location of the CSV, overridable through the ``CARSEATS_CSV`` env var."""
    return Path(os.environ.get(DATASET_ENV_VAR, DEFAULT_DATASET))


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Carseats table, dropping the row-index column if present."""
    csv_path = path or dataset_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Carseats dataset not found at {csv_path}. "
            f"Set {DATASET_ENV_VAR} to a different CSV path."
        )
    frame = pd.read_csv(csv_path)
    return frame.drop(columns=[c for c in frame.columns if c in _ROW_INDEX_COLUMNS])


def add_high_target(frame: pd.DataFrame, threshold: float = HIGH_SALES_THRESHOLD) -> pd.DataFrame:
    """Add the binary ``High`` label: 1 when ``Sales`` exceed ``threshold``."""
    labelled = frame.copy()
    labelled[TARGET_COLUMN] = (labelled[SALES_COLUMN] > threshold).astype(int)
    return labelled


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode the predictors and return ``(X, y)``."""
    features = frame.drop(columns=[SALES_COLUMN, TARGET_COLUMN])
    features = pd.get_dummies(features, drop_first=True)
    return features, frame[TARGET_COLUMN]


def prepare(
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
    path: Path | None = None,
) -> dict:
    """Prepare the scaled train/test splits used by every model.

    Returns a dict with the split arrays, the fitted scaler and the feature
    names, so callers never have to reimplement the pipeline.
    """
    features, target = split_features_target(add_high_target(load_raw(path)))

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    return {
        "feature_names": list(features.columns),
        "scaler": scaler,
        "X_train": scaler.fit_transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "X": features,
    }
