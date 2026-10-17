import pandas as pd
import pytest

from carseats_ml import data


def test_load_raw_drops_row_index_and_keeps_columns():
    frame = data.load_raw()
    assert frame.shape == (400, 11)
    assert "rownames" not in frame.columns
    assert data.SALES_COLUMN in frame.columns


def test_load_raw_fails_with_a_clear_message_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        data.load_raw(tmp_path / "missing.csv")


def test_add_high_target_uses_the_sales_threshold():
    frame = pd.DataFrame({"Sales": [7.9, 8.0, 8.1]})
    labelled = data.add_high_target(frame)
    assert labelled["High"].tolist() == [0, 0, 1]


def test_split_features_target_one_hot_encodes_categoricals():
    features, target = data.split_features_target(data.add_high_target(data.load_raw()))
    assert not features.select_dtypes(include=["object", "category"]).columns.tolist()
    assert {"ShelveLoc_Good", "ShelveLoc_Medium", "Urban_Yes", "US_Yes"} <= set(features.columns)
    assert set(target.unique()) == {0, 1}


def test_prepare_returns_scaled_splits(prepared):
    assert prepared["X_train"].shape == (280, 11)
    assert prepared["X_test"].shape == (120, 11)
    assert len(prepared["feature_names"]) == 11
    assert abs(float(prepared["X_train"].mean())) < 1e-6
    assert abs(float(prepared["X_train"].std()) - 1.0) < 1e-6
