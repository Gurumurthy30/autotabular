import io
import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app.db.session import init_db
from app.main import app
from app.tools.mlflow_tools import MLflowTools

client = TestClient(app)


def test_predict_endpoint_flow():
    init_db()

    # 1. Project house_price_predection should exist on disk
    train_csv_path = "projects/house_price_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

    # Ensure model and MLflow run exist
    models_dir = Path("projects/house_price_predection/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_pkl = models_dir / "best_model.pkl"
    if not best_pkl.exists():
        num_cols = train_df.drop(columns=["price", "id"], errors="ignore").select_dtypes(include=["number"]).columns
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", Ridge())])
        pipe.fit(train_df[num_cols], train_df["price"])
        joblib.dump(pipe, best_pkl)

    m_housing = MLflowTools("house_price_predection")
    if not m_housing.get_leaderboard("rmse"):
        m_housing.log_run("baseline_ridge", params={}, metrics={"rmse": 25000.0}, artifact_paths=[str(best_pkl)])

    # Make test dataframe with 10 rows without price
    test_df = train_df.head(10).drop(columns=["price"])

    csv_buf = io.BytesIO()
    test_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue()

    # Query leaderboard to get a valid run ID
    lb_res = client.get("/projects/house_price_predection/models?metric=rmse")
    assert lb_res.status_code == 200
    runs = lb_res.json()["leaderboard"]
    assert len(runs) > 0
    run_id = runs[0]["run_id"]

    # 2. Test valid prediction with explicit id_column
    res = client.post(
        f"/projects/house_price_predection/models/{run_id}/predict",
        files={"test_file": ("test.csv", csv_bytes, "text/csv")},
        data={"id_column": "id"},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")
    assert "submission.csv" in res.headers.get("content-disposition", "")

    # Parse response CSV
    res_df = pd.read_csv(io.StringIO(res.text))
    assert list(res_df.columns) == ["id", "price"]
    assert len(res_df) == 10
    assert not res_df["price"].isna().any()

    # 3. Test valid prediction with auto-detected ID column (id)
    res_auto = client.post(
        f"/projects/house_price_predection/models/{run_id}/predict",
        files={"test_file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert res_auto.status_code == 200, res_auto.text
    res_auto_df = pd.read_csv(io.StringIO(res_auto.text))
    assert list(res_auto_df.columns) == ["id", "price"]
    assert len(res_auto_df) == 10

    # 4. Test missing columns validation error (must return 400 with missing columns named)
    bad_df = pd.DataFrame({"random_column": [1, 2, 3]})
    bad_buf = io.BytesIO()
    bad_df.to_csv(bad_buf, index=False)

    res_missing = client.post(
        f"/projects/house_price_predection/models/{run_id}/predict",
        files={"test_file": ("bad.csv", bad_buf.getvalue(), "text/csv")},
    )
    assert res_missing.status_code == 400
    detail = res_missing.json().get("detail", "")
    assert "Missing required feature column(s)" in detail
    assert "area" in detail or "bedrooms" in detail

    # 5. Test invalid file extension
    res_ext = client.post(
        f"/projects/house_price_predection/models/{run_id}/predict",
        files={"test_file": ("bad.txt", b"some text", "text/plain")},
    )
    assert res_ext.status_code == 400
    assert "CSV" in res_ext.json().get("detail", "")

    # 6. Test empty CSV
    empty_buf = io.BytesIO(b"")
    res_empty = client.post(
        f"/projects/house_price_predection/models/{run_id}/predict",
        files={"test_file": ("empty.csv", empty_buf.getvalue(), "text/csv")},
    )
    assert res_empty.status_code == 400

    # 7. Test non-existent project
    res_404 = client.post(
        f"/projects/non_existent_project_xyz/models/{run_id}/predict",
        files={"test_file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert res_404.status_code == 404


@pytest.mark.skipif(
    not os.path.exists("projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"),
    reason="binary_rainfall_predection dataset not found on disk",
)
def test_predict_raw_data_auto_feature_engineering():
    """Verify that uploading unprocessed raw data auto-applies feature_pipeline.pkl."""
    init_db()

    train_csv_path = "projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

    # Setup model and pipeline if needed
    models_dir = Path("projects/binary_rainfall_predection/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path("projects/binary_rainfall_predection/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    pipe_pkl = features_dir / "feature_pipeline.pkl"
    best_model_pkl = models_dir / "best_model.pkl"
    raw_X = train_df.drop(columns=["rainfall"], errors="ignore")

    from app.ml_harness.preprocess import make_basic_preprocessor

    col_roles = {c: ("target" if c == "rainfall" else ("id" if c in ("id",) else "numeric")) for c in train_df.columns}
    feat_pipe = make_basic_preprocessor({"column_roles": col_roles})
    feat_pipe.fit(raw_X)
    joblib.dump(feat_pipe, pipe_pkl)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(feat_pipe.transform(raw_X), train_df["rainfall"])
    joblib.dump(model, best_model_pkl)

    m_rain = MLflowTools("binary_rainfall_predection")
    run_id = m_rain.log_run("GradientBoosting_test", params={}, metrics={"f1": 0.88}, artifact_paths=[str(best_model_pkl)])

    # 10 raw rows without rainfall target
    raw_test_df = train_df.head(10).drop(columns=["rainfall"])
    csv_buf = io.BytesIO()
    raw_test_df.to_csv(csv_buf, index=False)

    res = client.post(
        f"/projects/binary_rainfall_predection/models/{run_id}/predict",
        files={"test_file": ("raw_test.csv", csv_buf.getvalue(), "text/csv")},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")

    pred_df = pd.read_csv(io.StringIO(res.text))
    assert list(pred_df.columns) == ["id", "rainfall"]
    assert len(pred_df) == 10
    assert not pred_df["rainfall"].isna().any()


@pytest.mark.skipif(
    not os.path.exists("projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"),
    reason="binary_rainfall_predection dataset not found on disk",
)
def test_predict_gradient_boosting_with_nans():
    """Verify that uploading test data with NaNs does not crash GradientBoosting with NaN error."""
    import numpy as np

    init_db()

    train_csv_path = "projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

    models_dir = Path("projects/binary_rainfall_predection/models")
    features_dir = Path("projects/binary_rainfall_predection/features")
    best_model_pkl = models_dir / "best_model.pkl"
    pipe_pkl = features_dir / "feature_pipeline.pkl"

    raw_X = train_df.drop(columns=["rainfall"], errors="ignore")
    from app.ml_harness.preprocess import make_basic_preprocessor
    col_roles = {c: ("target" if c == "rainfall" else ("id" if c in ("id",) else "numeric")) for c in train_df.columns}
    feat_pipe = make_basic_preprocessor({"column_roles": col_roles})
    feat_pipe.fit(raw_X)
    joblib.dump(feat_pipe, pipe_pkl)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(feat_pipe.transform(raw_X), train_df["rainfall"])
    joblib.dump(model, best_model_pkl)

    m_rain = MLflowTools("binary_rainfall_predection")
    run_id = m_rain.log_run("GradientBoosting_nans_test", params={}, metrics={"f1": 0.88}, artifact_paths=[str(best_model_pkl)])

    # 50 rows with injected missing values
    test_df = train_df.iloc[1460:1510].drop(columns=["rainfall"]).copy()
    test_df.iloc[0:10, 2] = np.nan
    test_df.iloc[10:20, 4] = np.nan
    test_df.iloc[20:30, 6] = np.nan

    csv_buf = io.BytesIO()
    test_df.to_csv(csv_buf, index=False)

    res = client.post(
        f"/projects/binary_rainfall_predection/models/{run_id}/predict",
        files={"test_file": ("test_with_nans.csv", csv_buf.getvalue(), "text/csv")},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")

    pred_df = pd.read_csv(io.StringIO(res.text))
    assert list(pred_df.columns) == ["id", "rainfall"]
    assert len(pred_df) == 50
    assert not pred_df["rainfall"].isna().any()
