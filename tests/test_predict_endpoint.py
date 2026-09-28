import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db

client = TestClient(app)


def test_predict_endpoint_flow():
    init_db()

    # 1. Project house_price_predection should exist on disk
    train_csv_path = "projects/house_price_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

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


def test_predict_raw_data_auto_feature_engineering():
    """Verify that uploading unprocessed raw data auto-applies feature_pipeline.pkl."""
    init_db()

    train_csv_path = "projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

    # 10 raw rows without rainfall target
    raw_test_df = train_df.head(10).drop(columns=["rainfall"])
    csv_buf = io.BytesIO()
    raw_test_df.to_csv(csv_buf, index=False)

    res = client.post(
        "/projects/binary_rainfall_predection/models/f802a1aa7cc148e7a159258353c0105d/predict",
        files={"test_file": ("raw_test.csv", csv_buf.getvalue(), "text/csv")},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")

    pred_df = pd.read_csv(io.StringIO(res.text))
    assert list(pred_df.columns) == ["id", "rainfall"]
    assert len(pred_df) == 10
    assert not pred_df["rainfall"].isna().any()


def test_predict_gradient_boosting_with_nans():
    """Verify that uploading test data with NaNs does not crash GradientBoosting with NaN error."""
    import numpy as np

    init_db()

    train_csv_path = "projects/binary_rainfall_predection/datasets/dataset_v1/data.csv"
    train_df = pd.read_csv(train_csv_path)

    # 50 rows with injected missing values
    test_df = train_df.iloc[1460:1510].drop(columns=["rainfall"]).copy()
    test_df.iloc[0:10, 2] = np.nan
    test_df.iloc[10:20, 4] = np.nan
    test_df.iloc[20:30, 6] = np.nan

    csv_buf = io.BytesIO()
    test_df.to_csv(csv_buf, index=False)

    res = client.post(
        "/projects/binary_rainfall_predection/models/a10babff802d4afa858f6bcbe482b274/predict",
        files={"test_file": ("test_with_nans.csv", csv_buf.getvalue(), "text/csv")},
    )
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/csv")

    pred_df = pd.read_csv(io.StringIO(res.text))
    assert list(pred_df.columns) == ["id", "rainfall"]
    assert len(pred_df) == 50
    assert not pred_df["rainfall"].isna().any()

