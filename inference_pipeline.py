"""Inference pipeline: features of the last 7 days + live cloud_cover (RT) -> best registered model -> predictions.

Predicts every hour of the last 7 days (batch inference), compares each prediction with the actual label
(known 2 hours later) and writes output/predictions.html. The newest hour is the current forecast.
serve.py reuses connect() and predict() to run the same inference on every page load.

Run feature_pipeline.py (without --backfill) first, so the feature group holds the latest hours.

Usage:
    python inference_pipeline.py
"""
import json
import os
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from pipeline_config import FG_NAME, FG_VERSION, FV_NAME, FV_VERSION, LAT, LON, MODEL_NAME

TEMPLATE = Path(__file__).with_name("predictions_template.html")
OUTPUT = Path("output/predictions.html")


class ForecastUnavailableError(RuntimeError):
    """No fresh feature data is available for a current forecast."""


def connect():
    """Log in, get feature view + feature group, download the best model (done once)."""
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fv = fs.get_feature_view(FV_NAME, version=FV_VERSION)
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    # best model from the registry (Hopsworks counterpart to the MLflow "champion" alias)
    model = project.get_model_registry().get_best_model(MODEL_NAME, "f1", "max")
    clf = joblib.load(os.path.join(model.download(), "model.joblib"))
    return fv, fg, model, clf


def predict(fv, fg, model, clf) -> str:
    """Live RT feature + newest features -> predictions -> HTML page."""
    # 1. Live data: cloud_cover is the RT feature, only known at inference time
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON, "current": "cloud_cover", "timezone": "UTC"},
        timeout=30,
    )
    r.raise_for_status()
    cloud_cover_now = r.json()["current"]["cloud_cover"]

    # 2. Feature view, load the features of the last 7 days (precomputed aggregates)
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
    df = fv.get_batch_data(start_time=since, event_time=True)
    # event time comes back prefixed (<project>_<feature group>_<version>_time)
    df = df.rename(columns=lambda c: "time" if c.endswith("_time") else c)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    if df.empty or df.iloc[-1]["time"] != pd.Timestamp.now(tz="UTC").floor("h"):
        raise ForecastUnavailableError(
            "Current forecast unavailable: no weather data for the current hour. "
            "Please try again after the next hourly update."
        )
    df.loc[df.index[-1], "cloud_cover"] = cloud_cover_now  # newest hour = current forecast, uses the live value

    # 3. Predict every hour
    X = df.drop(columns=["time"])
    df["prediction"] = clf.predict(X)
    df["rain_probability"] = clf.predict_proba(X)[:, 1]

    # 4. Actual labels (filled in 2 hours later by the feature pipeline) to see how the model did
    labels = fg.select(["time", "rain_next_2h"]).filter(fg.time >= since).read()
    labels["time"] = pd.to_datetime(labels["time"], utc=True)
    df = df.merge(labels, on="time", how="left")

    latest = df.iloc[-1]
    print(f"model {MODEL_NAME} v{model.version}, features from {latest['time']}, live cloud_cover {cloud_cover_now}%")
    print("Rain in the next 2 hours" if latest["prediction"] else "No rain in the next 2 hours")

    # 5. Static HTML page: current forecast + last 7 days predicted vs actual
    records = [
        {"time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "pred": int(p), "prob": round(float(q), 3),
         "actual": None if pd.isna(a) else int(a)}
        for t, p, q, a in df[["time", "prediction", "rain_probability", "rain_next_2h"]].itertuples(index=False)
    ]
    meta = {"model": MODEL_NAME, "version": model.version, "cloud_cover": cloud_cover_now,
            "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
    return TEMPLATE.read_text().replace("__DATA__", json.dumps(records)).replace("__META__", json.dumps(meta))


if __name__ == "__main__":
    load_dotenv()
    html = predict(*connect())
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT}")
