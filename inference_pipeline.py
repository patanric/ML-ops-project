"""Inference pipeline: latest features + live cloud_cover (RT) -> best registered model -> prediction.

Run feature_pipeline.py (without --backfill) first, so the feature group holds the latest hours.

Usage:
    python inference_pipeline.py
"""
import os

import hopsworks
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from feature_pipeline import LAT, LON

FV_NAME, FV_VERSION = "weather_rain_fv", 1  # created by training_pipeline.py
MODEL_NAME = "rain_model"

load_dotenv()

# 1. Live data: cloud_cover is the RT feature, only known at inference time
r = requests.get(
    "https://api.open-meteo.com/v1/forecast",
    params={"latitude": LAT, "longitude": LON, "current": "cloud_cover", "timezone": "UTC"},
    timeout=30,
)
r.raise_for_status()
cloud_cover_now = r.json()["current"]["cloud_cover"]

project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# 2. Feature view, load the newest features (precomputed aggregates)
fv = fs.get_feature_view(FV_NAME, version=FV_VERSION)
since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
df = fv.get_batch_data(start_time=since, event_time=True)
# event time comes back prefixed (<project>_<feature group>_<version>_time)
df = df.rename(columns=lambda c: "time" if c.endswith("_time") else c)
df = df.sort_values("time").tail(1).reset_index(drop=True)
df["cloud_cover"] = cloud_cover_now

# 3. Best model from the registry (Hopsworks counterpart to the MLflow "champion" alias)
model = project.get_model_registry().get_best_model(MODEL_NAME, "f1", "max")
clf = joblib.load(os.path.join(model.download(), "model.joblib"))

# 4. Predict
df["prediction"] = clf.predict(df.drop(columns=["time"]))
print(f"model {MODEL_NAME} v{model.version}, features from {df.at[0, 'time']} UTC, live cloud_cover {cloud_cover_now}%")
print("Rain in the next 2 hours" if df.at[0, "prediction"] else "No rain in the next 2 hours")
