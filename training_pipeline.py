"""Training pipeline: feature groups -> feature view (join) -> training data -> model -> Hopsworks model registry.

Model, metrics and parameters follow course exercises 07/10 (GradientBoosting, 4 metrics).
Split is by time instead of random: neighbouring hours are almost identical, a random
split would leak them into the test set (training-serving skew, slide 20).

Usage:
    python training_pipeline.py
"""
import os

import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pipeline_config import FG_NAME, FG_VERSION, FV_NAME, FV_VERSION, LABEL_FG_NAME, LABEL_FG_VERSION, MODEL_NAME

TRAIN_START, TEST_START = "2024-01-01", "2026-01-01"

load_dotenv()
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# 1. Load feature groups
weather = fs.get_feature_group(FG_NAME, version=FG_VERSION)
labels = fs.get_feature_group(LABEL_FG_NAME, version=LABEL_FG_VERSION)

# 2. Feature view: weather features joined with the label on the key city + time, keys are not features.
#    Left join: the newest 2 hours have no label yet but their features are needed for inference.
fv = fs.get_or_create_feature_view(
    name=FV_NAME,
    version=FV_VERSION,
    query=weather.select_except(["city", "time"]).join(
        labels.select(["rain_next_2h"]), on=["city", "time"], join_type="left"
    ),
    labels=["rain_next_2h"],
    description="Hourly weather features joined with the rain_next_2h label on city + time",
)

# 3. Training data from the feature view, split by event time
X_train, X_test, y_train, y_test = fv.train_test_split(
    train_start=TRAIN_START,
    train_end=TEST_START,
    test_start=TEST_START,
    description=f"Time split: train {TRAIN_START} to {TEST_START}, test from {TEST_START}",
)
# the newest 2 hours have no label yet (see feature_pipeline.py)
test_mask = y_test["rain_next_2h"].notna()
X_test, y_test = X_test[test_mask], y_test[test_mask]
y_train, y_test = y_train["rain_next_2h"].astype(int), y_test["rain_next_2h"].astype(int)
print(f"train {len(X_train)} rows, test {len(X_test)} rows")

# 4. Train and evaluate
params = {"n_estimators": 10}
clf = GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
print(metrics)

# 5. Save locally, then upload to the model registry
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/model.joblib")
model = project.get_model_registry().python.create_model(
    name=MODEL_NAME,
    metrics=metrics,
    feature_view=fv,
    description=f"GradientBoostingClassifier {params}, test set from {TEST_START}",
)
model.save("model")
print(f"registered {MODEL_NAME} version {model.version}")
