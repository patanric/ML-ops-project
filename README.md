# MLOps Projektarbeit

## Data source
[Open-Meteo](https://open-meteo.com/) hourly weather for Zurich (47.37, 8.55), no API key needed.
- Historical (training): archive API, 2024-01-01 until ~6 days ago (`--backfill`)
- Recent (inference): forecast API, last 7 days up to the current hour

## Target and features
- Target: `rain_next_2h`, 1 if at least 0.1 mm precipitation falls in the next 2 hours
- Aggregated features: `humidity_mean_24h` (rolling 24h mean), `pressure_change_3h` (3h pressure difference)
- Real-time feature: `cloud_cover`, fetched live at inference time (stored historically for training)
- Raw hourly features: `temperature_2m`, `relative_humidity_2m`, `pressure_msl`, `precipitation`

Feature group `weather_hourly` (v1), primary key `city` + `time`, event time `time`.
The newest 2 hours are stored with an empty label (future not known yet), so inference always sees
features up to the current hour. The next run fills in the label through the upsert on the primary key.

## Model
`GradientBoostingClassifier(n_estimators=10)` from scikit-learn, evaluated with accuracy, precision,
recall and F1 (same setup as the course exercises). Model performance is not the goal of this project.

The train/test split is by time (train 2024-01-01 to 2025-12-31, test from 2026-01-01) instead of a random split:
neighbouring hours are almost identical, so a random split would leak information into the test set.

Model registry: Hopsworks, model name `rain_model`. Inference loads the version with the best F1
(`get_best_model`), the Hopsworks counterpart to the MLflow `champion` alias used in the course.

## Pipelines (FTI architecture)
| Script | Steps |
|---|---|
| `feature_pipeline.py` | Fetch Open-Meteo data, compute rolling features and label, insert into feature group `weather_hourly` |
| `training_pipeline.py` | Load feature group, create feature view `weather_rain_fv`, time-based train/test split, train, evaluate, save with joblib, upload to the model registry |
| `inference_pipeline.py` (+ `serve.py`) | Fetch live `cloud_cover`, load the features of the last 7 days from the feature view, download the best model, predict every hour, compare with the actual labels, write `output/predictions.html` (`serve.py` runs the same inference per page load) |

All timestamps are UTC.

## Setup
Requires Python 3.12 (Hopsworks needs <3.14) and a free account on [hopsworks.ai](https://www.hopsworks.ai/)
with **one project** (the client selects it automatically) and an API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set HOPSWORKS_API_KEY
```
`requirements.txt` uses `hopsworks[python]`: plain `hopsworks` lacks pyarrow, which the Python client needs to read and write features.

## Run
```bash
python feature_pipeline.py --backfill   # once: ~2 years of history
python feature_pipeline.py              # regularly: last 7 days (upsert)
python training_pipeline.py
python inference_pipeline.py
open output/predictions.html            # macOS; or open the file in any browser
```

`output/predictions.html` is a static page (no server needed, light and dark mode): the current 2 hour forecast with
rain probability on top, then hit rate, caught rain hours and false alarms, a timeline of the model's rain probability
against the hours where rain actually fell, and an hour by hour grid where a cross marks every wrong prediction.
The newest hour uses the live `cloud_cover`; older hours use the stored features. The last 2 hours show as "not known yet"
until the next feature pipeline run fills in their label. Times on the page are shown in Europe/Zurich.

## Run with Docker (alternative, no local Python needed)
Avoids the Python version requirement (Hopsworks does not support Python 3.14 yet). Only `.env` with the API key is needed.

```bash
docker build -t rain-pipelines .
docker run --rm --env-file .env rain-pipelines feature_pipeline.py --backfill   # once
docker run --rm --env-file .env rain-pipelines feature_pipeline.py
docker run --rm --env-file .env rain-pipelines training_pipeline.py
docker run --rm --env-file .env -v "$PWD/output:/app/output" rain-pipelines    # inference (default), writes output/predictions.html
```

The image is built for `linux/amd64`: `hops-deltalake` (required by `hopsworks[python]`) has no Linux ARM wheel.
On Apple Silicon Docker Desktop runs it under emulation, which works but is slower.

## Production-like setup (Docker Compose)
Runs the feature pipeline on a schedule and the inference as a web app, the two only connected through the feature store:

```bash
docker compose up --build -d     # then open http://localhost:8000
docker compose logs -f           # follow both services
docker compose down              # stop
```

| Service | What it does |
|---|---|
| `features` | Runs `feature_pipeline.py` immediately at start, then at minute 1 of every hour. Open-Meteo delivers hourly values and the pipeline keeps rows up to the current time, so a new hour is available right after the full hour; running more often would only rewrite the same hours. |
| `web` | `serve.py`: logs in and loads the best model once, then every page load or click on "Refresh prediction" runs a fresh inference (live `cloud_cover` + newest features) and returns the predictions page. |

Training stays a manual step (`docker run --rm --env-file .env rain-pipelines training_pipeline.py`). The web app keeps
the model it loaded at start, so restart it (`docker compose restart web`) after training a better one.

## Results (run on 2026-11-11, page on 2026-11-12)
- Backfill: 24,937 hourly rows (2024-01-01 to 2026-11-05), rain share 25.7%
- Training: 17,521 train rows, 7,554 test rows; accuracy 0.89, precision 0.83, recall 0.63, F1 0.72
- Inference: features up to the current hour, live `cloud_cover` 97%, prediction "no rain in the next 2 hours"
- Predictions page (2026-11-12, last 7 days): 135 of 166 hours correct (81%), 36 of 58 rainy hours caught, 9 false alarms

## Limitations and difficulties
- **Two data sources for one feature set:** the backfill uses the reanalysis archive (ERA5), recent hours come from the
  forecast API's past data. Both are model data, not station measurements, and can differ slightly
  (possible training-serving skew).
- **RT feature at inference:** `cloud_cover` is replaced by the current value from the forecast API. In training
  the same column comes from the hourly data, so the two values are close but not produced identically.
- **Labels and features in one feature group:** the newest 2 hours are stored without a label and completed later
  by upsert. A cleaner design would separate labels into their own feature group (spine group, point-in-time join).
- **No scheduling:** the pipelines are run by hand. In production the feature pipeline would run hourly
  (cron, GitHub Actions or Airflow), training occasionally, inference on demand.
- **Model choice:** inference uses the registered version with the best F1 (`get_best_model`). Hopsworks has
  no alias mechanism like MLflow's `champion` in the course.
- **One location only** (Zurich); predictions only go to the local HTML page, they are not written back to the feature store,
  so there is no prediction history or monitoring beyond the last 7 days.
- **Difficulties:**
  - `pip install hopsworks` alone is not enough, the `[python]` extra is needed (pyarrow).
  - Login fails with "Could not find any project" until a project exists in the Hopsworks UI.
  - `get_batch_data(event_time=True)` returns the event time under a prefixed column name
    (`<project>_<feature group>_<version>_time`), so the inference script renames it.
  - Docker build on Apple Silicon failed at first: `hops-deltalake` only has wheels for macOS ARM and Linux x86_64,
    so the image is pinned to `linux/amd64`.
