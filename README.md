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
Requirements: [Docker](https://www.docker.com/) with Docker Compose, and a free account on [hopsworks.ai](https://www.hopsworks.ai/)
with **one project** (the client selects it automatically) and an API key.

```bash
cp .env.example .env   # then set HOPSWORKS_API_KEY
```

Dependencies are pinned in `requirements.txt` and installed into the image by the `Dockerfile`
(Python 3.12.14, Hopsworks needs Python < 3.14). `hopsworks[python]` is needed instead of plain `hopsworks`,
which lacks pyarrow for reading and writing features.
The image is built for `linux/amd64`, because `hops-deltalake` (required by `hopsworks[python]`) has no Linux ARM wheel.
On Apple Silicon Docker Desktop runs it under emulation, which works but is slower.

## Run
```bash
docker compose run --rm web feature_pipeline.py --backfill   # once: ~2 years of history into the feature store
docker compose run --rm web training_pipeline.py             # train, evaluate, register the model
docker compose up --build -d                                 # start the two services below
```
Then open http://localhost:8000. `docker compose logs -f` follows both services, `docker compose down` stops them.

| Service | What it does |
|---|---|
| `features` | Runs `feature_pipeline.py` immediately at start, then at minute 1 of every hour. Open-Meteo delivers hourly values and the pipeline keeps rows up to the current time, so a new hour is available right after the full hour; running more often would only rewrite the same hours. |
| `web` | `serve.py`: logs in and loads the best model once, then every page load or click on "Refresh prediction" runs a fresh inference (live `cloud_cover` + newest features) and returns the predictions page. |

The feature pipeline and the inference only meet in the feature store: each runs on its own schedule
(features hourly, inference on request). Training stays a manual step; the web app keeps the model it loaded
at start, so restart it (`docker compose restart web`) after training a better one.

`docker compose run --rm web <script>` runs any of the scripts once, for example
`docker compose run --rm web inference_pipeline.py` writes the same page as a static file to `output/predictions.html`.

**The predictions page** (light and dark mode): the current 2 hour forecast with rain probability on top,
then hit rate, caught rain hours and false alarms, a timeline of the model's rain probability against the hours
where rain actually fell, and an hour by hour grid where a cross marks every wrong prediction.
The newest hour uses the live `cloud_cover`; older hours use the stored features. The last 2 hours show as "not known yet"
until the next feature pipeline run fills in their label. Times on the page are shown in Europe/Zurich.

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
- **Simple scheduling:** the hourly feature run is a sleep loop in a container, so it only runs while Docker runs
  on this machine. In production a scheduler would trigger it (cron, GitHub Actions, Airflow or a Hopsworks job).
  Training is still started by hand.
- **Online inference from the offline store:** the web app predicts on request (online prediction), but reads the
  features from the offline feature group. A request took about 3.6 s in our test (Apple Silicon, amd64 emulation).
  For low latency the feature group would be `online_enabled` so the newest features come from the online store.
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
