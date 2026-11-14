# Rain in Zurich: FTI pipelines with Hopsworks
MLOps Projektarbeit, CAS AI Operations (FHNW). Predicts whether it will rain in Zurich in the next 2 hours, built as
three separate pipelines (feature, training, inference) around the Hopsworks feature store and model registry.

| Assessment criterion | Where to find it |
|---|---|
| FTI architecture, feature store used for the features | [Architecture](#architecture-fti) |
| Pipeline steps complete, aggregated and RT feature used at inference | [Pipelines](#pipelines), [Features](#features) |
| Data and model described, dependencies, run instructions | [Data](#data), [Setup and run](#setup-and-run), [Model](#model) |
| Limitations and difficulties | [Limitations and difficulties](#limitations-and-difficulties), [Reflection](#reflection-from-batch-scripts-to-a-production-like-setup) |

## Architecture (FTI)
No pipeline starts another one or passes data to it directly. They only exchange data through the Hopsworks
feature store and the Hopsworks model registry, so each one can run on its own schedule.

```mermaid
flowchart TB
    API[Open-Meteo API] --> F[feature_pipeline.py]
    F -- features + labels --> FG[(Feature group<br>weather_hourly)]
    FG --> FV[Feature view<br>weather_rain_fv]
    FV -- training data --> T[training_pipeline.py]
    T -- model --> MR[(Model registry<br>rain_model)]
    FV -- newest features --> I[inference_pipeline.py]
    MR -- best model --> I
    LIVE[Open-Meteo live<br>cloud_cover] -- RT feature --> I
    I --> P[predictions]
```

| Hopsworks object | Details |
|---|---|
| Feature group `weather_hourly` (v1) | Primary key `city` + `time`, event time `time`, offline store. Written by the feature pipeline (insert = upsert on the primary key). |
| Feature view `weather_rain_fv` (v1) | Query: all features of the feature group except `city` and `time`, label `rain_next_2h`. Used for training data and for loading the newest features at inference. |
| Model registry, model `rain_model` | Each training run registers a new version with its metrics. Inference loads the version with the best F1. |

### Why the decoupling matters
- **Own schedule for each pipeline:** the feature pipeline runs hourly (as often as the data changes), training runs
  rarely and by hand, inference runs whenever a prediction is requested. None of them has to wait for or start another.
- **Features are computed once:** the aggregated features are engineered only in the feature pipeline. Training and
  inference both read them from the same feature view, so the inference does not reimplement the feature engineering,
  a common cause of training-serving skew (course slide decks 15 and 20). Only the RT feature is fetched at inference.
- **Failures stay local:** if a feature pipeline run fails, the inference still works with the newest features already
  in the store; if a training run fails, the best registered model keeps serving.
- **Each pipeline can be changed or replaced on its own:** a different model, a streaming feature pipeline or another
  compute engine would not require changes in the other pipelines, as long as the names of the Hopsworks objects stay
  the same.

The code follows the same rule: no pipeline imports another one. The only shared file is
[`pipeline_config.py`](pipeline_config.py), which holds no logic, just the contract between the pipelines: the
location (stored features and the live RT feature must come from the same place) and the names and versions of the
feature group, feature view and model.

## Pipelines
| Pipeline | Input | Steps | Output |
|---|---|---|---|
| `feature_pipeline.py` | Raw hourly weather data (Open-Meteo) | 1. Fetch raw data via API<br>2. Engineer rolling window features<br>3. Build one dataframe with features and label<br>4. Get or create the feature group (primary key + event time)<br>5. Insert the dataframe | Features and labels in `weather_hourly` |
| `training_pipeline.py` | Features and labels from the feature store | 1. Load the feature group<br>2. Get or create the feature view (features + label)<br>3. Create the training dataset from the feature view (time based train/test split)<br>4. Train and evaluate<br>5. Save locally with `joblib.dump`, upload with `create_model` + `model.save` | New model version in the registry |
| `inference_pipeline.py` | Newest features, live data, model | 1. Fetch live `cloud_cover` (RT feature)<br>2. Get the feature view and load the newest features<br>3. Download the best model from the registry<br>4. Predict | Prediction for the next 2 hours (plus the last 7 days for comparison) |

`serve.py` is a small web app around the same inference code (`connect()` once at start, `predict()` per page load).
All timestamps are stored in UTC.

## Features
| Feature | Type | Computed where | Used at inference |
|---|---|---|---|
| `humidity_mean_24h` | Aggregated (rolling 24h mean of relative humidity) | Feature pipeline, stored in the feature group | Loaded from the feature view |
| `pressure_change_3h` | Aggregated (pressure difference over 3 hours) | Feature pipeline, stored in the feature group | Loaded from the feature view |
| `cloud_cover` | RT feature | Fetched live from the Open-Meteo `current` endpoint at the moment of the prediction | Replaces the stored value of the newest hour |
| `temperature_2m`, `relative_humidity_2m`, `pressure_msl`, `precipitation` | Raw hourly values | Feature pipeline | Loaded from the feature view |

The historical `cloud_cover` values are stored in the feature group so the model can be trained on them;
at inference only the live value is used for the current hour.

**Label:** `rain_next_2h` = 1 if at least 0.1 mm precipitation falls in the 2 hours after the hour.
The newest 2 hours are stored with an empty label (the future is not known yet), so inference always sees features
up to the current hour. The next feature pipeline run fills in the label through the upsert on the primary key.

## Data
[Open-Meteo](https://open-meteo.com/) hourly weather for Zurich (47.37, 8.55), no API key needed.
The data splits naturally into historical data for training and current data for inference:
- **Historical:** archive API (ERA5 reanalysis), 2024-01-01 until about 6 days ago, loaded once with `--backfill`
- **Current:** forecast API, last 7 days up to the current hour, loaded by every regular feature pipeline run
- **Live (RT):** forecast API `current=cloud_cover`, fetched by the inference at prediction time

## Setup and run
**Requirements:** [Docker](https://www.docker.com/) with Docker Compose, and a free account on
[hopsworks.ai](https://www.hopsworks.ai/) with **one project** (the client selects it automatically) and an API key.

```bash
cp .env.example .env                                         # then set HOPSWORKS_API_KEY
docker compose run --rm web feature_pipeline.py --backfill   # once: ~2 years of history into the feature store
docker compose run --rm web training_pipeline.py             # train, evaluate, register the model
docker compose up --build -d                                 # start the two services below
```
Then open http://localhost:8000. `docker compose logs -f` follows both services, `docker compose down` stops them.

| Service | What it does |
|---|---|
| `features` | Runs `feature_pipeline.py` immediately at start, then at minute 1 of every hour. Open-Meteo delivers hourly values and the pipeline keeps rows up to the current time, so a new hour is available right after the full hour; running more often would only rewrite the same hours. |
| `web` | `serve.py`: logs in and loads the best model once, then every page load or click on "Refresh prediction" runs a fresh inference (live `cloud_cover` + newest features) and returns the predictions page. |

Training stays a manual step. The web app keeps the model it loaded at start, so restart it
(`docker compose restart web`) after training a better one.
`docker compose run --rm web <script>` runs any script once, for example
`docker compose run --rm web inference_pipeline.py` writes the predictions page as a static file to `output/predictions.html`.

**Dependencies:** Python 3.12.14 (Hopsworks needs Python < 3.14), packages pinned in `requirements.txt`, both installed
into the image by the `Dockerfile`. `hopsworks[python]` is needed instead of plain `hopsworks`, which lacks pyarrow
for reading and writing features. The image is built for `linux/amd64`, because `hops-deltalake` (required by
`hopsworks[python]`) has no Linux ARM wheel; on Apple Silicon Docker Desktop runs it under emulation, which works but is slower.

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
  features from the offline feature group. A request took about 3.6 s in my test (Apple Silicon, amd64 emulation).
  For low latency the feature group would be `online_enabled` so the newest features come from the online store.
- **Model choice:** inference uses the registered version with the best F1 (`get_best_model`). Hopsworks has
  no alias mechanism like MLflow's `champion` in the course.
- **One location only** (Zurich); predictions only go to the HTML page, they are not written back to the feature store,
  so there is no prediction history or monitoring beyond the last 7 days.
- **Difficulties:**
  - `pip install hopsworks` alone is not enough, the `[python]` extra is needed (pyarrow).
  - Login fails with "Could not find any project" until a project exists in the Hopsworks UI.
  - `get_batch_data(event_time=True)` returns the event time under a prefixed column name
    (`<project>_<feature group>_<version>_time`), so the inference script renames it.
  - Docker build on Apple Silicon failed at first: `hops-deltalake` only has wheels for macOS ARM and Linux x86_64,
    so the image is pinned to `linux/amd64`.

## Reflection: from batch scripts to a production-like setup
**What I noticed.** After building the predictions page I saw that it only showed the current hour after running
both scripts again: first the feature pipeline, then the inference pipeline. My first idea was to always run them
together. Looking at it more closely, that would turn the three pipelines back into one chained process, which is
what the FTI architecture tries to avoid (course slide deck 15: pipelines connected only through the feature store and
the model registry, each one optimizable and scalable on its own).

**What I changed because of it.** The pipelines now run on their own rhythm and only meet in the feature store:
- The feature pipeline runs as often as the data changes. Open-Meteo delivers hourly values, so it runs at minute 1 of
  every hour, right after a new hour becomes available. Running it every few minutes would only rewrite the same rows.
- The inference runs when someone needs a prediction (page load or "Refresh prediction") and simply uses the newest
  features that are in the feature store at that moment, plus the live `cloud_cover` as RT feature.
- Training runs rarely and by hand; the web app picks the best registered model at start.

In the terms of the course (slide deck 11) this moved the project from offline (batch) prediction to online prediction
with batch features plus one RT feature.

**What a faster use case would need.** I asked myself what would happen with data that changes every second, like a
ticker. Re-running scripts every second would not work: in my setup one inference request already takes several
seconds (reading the newest features from the offline store). Such a system would need a different architecture:
a streaming feature pipeline that keeps running and computes NRT features, the online store of the feature store for
fast lookups, and a model that stays loaded in a serving process. Hourly weather data does not need this, so I kept
the batch feature pipeline.

**What I would do next.**
- Enable the online store for the feature group, so online inference does not read from the offline store.
- Run the feature pipeline with a real scheduler (GitHub Actions or a Hopsworks job) instead of a loop on my machine.
- Write predictions back to the feature store. The labels arrive 2 hours later (natural labels), so a separate
  monitoring pipeline could then track the hit rate over time instead of only the last 7 days shown on the page.

---

*The following sections are not part of the assessment (model performance and extensions are not graded) and are kept
for completeness.*

## Model
`GradientBoostingClassifier(n_estimators=10)` from scikit-learn, evaluated with accuracy, precision, recall and F1
(same setup as the course exercises).

The train/test split is by time (train 2024-01-01 to 2025-12-31, test from 2026-01-01) instead of a random split:
neighbouring hours are almost identical, so a random split would leak information into the test set.
Inference loads the registered version with the best F1 (`get_best_model`), the Hopsworks counterpart to the MLflow
`champion` alias used in the course.

## Results (run on 2026-11-11, page on 2026-11-12)
- Backfill: 24,937 hourly rows (2024-01-01 to 2026-11-05), rain share 25.7%
- Training: 17,521 train rows, 7,554 test rows; accuracy 0.89, precision 0.83, recall 0.63, F1 0.72
- Inference: features up to the current hour, live `cloud_cover` 97%, prediction "no rain in the next 2 hours"
- Predictions page (2026-11-12, last 7 days): 135 of 166 hours correct (81%), 36 of 58 rainy hours caught, 9 false alarms

## Predictions page
Light and dark mode. The current 2 hour forecast with rain probability on top, then hit rate, caught rain hours and
false alarms, a timeline of the model's rain probability against the hours where rain actually fell, and an hour by
hour grid where a cross marks every wrong prediction. The newest hour uses the live `cloud_cover`; older hours use the
stored features. The last 2 hours show as "not known yet" until the next feature pipeline run fills in their label.
Times on the page are shown in Europe/Zurich.

## Extensions implemented
From the list of possible extensions in the assignment:
- **Train/test split in the Hopsworks training dataset** (`fv.train_test_split`) instead of scikit-learn's `train_test_split()`
- **Inference in a web application** (`serve.py`) in addition to the inference pipeline
- **Containerization** (`Dockerfile`, `docker-compose.yml`)
