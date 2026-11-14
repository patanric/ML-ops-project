"""Feature pipeline: Open-Meteo hourly weather -> features + label -> Hopsworks feature group.

Target:     rain_next_2h  (1 if >= 0.1 mm precipitation in the next 2 hours)
Aggregated: humidity_mean_24h, pressure_change_3h
RT feature: cloud_cover (stored here for training, fetched live at inference)

Usage:
    python feature_pipeline.py --backfill   # ~2 years from the archive API, run once
    python feature_pipeline.py              # last 7 days from the forecast API, run regularly
    add --dry-run to print the dataframe instead of writing to Hopsworks
"""
import argparse
import os
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from pipeline_config import CITY, FG_NAME, FG_VERSION, LAT, LON

HOURLY = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "pressure_msl", "precipitation"]


def fetch(backfill: bool) -> pd.DataFrame:
    """1. Fetch raw hourly data (UTC)."""
    params = {"latitude": LAT, "longitude": LON, "hourly": ",".join(HOURLY), "timezone": "UTC"}
    if backfill:
        url = "https://archive-api.open-meteo.com/v1/archive"
        # archive lags ~5 days; the regular run's 7-day window closes the gap
        params |= {"start_date": "2024-01-01", "end_date": str(date.today() - timedelta(days=6))}
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params |= {"past_days": 7, "forecast_days": 1}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return df[df["time"] <= now].dropna()  # forecast hours are not observations


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """2.+3. Rolling-window features and label in one dataframe."""
    df = df.sort_values("time").reset_index(drop=True)
    df["humidity_mean_24h"] = df["relative_humidity_2m"].rolling(24, min_periods=24).mean()
    df["pressure_change_3h"] = df["pressure_msl"].diff(3)
    next_2h = df["precipitation"].shift(-1) + df["precipitation"].shift(-2)
    # last 2h have no label yet: kept with <NA> so inference sees fresh features,
    # the next run fills the label via upsert (primary key city + time)
    df["rain_next_2h"] = (next_2h >= 0.1).astype("Int64").where(next_2h.notna())
    df = df.dropna(subset=["humidity_mean_24h", "pressure_change_3h"]).copy()  # 24h warm-up
    df.insert(0, "city", CITY)
    return df


def upload(df: pd.DataFrame) -> None:
    """4.+5. Create feature group (primary key + event time) and insert."""
    import hopsworks  # imported here so --dry-run works without Hopsworks

    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fg = project.get_feature_store().get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["city", "time"],
        event_time="time",
        description="Hourly Open-Meteo weather for Zurich with 24h/3h aggregates and rain_next_2h label",
    )
    fg.insert(df, write_options={"wait_for_job": True})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    df = engineer(fetch(args.backfill))
    print(df.tail(), f"\n{len(df)} rows, rain share {df['rain_next_2h'].mean():.2%}")
    if not args.dry_run:
        upload(df)
