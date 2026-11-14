"""Contract between the three pipelines: location and names of the Hopsworks objects.

Constants only, no pipeline code. The pipelines never import each other; they only agree on these names
and exchange data through the feature store and the model registry.
"""
CITY, LAT, LON = "zurich", 47.37, 8.55  # stored features and live RT feature must come from the same place
FG_NAME, FG_VERSION = "weather_hourly", 1  # written by feature_pipeline.py
FV_NAME, FV_VERSION = "weather_rain_fv", 1  # created by training_pipeline.py, read by inference_pipeline.py
MODEL_NAME = "rain_model"  # registered by training_pipeline.py, loaded by inference_pipeline.py
