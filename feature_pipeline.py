"""Feature pipeline: raw data -> features + label -> Hopsworks feature group."""
# 1. Fetch raw data from the API
# 2. Engineer features (rolling windows, aggregations)
# 3. Build dataframe with features + label
# 4. Create feature group (primary key + event time)
# 5. Insert dataframe into the feature group
