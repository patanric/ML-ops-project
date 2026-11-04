"""Training pipeline: feature view -> training data -> model -> model registry."""
# 1. Load feature group
# 2. Create feature view (features + label)
# 3. Create training dataset from the feature view
# 4. Train and evaluate model
# 5. Save model locally (joblib.dump), upload to model registry
