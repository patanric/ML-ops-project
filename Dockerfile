# check=skip=FromPlatformFlagConstDisallowed
# Same base as the course development_env, pinned to the Python version the pipelines were tested with.
# amd64 only: hops-deltalake (needed by hopsworks[python]) has no Linux ARM wheel,
# on Apple Silicon this runs under Docker Desktop emulation
FROM --platform=linux/amd64 python:3.12.14-slim

# gcc is needed to compile the twofish package that hopsworks depends on (as in the course Dockerfile)
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# print() output shows up in `docker compose logs` right away
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline_config.py feature_pipeline.py training_pipeline.py inference_pipeline.py serve.py predictions_template.html ./

# the API key comes from `docker run --env-file .env`, never baked into the image
ENTRYPOINT ["python"]
CMD ["inference_pipeline.py"]
