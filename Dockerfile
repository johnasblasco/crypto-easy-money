# syntax=docker/dockerfile:1
#
# One image for everything: the dashboard (default command), the scanner, the
# research studies and the engine trainer. Data and models live in mounted
# volumes (see compose.yaml), never inside the image.

# Override if Docker Hub rate-limits you, e.g.
#   --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.13-slim
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM and XGBoost need the OpenMP runtime, which slim images leave out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing code does not reinstall them. Versions are pinned
# to the tested set (constraints.txt). xgboost-cpu is the same library as xgboost
# without the ~300 MB GPU (NCCL) wheel that plain xgboost pulls in on Linux.
COPY requirements.txt constraints.txt ./
RUN grep -v '^xgboost' requirements.txt > /tmp/requirements-docker.txt \
 && pip install -r /tmp/requirements-docker.txt xgboost-cpu -c constraints.txt \
 && rm /tmp/requirements-docker.txt

# Run as an unprivileged user. On Linux, match your host user so files written to
# the mounted data/ and models/ folders stay yours: HOST_UID/HOST_GID in compose.
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "${GID}" app && useradd -o -m -u "${UID}" -g "${GID}" app

COPY --chown=app:app app ./app
COPY --chown=app:app cryptopredict ./cryptopredict
COPY --chown=app:app quant ./quant
COPY --chown=app:app tests ./tests
COPY --chown=app:app pytest.ini ./
RUN mkdir -p data models && chown app:app data models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
