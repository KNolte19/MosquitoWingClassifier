# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for h5py and other Python packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

ENV NUMBA_CACHE_DIR=/tmp    

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Copy requirements.txt into the container for pip install.
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["./entrypoint.sh"]
