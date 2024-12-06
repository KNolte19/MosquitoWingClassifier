# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.12
FROM python:${PYTHON_VERSION} as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

COPY u2net.onnx /home/appuser/.u2net/u2net.onnx

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    llvm llvm-dev clang \
    && apt-get clean

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

RUN mkdir -p /app/static/requests \
    && chown -R appuser:appuser /app/static/requests

ENV NUMBA_CACHE_DIR=/tmp    

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Copy requirements.txt into the container for pip install.
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Ensure the entrypoint script is executable.
RUN chmod +x /app/entrypoint.sh

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 5050

# Run the application.
ENTRYPOINT ["./entrypoint.sh"]
