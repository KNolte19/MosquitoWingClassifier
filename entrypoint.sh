#!/bin/bash
mkdir -p /app/static/requests
chown -R appuser:appuser /app/static/requests

exec gunicorn app:app -b 0.0.0.0:5050

