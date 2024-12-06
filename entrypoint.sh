#!/bin/bash
mkdir -p ~/MosquitoWingClassifier/static/requests
chown -R appuser:appuser ~/MosquitoWingClassifier/static/requests

exec gunicorn app:app -b 0.0.0.0:8080 --timeout 360

