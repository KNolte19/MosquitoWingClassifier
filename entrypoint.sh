#!/bin/bash
mkdir -p ~/MosquitoWingClassifier/app/static/requests
chown -R appuser:appuser ~/MosquitoWingClassifier/app/static/requests

exec gunicorn app:app -b 0.0.0.0:8080 --timeout 360

