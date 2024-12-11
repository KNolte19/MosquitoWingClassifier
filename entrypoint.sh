#!/bin/bash

exec gunicorn app:app -b 0.0.0.0:8080 --timeout 360
