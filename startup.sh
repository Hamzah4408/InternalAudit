#!/bin/bash
exec uvicorn personnel_activeness:app --host 0.0.0.0 --port 8000
