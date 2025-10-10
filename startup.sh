#!/bin/bash
gunicorn personnel_activeness:app --bind=0.0.0.0:$PORT

