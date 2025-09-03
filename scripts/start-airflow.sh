#!/bin/bash

# Initialize Airflow database
uv run airflow db init

# Create admin user if it doesn't exist
if ! uv run airflow users list | grep -q admin; then
    uv run airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
fi

# Start Airflow in standalone mode
uv run airflow standalone
