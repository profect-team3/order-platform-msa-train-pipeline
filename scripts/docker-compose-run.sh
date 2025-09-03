#!/bin/bash

# Docker Compose script to run the full MLOps stack

echo "Starting MLOps services with Docker Compose..."
echo "Airflow will be available at http://localhost:8000"
echo "MLflow will be available at http://localhost:8001"
echo "Check logs in Docker Desktop"

docker-compose up --build
