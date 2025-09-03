#!/bin/bash

# Local CI script for MLOps pipeline

echo "Running local CI..."

# Install dependencies
echo "Installing dependencies..."
uv sync

# Run linting
echo "Running linting..."
uv run ruff check .

# Run tests
echo "Running tests..."
uv run pytest

# Run training script
echo "Running training script..."
uv run python scripts/train.py

echo "Local CI completed!"
