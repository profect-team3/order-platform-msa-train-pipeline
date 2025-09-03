# Order Platform MSA Train Pipeline

This project sets up an MLOps pipeline using Airflow, MLflow, and DVC for training machine learning models in a microservices architecture.

## Project Structure

- `dags/`: Airflow DAGs for orchestrating workflows
- `plugins/`: Custom Airflow plugins
- `models/`: Trained machine learning models
- `data/`: Data files managed by DVC
- `experiments/`: MLflow experiments and runs
- `scripts/`: Training and utility scripts
- `config/`: Configuration files
- `notebooks/`: Jupyter notebooks for experimentation

## Setup

1. Install dependencies using uv:
   ```
   uv sync
   ```

2. Initialize DVC (already done):
   ```
   uv run dvc init
   ```

3. Start Airflow:
   ```
   uv run airflow standalone
   ```

4. Start MLflow UI:
   ```
   uv run mlflow ui
   ```

## Usage

- Place your DAGs in the `dags/` directory.
- Run training scripts from `scripts/`.
- Track experiments with MLflow.
- Version data with DVC.

## Technologies

- **Airflow**: Workflow orchestration
- **MLflow**: Experiment tracking and model management
- **DVC**: Data versioning
- **uv**: Fast Python package manager
