import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import mlflow
import os
import logging
import random
from dotenv import load_dotenv
import mlflow.exceptions

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs('/tmp', exist_ok=True)

# Global variables
DATA_PATH = os.getenv('DATA_PATH')
MODEL_ARTIFACT_PATH = os.getenv('MODEL_ARTIFACT_PATH', '/tmp/model')
PREDICTION_LENGTH = 24
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:8001')

# Logger setup
logger = logging.getLogger(__name__)

def load_data():
    """Load data from CSV and convert to TimeSeriesDataFrame."""
    logger.info("Starting load_data: Loading data from CSV")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded: {df.shape}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.rename(columns={"store_id": "item_id", "real_order_quantity": "target"})
        df = df.sort_values(["item_id", "timestamp"])
        data = TimeSeriesDataFrame.from_data_frame(df)
        logger.info("Finished load_data: Data loaded and converted to TimeSeriesDataFrame")
        return data
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        raise


def train_test_split(data):
    """Split data into train and test sets."""
    logger.info("Starting train_test_split: Splitting data into train and test")
    train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
    logger.info("Finished train_test_split: Data split completed")
    return train_data, test_data

def train_model(train_data):
    """Train the model using AutoGluon and log to MLflow."""
    logger.info("Starting train_model: Training the model")
    try:
        predictor = TimeSeriesPredictor(
            path=MODEL_ARTIFACT_PATH,
            prediction_length=PREDICTION_LENGTH,
        ).fit(
            train_data=train_data,
            hyperparameters={
                "Chronos": [
                    {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                    {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
                ]
            },
            time_limit=1200,
            enable_ensemble=False,
        )

        # Log hyperparameters
        mlflow.log_param("prediction_length", PREDICTION_LENGTH)
        mlflow.log_param("time_limit", 60) # 수정: fit()의 time_limit과 일치시킴
        mlflow.log_param("enable_ensemble", False)

        # Log leaderboard metrics
        leaderboard = predictor.leaderboard()
        for idx, row in leaderboard.iterrows():
            model_name = row['model']
            sanitized_model_name = model_name.replace("[", "_").replace("]", "_")
            if 'score_val' in row:
                mlflow.log_metric(f"{sanitized_model_name}_val_score", row['score_val'])
            if 'pred_time_val' in row:
                mlflow.log_metric(f"{sanitized_model_name}_pred_time_val", row['pred_time_val'])

        # Log best model score
        best_score = leaderboard['score_val'].max()
        mlflow.log_metric("best_model_val_score", best_score)

        mlflow.log_artifact(predictor.path, 'predictor')

    except Exception as e:
        logger.error(f"Model training or MLflow logging failed: {e}")
        raise

    logger.info("Finished train_model: Model training completed")
    return predictor


def predict(predictor, train_data):
    """Make predictions using the trained model."""
    logger.info("Starting predict: Making predictions")
    predictions = predictor.predict(train_data)
    logger.info("Finished predict: Predictions generated")

    predictions.to_csv('/tmp/predictions.csv')
    mlflow.log_artifact('/tmp/predictions.csv', 'predictions')
    logger.info("Predictions saved and logged to MLflow")
    return predictions


def evaluate(predictor, test_data):
    """Evaluate the model and log leaderboard."""
    logger.info("Starting evaluate: Evaluating the model")
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv('/tmp/leaderboard.csv')
    mlflow.log_artifact('/tmp/leaderboard.csv', 'leaderboard')
    logger.info("Evaluation completed and leaderboard saved to MLflow")
    return leaderboard


def visualize(predictor, data, predictions):
    """Generate and save visualization plot."""
    logger.info("Starting visualize: Generating visualization")
    # 수정: 이미 TimeSeriesDataFrame이므로 재변환 불필요
    item_ids_to_visualize = random.sample(list(data.item_ids), min(10, len(data.item_ids)))
    fig = predictor.plot(
        data=data,
        predictions=predictions,
        item_ids=item_ids_to_visualize,
        max_history_length=200,
    )
    fig.savefig('/tmp/forecast_plot.png')
    mlflow.log_artifact('/tmp/forecast_plot.png', 'forecast_plot')
    logger.info("Visualization saved to MLflow")


if __name__ == "__main__":
    logger.info("Starting ML pipeline execution")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment("train_model_20M")
    logger.info("Using MLflow experiment: ml_pipeline_experiment")

    with mlflow.start_run():
        data = load_data()
        train_data, test_data = train_test_split(data)
        predictor = train_model(train_data)
        predictions = predict(predictor, train_data)
        leaderboard = evaluate(predictor, test_data)
        visualize(predictor, test_data, predictions)

    logger.info("ML pipeline execution completed. Check MLflow UI at http://localhost:8001")