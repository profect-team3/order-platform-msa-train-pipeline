import json
import logging
import os
import random
from datetime import datetime, timedelta

import pendulum
from airflow.decorators import dag, task
# Set environment variables BEFORE any heavy imports


# Global variables
DATA_PATH = "/Users/coldbrew_groom/Documents/order-platform-msa-train-pipeline/data/forecast_data_featured.csv"
PREDICTION_LENGTH = 24
MLFLOW_TRACKING_URI = "http://localhost:5001"  # 로컬 MLflow 서버로 변경


@dag(
    dag_id="chronos_train_dag",
    schedule=None,
    start_date=pendulum.datetime(2025, 9, 5, tz="UTC"),
    catchup=False,
    tags=["ml", "pipeline"],
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2023, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    },
)
def chronos_train_dag():
    """
    ### ML Pipeline DAG Documentation
    This is a simple ML pipeline example which demonstrates the use of
    the TaskFlow API for loading data, preprocessing, training, predicting,
    evaluating, visualizing, and saving a model.
    """

    @task()
    def load_data():
        """
        #### Load Data Task
        Load data from CSV and convert to TimeSeriesDataFrame.
        """
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame

        logging.info("Starting load_data: Loading data from CSV")
        try:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            logging.info(f"Data loaded: {df.shape}")
            df = df.rename(columns={"store_id": "item_id", "order_count": "target"})
            df = df.sort_values(["item_id", "timestamp"])
            data = TimeSeriesDataFrame.from_data_frame(df)
            logging.info(
                "Finished load_data: Data loaded and converted to TimeSeriesDataFrame"
            )
            # Convert back to pandas DataFrame for XCom serialization
            return data.to_data_frame()
        except Exception as e:
            logging.error(f"Error in load_data: {e}")
            raise

    @task(multiple_outputs=True)
    def train_test_split(data_df):
        """
        #### Train Test Split Task
        Split data into train and test sets.
        """
        from autogluon.timeseries import TimeSeriesDataFrame

        logging.info("Starting train_test_split: Splitting data into train and test")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        data = TimeSeriesDataFrame.from_data_frame(data_df)
        train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
        logging.info("Finished train_test_split: Data split completed")
        # Convert to pandas DataFrames for XCom serialization
        return {
            "train_data": train_data.to_data_frame(),
            "test_data": test_data.to_data_frame(),
        }

    @task()
    def train_model(train_data_df):
        """
        #### Train Model Task
        Train the model using AutoGluon and log to MLflow.
        """

        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame
        import shutil

        MODEL_DIR = "/tmp/autogluon_models"

        # Clean up existing model directory
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

        logging.info("Starting train_model: Training the model")
        try:
            # Convert pandas DataFrame back to TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("chronos_train_dag")  # Set experiment name

            mlflow.autolog(disable=True)

            from autogluon.timeseries import TimeSeriesPredictor

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                predictor = TimeSeriesPredictor(
                    prediction_length=PREDICTION_LENGTH, path=MODEL_DIR
                ).fit(
                    train_data=train_data,
                    hyperparameters={
                        "Chronos": [
                            {
                                "model_path": "bolt_small",
                                "ag_args": {"name_suffix": "ZeroShot"},
                            },
                            {
                                "model_path": "bolt_small",
                                "fine_tune": False,
                                "device": "cpu",
                                "ag_args": {"name_suffix": "ZeroShot_Fast"},
                            },
                        ]
                    },
                    time_limit=300,  # 60 seconds (1 minute)
                    enable_ensemble=False,
                )

                # Re-enable MLflow autologging after AutoGluon training
                mlflow.autolog()

                # Log hyperparameters
                mlflow.log_param("prediction_length", PREDICTION_LENGTH)
                mlflow.log_param("time_limit", 60)
                mlflow.log_param("enable_ensemble", False)

                # Get leaderboard and log metrics
                leaderboard = predictor.leaderboard()
                for idx, row in leaderboard.iterrows():
                    model_name = row["model"]
                    sanitized_model_name = model_name.replace("[", "_").replace(
                        "]", "_"
                    )
                    if "score_val" in row:
                        mlflow.log_metric(
                            f"{sanitized_model_name}_val_score", row["score_val"]
                        )
                    if "pred_time_val" in row:
                        mlflow.log_metric(
                            f"{sanitized_model_name}_pred_time_val",
                            row["pred_time_val"],
                        )

                # Log best model score
                best_score = leaderboard["score_val"].max()
                mlflow.log_metric("best_model_val_score", best_score)

                # Ensure predictor saved to MODEL_DIR
                predictor.save()
                mlflow.log_artifact(MODEL_DIR, "predictor")
        except Exception as e:
            logging.error(f"MLflow connection failed: {e}")
            raise
        logging.info("Finished train_model: Model training completed")
        # Return both model path and run_id for downstream tasks
        return {"model_path": MODEL_DIR, "run_id": run_id}

    @task()
    def predict(model_info, train_data_df):
        """
        #### Predict Task
        Make predictions using the trained model.
        """

        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os

        # Convert pandas DataFrame back to TimeSeriesDataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_info["model_path"])

        predictions = predictor.predict(train_data)
        logging.info("Finished predict: Predictions generated")

        # Log predictions to the same MLflow run
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=model_info["run_id"]):
            mlflow.log_artifact("/tmp/predictions.csv", "predictions")
        return predictions.to_data_frame()

    @task(multiple_outputs=True)
    def evaluate(model_info, test_data_df):
        """
        #### Evaluate Task
        Evaluate the model and log leaderboard.
        """

        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os

        # Convert pandas DataFrame back to TimeSeriesDataFrame
        test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_info["model_path"])

        leaderboard = predictor.leaderboard(test_data)
        leaderboard.to_csv("/tmp/leaderboard.csv")

        # Log leaderboard to the same MLflow run
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=model_info["run_id"]):
            mlflow.log_artifact("/tmp/leaderboard.csv", "leaderboard")

        logging.info("Finished evaluate: Evaluation completed")
        return {"leaderboard_path": "/tmp/leaderboard.csv"}

    @task()
    def visualize(model_info, data_df, predictions):
        """
        #### Visualize Task
        Generate and save visualization plot.
        """
        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import random
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        logging.info("Starting visualize: Generating visualization")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        data = TimeSeriesDataFrame.from_data_frame(data_df)
        predictions_ts = TimeSeriesDataFrame.from_data_frame(predictions)

        # Randomly select a subset of item_ids for visualization
        item_ids_to_visualize = random.sample(
            list(data.item_ids), min(10, len(data.item_ids))
        )  # Randomly select up to 10 items for visualization

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_info["model_path"])

        fig = predictor.plot(
            data=data,
            predictions=predictions_ts,
            item_ids=item_ids_to_visualize,  # Limit to randomly selected item_ids
            max_history_length=200,
        )
        fig.savefig("/tmp/forecast_plot.png")

        # Log visualization to the same MLflow run
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=model_info["run_id"]):
            mlflow.log_artifact("/tmp/forecast_plot.png", "forecast_plot")

        logging.info("Finished visualize: Visualization saved")

    @task()
    def save_model(model_info):
        """
        #### Save Model Task
        Save the model (additional saving if needed).
        """
        logging.info("Starting save_model: Saving the model")
        # Model already saved in train_model at model_path; no action needed by default
        logging.info(f"Model is available at: {model_info['model_path']}")
        logging.info("Finished save_model: Model saving completed")

    # Build the flow
    data = load_data()
    # Skip preprocess_data as it's not doing any actual preprocessing
    train_test_result = train_test_split(data)
    train_data = train_test_result["train_data"]
    test_data = train_test_result["test_data"]
    model_info = train_model(train_data)
    predictions = predict(model_info, train_data)
    leaderboard_path = evaluate(model_info, test_data)
    visualize(model_info, test_data, predictions)  # Use test data for visualization
    save_model(model_info)


# Invoke the DAG
chronos_train_dag()
