import json
import logging
import os
import random
from datetime import datetime, timedelta

import pendulum
from airflow.decorators import dag, task

# Global variables
PREDICTION_LENGTH = 24

@dag(
    dag_id="chronos_train_dag_no_mlflow",
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
    }
)
def chronos_train_dag_no_mlflow():
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
        Load data from GCS CSV and convert to TimeSeriesDataFrame.
        """
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        OBJECT_NAME = os.environ.get('OBJECT_NAME')

        if not BUCKET_NAME or not OBJECT_NAME:
            raise ValueError("BUCKET_NAME or OBJECT_NAME environment variables not set")

        logging.info("Starting load_data: Loading data from GCS")
        try:
            hook = GCSHook()
            local_path = '/tmp/forecast_data_featured.csv'
            hook.download(
                bucket_name=BUCKET_NAME,
                object_name=OBJECT_NAME,
                filename=local_path
            )
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Data file not found after download: {local_path}")
            df = pd.read_csv(local_path)
            logging.info(f"Data loaded: {df.shape}")
            df = df.rename(columns={"store_id": "item_id", "real_order_quantity": "target"})
            df = df.sort_values(["item_id", "timestamp"])
            data = TimeSeriesDataFrame.from_data_frame(df)
            logging.info(f"Data loaded: {data.shape}")
            logging.info("Finished load_data: Data loaded and converted to TimeSeriesDataFrame")
            # Clean up local file
            os.remove(local_path)
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
        return {"train_data": train_data.to_data_frame(), "test_data": test_data.to_data_frame()}

    @task()
    def train_model(train_data_df):
        """
        #### Train Model Task
        Train the model using AutoGluon.
        """
        from autogluon.timeseries import TimeSeriesDataFrame
        import shutil

        MODEL_DIR = '/tmp/autogluon_models'

        # Clean up existing model directory
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

        logging.info("Starting train_model: Training the model")
        try:
            # Convert pandas DataFrame back to TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

            from autogluon.timeseries import TimeSeriesPredictor

            predictor = TimeSeriesPredictor(prediction_length=PREDICTION_LENGTH, path=MODEL_DIR).fit(
                train_data=train_data,
                hyperparameters={
                    "Chronos": [
                        {"model_path": "bolt_small",
                         "ag_args": {"name_suffix": "ZeroShot"}
                         },
                        {
                            "model_path": "bolt_small",
                            "fine_tune": True,
                            "device": "cpu",
                            "ag_args": {"name_suffix": "ZeroShot_Fast"},
                        },
                    ]
                },
                time_limit=10,  # 300 seconds (5 minutes)
                enable_ensemble=False
            )

            # Ensure predictor saved to MODEL_DIR
            predictor.save()
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise
        logging.info("Finished train_model: Model training completed")
        # Return model path for downstream tasks
        return MODEL_DIR

    @task()
    def predict(model_path, train_data_df):
        """
        #### Predict Task
        Make predictions using the trained model.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os

        logging.info("Starting predict: Making predictions")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_path)

        predictions = predictor.predict(train_data)
        logging.info("Finished predict: Predictions generated")

        return predictions.to_data_frame()

    @task(multiple_outputs=True)
    def evaluate(model_path, test_data_df):
        """
        #### Evaluate Task
        Evaluate the model and log leaderboard.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os

        logging.info("Starting evaluate: Evaluating the model")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_path)

        leaderboard = predictor.leaderboard(test_data)
        leaderboard.to_csv('/tmp/leaderboard.csv')

        logging.info("Finished evaluate: Evaluation completed")
        return {"leaderboard_path": '/tmp/leaderboard.csv'}

    @task()
    def visualize(model_path, data_df, predictions):
        """
        #### Visualize Task
        Generate and save visualization plot.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import random
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        logging.info("Starting visualize: Generating visualization")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        data = TimeSeriesDataFrame.from_data_frame(data_df)
        predictions_ts = TimeSeriesDataFrame.from_data_frame(predictions)

        # Randomly select a subset of item_ids for visualization
        item_ids_to_visualize = random.sample(list(data.item_ids), min(10, len(data.item_ids)))  # Randomly select up to 10 items for visualization

        # Load the AutoGluon TimeSeriesPredictor from the given path
        predictor = TimeSeriesPredictor.load(model_path)

        fig = predictor.plot(
            data=data,
            predictions=predictions_ts,
            item_ids=item_ids_to_visualize,  # Limit to randomly selected item_ids
            max_history_length=200,
        )
        fig.savefig('/tmp/forecast_plot.png')

        logging.info("Finished visualize: Visualization saved")

    @task()
    def save_model(model_path):
        """
        #### Save Model Task
        Save the model (additional saving if needed).
        """
        logging.info("Starting save_model: Saving the model")
        # Model already saved in train_model at model_path; no action needed by default
        logging.info(f"Model is available at: {model_path}")
        logging.info("Finished save_model: Model saving completed")

    # Build the flow
    data = load_data()
    # Skip preprocess_data as it's not doing any actual preprocessing
    train_test_result = train_test_split(data)
    train_data = train_test_result['train_data']
    test_data = train_test_result['test_data']
    model_path = train_model(train_data)
    predictions = predict(model_path, train_data)
    leaderboard_path = evaluate(model_path, test_data)
    visualize(model_path, test_data, predictions)  # Use test data for visualization
    save_model(model_path)

# Invoke the DAG
chronos_train_dag_no_mlflow()