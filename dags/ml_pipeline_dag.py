import json
import logging
import os
from datetime import datetime, timedelta

import pendulum
from airflow.sdk import dag, task

# Global variables
DATA_PATH = '/Users/coldbrew_groom/Documents/order-platform-msa-train-pipeline/data/forecast_data_featured.csv'
PREDICTION_LENGTH = 48
MLFLOW_TRACKING_URI = 'http://localhost:5001'  # 로컬 MLflow 서버로 변경

@dag(
    dag_id="ml_pipeline_dag",
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
def ml_pipeline_dag():
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
            logging.info("Finished load_data: Data loaded and converted to TimeSeriesDataFrame")
            # Convert back to pandas DataFrame for XCom serialization
            return data.to_data_frame()
        except Exception as e:
            logging.error(f"Error in load_data: {e}")
            raise

    @task()
    def preprocess_data(data_df):
        """
        #### Preprocess Data Task
        Preprocess the data (e.g., handling missing values).
        """
        from autogluon.timeseries import TimeSeriesDataFrame
        
        logging.info("Starting preprocess_data: Preprocessing data")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        data = TimeSeriesDataFrame.from_data_frame(data_df)
        # Add preprocessing if needed, e.g., handling missing values
        logging.info("Finished preprocess_data: Data preprocessing completed")
        return data

    @task(multiple_outputs=True)
    def train_test_split(data):
        """
        #### Train Test Split Task
        Split data into train and test sets.
        """
        logging.info("Starting train_test_split: Splitting data into train and test")
        train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
        logging.info("Finished train_test_split: Data split completed")
        # Convert to pandas DataFrames for XCom serialization
        return {"train_data": train_data.to_data_frame(), "test_data": test_data.to_data_frame()}

    @task()
    def train_model(train_data_df):
        """
        #### Train Model Task
        Train the model using AutoGluon and log to MLflow.
        """
        import mlflow
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        
        logging.info("Starting train_model: Training the model")
        try:
            # Convert pandas DataFrame back to TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.autolog()
            
            with mlflow.start_run():
                predictor = TimeSeriesPredictor(prediction_length=PREDICTION_LENGTH).fit(
                    train_data=train_data,
                    hyperparameters={
                        "Chronos": [
                            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
                            {
                                "model_path": "bolt_small",
                                "fine_tune": True,
                                "ag_args": {"name_suffix": "FineTuned"},
                            },
                        ]
                    },
                    time_limit=10,  # 5 minutes
                    enable_ensemble=False,
                )
                predictor.save('/tmp/predictor')
                mlflow.log_artifact('/tmp/predictor', 'predictor')
        except Exception as e:
            logging.error(f"MLflow connection failed: {e}")
            raise
        logging.info("Finished train_model: Model training completed")
        return predictor

    @task()
    def predict(predictor, train_data_df):
        """
        #### Predict Task
        Make predictions using the trained model.
        """
        from autogluon.timeseries import TimeSeriesDataFrame
        
        logging.info("Starting predict: Making predictions")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)
        predictions = predictor.predict(train_data)
        logging.info("Finished predict: Predictions generated")
        return predictions

    @task()
    def evaluate(predictor, test_data_df):
        """
        #### Evaluate Task
        Evaluate the model and log leaderboard.
        """
        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame
        
        logging.info("Starting evaluate: Evaluating the model")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)
        leaderboard = predictor.leaderboard(test_data)
        leaderboard.to_csv('/tmp/leaderboard.csv')
        mlflow.log_artifact('/tmp/leaderboard.csv', 'leaderboard')
        logging.info("Finished evaluate: Evaluation completed")
        return leaderboard

    @task()
    def visualize(predictor, data_df, predictions):
        """
        #### Visualize Task
        Generate and save visualization plot.
        """
        import mlflow
        from autogluon.timeseries import TimeSeriesDataFrame
        
        logging.info("Starting visualize: Generating visualization")
        # Convert pandas DataFrame back to TimeSeriesDataFrame
        data = TimeSeriesDataFrame.from_data_frame(data_df)
        fig = predictor.plot(
            data=data,
            predictions=predictions,
            item_ids=data.item_ids[:5],  # Limit to first 5 for visualization
            max_history_length=200,
        )
        fig.savefig('/tmp/forecast_plot.png')
        mlflow.log_artifact('/tmp/forecast_plot.png', 'forecast_plot')
        logging.info("Finished visualize: Visualization saved")

    @task()
    def save_model(predictor):
        """
        #### Save Model Task
        Save the model (additional saving if needed).
        """
        logging.info("Starting save_model: Saving the model")
        # Model already saved in train_model, but can add additional saving if needed
        logging.info("Finished save_model: Model saving completed")

    # Build the flow
    data = load_data()
    preprocessed_data = preprocess_data(data)
    train_test_result = train_test_split(preprocessed_data)
    train_data = train_test_result['train_data']
    test_data = train_test_result['test_data']
    predictor = train_model(train_data)
    predictions = predict(predictor, train_data)
    leaderboard = evaluate(predictor, test_data)
    visualize(predictor, preprocessed_data, predictions)
    save_model(predictor)

# Invoke the DAG
ml_pipeline_dag()
