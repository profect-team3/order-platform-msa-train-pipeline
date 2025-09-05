import pendulum
from airflow.sdk import DAG
from airflow.sdk import task
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import mlflow
import os

# Global variables
DATA_PATH = '../data/forecast_data_featured.csv'
PREDICTION_LENGTH = 48
MLFLOW_TRACKING_URI = 'http://mlflow:5000'  # Adjust if needed

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"store_id": "item_id", "order_count": "target"})
    df = df.sort_values(["item_id", "timestamp"])
    data = TimeSeriesDataFrame.from_data_frame(df)
    data.to_pickle('/tmp/data.pkl')

def preprocess_data():
    data = pd.read_pickle('/tmp/data.pkl')
    # Add preprocessing if needed, e.g., handling missing values
    data.to_pickle('/tmp/data_preprocessed.pkl')

def train_test_split():
    data = pd.read_pickle('/tmp/data_preprocessed.pkl')
    train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
    train_data.to_pickle('/tmp/train_data.pkl')
    test_data.to_pickle('/tmp/test_data.pkl')

def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.autolog()
    train_data = pd.read_pickle('/tmp/train_data.pkl')
    
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
            time_limit=300,  # 5 minutes
            enable_ensemble=False,
        )
        predictor.save('/tmp/predictor')
        mlflow.log_artifact('/tmp/predictor', 'predictor')

def predict():
    predictor = TimeSeriesPredictor.load('/tmp/predictor')
    train_data = pd.read_pickle('/tmp/train_data.pkl')
    predictions = predictor.predict(train_data)
    predictions.to_pickle('/tmp/predictions.pkl')

def evaluate():
    predictor = TimeSeriesPredictor.load('/tmp/predictor')
    test_data = pd.read_pickle('/tmp/test_data.pkl')
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv('/tmp/leaderboard.csv')
    mlflow.log_artifact('/tmp/leaderboard.csv', 'leaderboard')

def visualize():
    predictor = TimeSeriesPredictor.load('/tmp/predictor')
    data = pd.read_pickle('/tmp/data_preprocessed.pkl')
    predictions = pd.read_pickle('/tmp/predictions.pkl')
    # Save plot as image
    fig = predictor.plot(
        data=data,
        predictions=predictions,
        item_ids=data.item_ids[:5],  # Limit to first 5 for visualization
        max_history_length=200,
    )
    fig.savefig('/tmp/forecast_plot.png')
    mlflow.log_artifact('/tmp/forecast_plot.png', 'forecast_plot')

def save_model():
    # Model already saved in train_model, but can add additional saving if needed
    pass

with DAG(
    dag_id="ml_pipeline_dag",
    schedule=None,
    start_date=pendulum.datetime(2025, 9, 5, tz="UTC"),
    catchup=False,
    tags=["ml", "pipeline"],
) as dag:

    @task()
    def task_load_data():
        load_data()

    @task()
    def task_preprocess_data():
        preprocess_data()

    @task()
    def task_train_test_split():
        train_test_split()

    @task()
    def task_train_model():
        train_model()

    @task()
    def task_predict():
        predict()

    @task()
    def task_evaluate():
        evaluate()

    @task()
    def task_visualize():
        visualize()

    @task()
    def task_save_model():
        save_model()

    # Dependencies
    task_load_data() >> task_preprocess_data() >> task_train_test_split() >> task_train_model() >> task_predict() >> task_evaluate() >> task_visualize() >> task_save_model()
