import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import mlflow
import os

# Global variables
DATA_PATH = '/Users/coldbrew_groom/Documents/order-platform-msa-train-pipeline/data/forecast_data_featured.csv'
PREDICTION_LENGTH = 48
MLFLOW_TRACKING_URI = 'http://localhost:5001'  # MLflow server on port 5001

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"store_id": "item_id", "order_count": "target"})
    df = df.sort_values(["item_id", "timestamp"])
    data = TimeSeriesDataFrame.from_data_frame(df)
    data.to_pickle('/tmp/data.pkl')
    print("Data loaded and saved to /tmp/data.pkl")

def preprocess_data():
    data = pd.read_pickle('/tmp/data.pkl')
    # Add preprocessing if needed, e.g., handling missing values
    data.to_pickle('/tmp/data_preprocessed.pkl')
    print("Data preprocessed and saved to /tmp/data_preprocessed.pkl")

def train_test_split():
    data = pd.read_pickle('/tmp/data_preprocessed.pkl')
    train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
    train_data.to_pickle('/tmp/train_data.pkl')
    test_data.to_pickle('/tmp/test_data.pkl')
    print("Train-test split completed")

def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("chronos_forecasting_pipeline")
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
        
        # Log hyperparameters
        mlflow.log_param("prediction_length", PREDICTION_LENGTH)
        mlflow.log_param("time_limit", 300)
        mlflow.log_param("enable_ensemble", False)
        
        predictor.save('/tmp/predictor')
        mlflow.log_artifact('/tmp/predictor', 'predictor')
        print("Model trained and saved to MLflow")

def predict():
    predictor = TimeSeriesPredictor.load('/tmp/predictor')
    train_data = pd.read_pickle('/tmp/train_data.pkl')
    predictions = predictor.predict(train_data)
    predictions.to_pickle('/tmp/predictions.pkl')
    
    # Log predictions to MLflow
    with mlflow.start_run():
        predictions.to_csv('/tmp/predictions.csv')
        mlflow.log_artifact('/tmp/predictions.csv', 'predictions')
        mlflow.log_param("prediction_length", len(predictions))
    
    print("Predictions generated and saved")

def evaluate():
    predictor = TimeSeriesPredictor.load('/tmp/predictor')
    test_data = pd.read_pickle('/tmp/test_data.pkl')
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv('/tmp/leaderboard.csv')
    
    # Log metrics to MLflow
    with mlflow.start_run():
        for idx, row in leaderboard.iterrows():
            model_name = row['model']
            score = row['score_test']
            mlflow.log_metric(f"{model_name}_score", score)
            print(f"Logged metric for {model_name}: {score}")
        
        # Log best model score
        best_score = leaderboard['score_test'].max()
        mlflow.log_metric("best_model_score", best_score)
        
        mlflow.log_artifact('/tmp/leaderboard.csv', 'leaderboard')
    
    print("Evaluation completed and leaderboard saved to MLflow")

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
    
    # Log visualization to MLflow
    with mlflow.start_run():
        mlflow.log_artifact('/tmp/forecast_plot.png', 'forecast_plot')
    
    print("Visualization saved to MLflow")

def save_model():
    # Model already saved in train_model, but can add additional saving if needed
    print("Model save completed (already done in train_model)")

if __name__ == "__main__":
    # Run the pipeline
    load_data()
    preprocess_data()
    train_test_split()
    train_model()
    predict()
    evaluate()
    visualize()
    save_model()
    print("ML pipeline execution completed. Check MLflow UI at http://localhost:5001")
