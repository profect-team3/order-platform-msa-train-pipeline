import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import mlflow
import os
import logging
import random
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

# Ensure /tmp directory exists
os.makedirs('/tmp', exist_ok=True)

# Global variables
DATA_PATH = '/Users/coldbrew_groom/Documents/order-platform-msa-train-pipeline/data/forecast_data_featured.csv'
PREDICTION_LENGTH = 24
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')  # MLflow server on port 5001
S3_UPLOAD = os.getenv('S3_UPLOAD', 'false').lower() == 'true'
S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', 'models/')

def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str) -> None:
    try:
        s3_client = boto3.client('s3')
        for root, _, files in os.walk(local_dir):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(local_path, start=local_dir)
                s3_key = f"{prefix.rstrip('/')}/{rel_path}"
                s3_client.upload_file(local_path, bucket, s3_key)
                logger.info(f"Uploaded to s3://{bucket}/{s3_key}")
    except (BotoCoreError, NoCredentialsError) as e:
        logger.error(f"S3 upload failed: {e}")
        raise

def load_data():
    """Load data from CSV and convert to TimeSeriesDataFrame."""
    logger.info("Starting load_data: Loading data from CSV")
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded: {df.shape}")
        df = df.rename(columns={"store_id": "item_id", "order_count": "target"})
        df = df.sort_values(["item_id", "timestamp"])
        data = TimeSeriesDataFrame.from_data_frame(df)
        logger.info("Finished load_data: Data loaded and converted to TimeSeriesDataFrame")
        
        # Save as pandas DataFrame for consistency
        data.to_data_frame().to_pickle('/tmp/data.pkl')
        logger.info("Data saved to /tmp/data.pkl")
        return data.to_data_frame()
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        raise

def train_test_split(data_df):
    """Split data into train and test sets."""
    logger.info("Starting train_test_split: Splitting data into train and test")
    # Convert pandas DataFrame back to TimeSeriesDataFrame
    data = TimeSeriesDataFrame.from_data_frame(data_df)
    train_data, test_data = data.train_test_split(PREDICTION_LENGTH)
    logger.info("Finished train_test_split: Data split completed")
    
    # Save as pandas DataFrames
    train_data.to_data_frame().to_pickle('/tmp/train_data.pkl')
    test_data.to_data_frame().to_pickle('/tmp/test_data.pkl')
    logger.info("Train and test data saved")
    return train_data.to_data_frame(), test_data.to_data_frame()

def train_model(train_data_df):
    """Train the model using AutoGluon and log to MLflow."""
    logger.info("Starting train_model: Training the model")
    try:
        train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ml_pipeline_experiment")  # Set experiment name

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
        
        # Log hyperparameters
        mlflow.log_param("prediction_length", PREDICTION_LENGTH)
        mlflow.log_param("time_limit", 10)
        mlflow.log_param("enable_ensemble", False)
        
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
        
        predictor.save()

        mlflow.log_artifact(predictor.path, 'predictor')

        # Optional: upload trained predictor directory to S3
        if S3_UPLOAD and S3_BUCKET:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run is not None else 'no-run'
            upload_prefix = f"{S3_PREFIX.rstrip('/')}/predictor/{run_id}"
            logger.info(f"Uploading predictor directory to s3://{S3_BUCKET}/{upload_prefix}")
            upload_dir_to_s3(predictor.path, S3_BUCKET, upload_prefix)
            # Record uploaded S3 URI as MLflow artifact and tag
            s3_uri_prefix = f"s3://{S3_BUCKET}/{upload_prefix}"
            try:
                s3_uri_file = '/tmp/s3_model_uri.txt'
                with open(s3_uri_file, 'w') as f:
                    f.write(s3_uri_prefix)
                mlflow.log_artifact(s3_uri_file, 's3')
                mlflow.set_tag('s3_model_uri', s3_uri_prefix)
                logger.info(f"Logged S3 URI to MLflow: {s3_uri_prefix}")
            except Exception as e:
                logger.error(f"Failed to log S3 URI to MLflow: {e}")
        
        
    except Exception as e:
        logger.error(f"MLflow connection failed: {e}")
        raise
    logger.info("Finished train_model: Model training completed")
    return predictor

def predict(predictor, train_data_df):
    """Make predictions using the trained model."""
    logger.info("Starting predict: Making predictions")
    # Convert pandas DataFrame back to TimeSeriesDataFrame
    train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)
    predictions = predictor.predict(train_data)
    logger.info("Finished predict: Predictions generated")
    
    # Save predictions
    predictions.to_pickle('/tmp/predictions.pkl')
    
    # Log predictions to MLflow
    predictions.to_csv('/tmp/predictions.csv')
    mlflow.log_artifact('/tmp/predictions.csv', 'predictions')
    
    logger.info("Predictions saved and logged to MLflow")
    return predictions

def evaluate(predictor, test_data_df):
    """Evaluate the model and log leaderboard."""
    logger.info("Starting evaluate: Evaluating the model")
    # Convert pandas DataFrame back to TimeSeriesDataFrame
    test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv('/tmp/leaderboard.csv')
    
    # Log leaderboard to MLflow
    mlflow.log_artifact('/tmp/leaderboard.csv', 'leaderboard')
    
    logger.info("Evaluation completed and leaderboard saved to MLflow")
    return leaderboard

def visualize(predictor, data_df, predictions):
    """Generate and save visualization plot."""
    logger.info("Starting visualize: Generating visualization")
    # Convert pandas DataFrame back to TimeSeriesDataFrame
    data = TimeSeriesDataFrame.from_data_frame(data_df)
    
    # Randomly select a subset of item_ids for visualization
    item_ids_to_visualize = random.sample(list(data.item_ids), min(10, len(data.item_ids)))  # Randomly select up to 10 items for visualization
    
    fig = predictor.plot(
        data=data,
        predictions=predictions,
        item_ids=item_ids_to_visualize,  # Limit to randomly selected item_ids
        max_history_length=200,
    )
    fig.savefig('/tmp/forecast_plot.png')
    
    # Log visualization to MLflow
    mlflow.log_artifact('/tmp/forecast_plot.png', 'forecast_plot')
    
    logger.info("Visualization saved to MLflow")

def save_model(predictor):
    """Save the model (additional saving if needed)."""
    logger.info("Starting save_model: Saving the model")
    # Model already saved in train_model, but can add additional saving if needed
    logger.info("Finished save_model: Model saving completed")

if __name__ == "__main__":
    # Run the pipeline
    logger.info("Starting ML pipeline execution")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ml_pipeline_experiment")  # Set experiment name
    
    with mlflow.start_run():
        data_df = load_data()
        # Skip preprocess_data as it's not doing any actual preprocessing
        train_data_df, test_data_df = train_test_split(data_df)
        predictor = train_model(train_data_df)
        predictions = predict(predictor, train_data_df)
        leaderboard = evaluate(predictor, test_data_df)
        visualize(predictor, test_data_df, predictions)  # Use test data for visualization
        save_model(predictor)
    
    logger.info("ML pipeline execution completed. Check MLflow UI at http://localhost:5001")
