import json
import logging
import os
import random
import shutil
from datetime import datetime, timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.google.cloud.operators.mlengine import MLEngineStartTrainingJobOperator # Import the operator

# Global variables
PREDICTION_LENGTH = 24

# Define AI Platform related variables
# These should ideally be Airflow Variables or fetched from a config
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-gcp-project-id') # Replace with your GCP Project ID
GCS_BUCKET_FOR_TRAINER = os.environ.get('GCS_BUCKET_FOR_TRAINER', 'your-trainer-bucket') # Replace with a GCS bucket for your trainer package
TRAINER_PACKAGE_PATH = 'trainer/autogluon_trainer-0.1.tar.gz' # Assuming you'll package your trainer and upload it here
TRAINER_URI = f"gs://{GCS_BUCKET_FOR_TRAINER}/{TRAINER_PACKAGE_PATH}"
TRAINER_PY_MODULE = 'trainer.task' # Corresponds to trainer/task.py

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
    def load_data_gcs_info():
        """
        #### Load Data GCS Info Task
        Returns GCS bucket and object name for data.
        """
        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        OBJECT_NAME = os.environ.get('OBJECT_NAME')

        if not BUCKET_NAME or not OBJECT_NAME:
            raise ValueError("BUCKET_NAME or OBJECT_NAME environment variables not set")
        return {"bucket_name": BUCKET_NAME, "object_name": OBJECT_NAME}

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

    # The train_model task is replaced by MLEngineStartTrainingJobOperator
    # This is defined outside the taskflow to be directly callable
    ai_platform_training_job_operator = MLEngineStartTrainingJobOperator(
        task_id="start_ai_platform_training",
        project_id=PROJECT_ID,
        region="us-central1", # You can make this configurable
        job_id=f"autogluon-training-job-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 10000)}", # Unique job ID
        runtime_version="2.11", # Use a recent runtime version that supports Python 3.9+
        python_version="3.9", # Match with your trainer's Python version
        job_dir=f"gs://{GCS_BUCKET_FOR_TRAINER}/ai_platform_jobs/autogluon_training_output/", # GCS path for job output
        package_uris=[TRAINER_URI],
        training_python_module=TRAINER_PY_MODULE,
        training_args=[
            f"--data-bucket-name={os.environ.get('BUCKET_NAME')}",
            f"--data-object-name={os.environ.get('OBJECT_NAME')}",
            f"--prediction-length={PREDICTION_LENGTH}"
        ],
        labels={"job_type": "autogluon_training"},
    )

    @task()
    def get_model_gcs_path(ai_platform_job_output_dict): # Changed parameter name to avoid conflict with operator instance
        """
        Extracts the model GCS path from AI Platform training job output.
        """
        # The job_dir is where the model is saved by the AI Platform job
        # The MLEngineStartTrainingJobOperator returns the job_id and other info.
        # We need to construct the full GCS path to the model from job_dir.
        # Assuming the trainer saves the model directly to job_dir.
        # The job_dir is passed as an argument to the trainer.
        # The operator's output might not directly contain the job_dir in a simple way.
        # Let's assume the job_dir is known from the operator's definition.
        # A more robust way would be to have the trainer output the exact model path.

        # For now, we'll reconstruct it based on the job_dir defined in the operator.
        # This is a simplification. In a real scenario, the trainer might output a specific model artifact path.
        # The ai_platform_job_output_dict contains the job details, including 'job_dir' if it was set.
        # However, the operator itself has the job_dir property.
        # We can directly access the job_dir from the operator instance if it's defined outside the taskflow.
        model_gcs_path = ai_platform_job_output_dict['job_dir'] # Access job_dir from the XCom pushed by the operator
        logging.info(f"Model GCS path from AI Platform job: {model_gcs_path}")
        return model_gcs_path

    @task()
    def predict(model_gcs_path, train_data_df):
        """
        #### Predict Task
        Make predictions using the trained model from GCS.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import shutil
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        LOCAL_MODEL_PATH = '/tmp/downloaded_autogluon_models'

        logging.info("Starting predict: Making predictions")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(f"Downloading model from {model_gcs_path} to {LOCAL_MODEL_PATH}")
            hook = GCSHook()
            # Parse model_gcs_path to get bucket name and prefix
            if not model_gcs_path.startswith("gs://"):
                raise ValueError(f"model_gcs_path must be a GCS path starting with 'gs://': {model_gcs_path}")

            path_parts = model_gcs_path[len("gs://"):]
            gcs_bucket_name = path_parts.split("/", 1)[0]
            gcs_prefix = path_parts.split("/", 1)[1] if "/" in path_parts else ""

            blobs = hook.list(bucket_name=gcs_bucket_name, prefix=gcs_prefix)
            for blob_name in blobs:
                relative_path = os.path.relpath(blob_name, gcs_prefix)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                hook.download(
                    bucket_name=gcs_bucket_name,
                    object_name=blob_name,
                    filename=local_file_path
                )
                logging.info(f"Downloaded {blob_name} to {local_file_path}")

            # Convert pandas DataFrame back to TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

            # Load the AutoGluon TimeSeriesPredictor from the downloaded path
            predictor = TimeSeriesPredictor.load(LOCAL_MODEL_PATH)

            predictions = predictor.predict(train_data)
            logging.info("Finished predict: Predictions generated")

            return predictions.to_data_frame()
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
        finally:
            # Clean up downloaded local model directory
            if os.path.exists(LOCAL_MODEL_PATH):
                shutil.rmtree(LOCAL_MODEL_PATH)
                logging.info(f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}")


    @task(multiple_outputs=True)
    def evaluate(model_gcs_path, test_data_df):
        """
        #### Evaluate Task
        Evaluate the model from GCS and log leaderboard.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import shutil
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        LOCAL_MODEL_PATH = '/tmp/downloaded_autogluon_models_eval' # Use a different path for evaluation

        logging.info("Starting evaluate: Evaluating the model")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(f"Downloading model from {model_gcs_path} to {LOCAL_MODEL_PATH}")
            hook = GCSHook()
            # Parse model_gcs_path to get bucket name and prefix
            if not model_gcs_path.startswith("gs://"):
                raise ValueError(f"model_gcs_path must be a GCS path starting with 'gs://': {model_gcs_path}")

            path_parts = model_gcs_path[len("gs://"):]
            gcs_bucket_name = path_parts.split("/", 1)[0]
            gcs_prefix = path_parts.split("/", 1)[1] if "/" in path_parts else ""

            blobs = hook.list(bucket_name=gcs_bucket_name, prefix=gcs_prefix)
            for blob_name in blobs:
                relative_path = os.path.relpath(blob_name, gcs_prefix)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                hook.download(
                    bucket_name=gcs_bucket_name,
                    object_name=blob_name,
                    filename=local_file_path
                )
                logging.info(f"Downloaded {blob_name} to {local_file_path}")

            # Convert pandas DataFrame back to TimeSeriesDataFrame
            test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)

            # Load the AutoGluon TimeSeriesPredictor from the downloaded path
            predictor = TimeSeriesPredictor.load(LOCAL_MODEL_PATH)

            leaderboard = predictor.leaderboard(test_data)
            leaderboard.to_csv('/tmp/leaderboard.csv')

            logging.info("Finished evaluate: Evaluation completed")
            return {"leaderboard_path": '/tmp/leaderboard.csv'}
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
        finally:
            # Clean up downloaded local model directory
            if os.path.exists(LOCAL_MODEL_PATH):
                shutil.rmtree(LOCAL_MODEL_PATH)
                logging.info(f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}")

    @task()
    def visualize(model_gcs_path, data_df, predictions):
        """
        #### Visualize Task
        Generate and save visualization plot using the model from GCS.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import random
        import matplotlib
        import shutil
        from airflow.providers.google.cloud.hooks.gcs import GCSHook
        matplotlib.use('Agg')  # Use non-interactive backend

        LOCAL_MODEL_PATH = '/tmp/downloaded_autogluon_models_viz' # Use a different path for visualization

        logging.info("Starting visualize: Generating visualization")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(f"Downloading model from {model_gcs_path} to {LOCAL_MODEL_PATH}")
            hook = GCSHook()
            # Parse model_gcs_path to get bucket name and prefix
            if not model_gcs_path.startswith("gs://"):
                raise ValueError(f"model_gcs_path must be a GCS path starting with 'gs://': {model_gcs_path}")

            path_parts = model_gcs_path[len("gs://"):]
            gcs_bucket_name = path_parts.split("/", 1)[0]
            gcs_prefix = path_parts.split("/", 1)[1] if "/" in path_parts else ""

            blobs = hook.list(bucket_name=gcs_bucket_name, prefix=gcs_prefix)
            for blob_name in blobs:
                relative_path = os.path.relpath(blob_name, gcs_prefix)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                hook.download(
                    bucket_name=gcs_bucket_name,
                    object_name=blob_name,
                    filename=local_file_path
                )
                logging.info(f"Downloaded {blob_name} to {local_file_path}")

            # Convert pandas DataFrame back to TimeSeriesDataFrame
            data = TimeSeriesDataFrame.from_data_frame(data_df)
            predictions_ts = TimeSeriesDataFrame.from_data_frame(predictions)

            # Randomly select a subset of item_ids for visualization
            item_ids_to_visualize = random.sample(list(data.item_ids), min(10, len(data.item_ids)))  # Randomly select up to 10 items for visualization

            # Load the AutoGluon TimeSeriesPredictor from the downloaded path
            predictor = TimeSeriesPredictor.load(LOCAL_MODEL_PATH)

            fig = predictor.plot(
                data=data,
                predictions=predictions_ts,
                item_ids=item_ids_to_visualize,  # Limit to randomly selected item_ids
                max_history_length=200,
            )
            fig.savefig('/tmp/forecast_plot.png')

            logging.info("Finished visualize: Visualization saved")
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            raise
        finally:
            # Clean up downloaded local model directory
            if os.path.exists(LOCAL_MODEL_PATH):
                shutil.rmtree(LOCAL_MODEL_PATH)
                logging.info(f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}")

    @task()
    def save_model(model_gcs_path):
        """
        #### Save Model Task
        Confirm model is saved to GCS.
        """
        logging.info("Starting save_model: Confirming model saving to GCS")
        logging.info(f"Model is expected to be available in GCS at: {model_gcs_path}")
        logging.info("Finished save_model: Model saving confirmation completed")

    # Build the flow
    gcs_data_info = load_data_gcs_info()
    data = load_data()
    train_test_result = train_test_split(data)
    train_data = train_test_result['train_data']
    test_data = train_test_result['test_data']

    # Start AI Platform Training Job
    ai_platform_training_job_output = ai_platform_training_job_operator.execute(context={})

    model_gcs_path = get_model_gcs_path(ai_platform_training_job_output)

    predictions = predict(model_gcs_path, train_data)
    leaderboard_path = evaluate(model_gcs_path, test_data)
    visualize(model_gcs_path, test_data, predictions)
    save_model(model_gcs_path)

# Invoke the DAG
chronos_train_dag_no_mlflow()