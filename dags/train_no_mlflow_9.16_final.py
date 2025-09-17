import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)
logging.getLogger("autogluon").setLevel(logging.DEBUG)
from datetime import datetime, timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.providers.google.cloud.hooks.gcs import GCSHook  # Added GCSHook import

# Global variables
PREDICTION_LENGTH = 24


@dag(
    dag_id="train_no_mlflow_9.16_final",
    schedule=None,
    start_date=pendulum.datetime(2025, 9, 5, tz="UTC"),
    schedule_interval='@weekly',
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
def chronos_train_dag_no_mlflow():
    """
    ### ML Pipeline DAG Documentation
    This is a simple ML pipeline example which demonstrates the use of
    the TaskFlow API for loading data, preprocessing, training, predicting,
    evaluating, visualizing, and saving a model.
    """
    @task()
    def _preprocess_data():
        """
        Function to run the preprocessing logic.
        """
        import pandas as pd
        raw_data_path = 'gs://us-central1-mlpipeline-comp-9bf7861c-bucket/data/consumed_orders.csv'
        output_path = 'gs://us-central1-mlpipeline-comp-9bf7861c-bucket/data/train_data.csv'

        logging.info(f"Reading raw data from {raw_data_path}")
        try:
            df = pd.read_csv(raw_data_path)
        except Exception as e:
            logging.error(f"Failed to read raw data file: {e}")
            raise

        logging.info("Preprocessing data...")

        logging.info(f"Initial df columns: {df.columns.tolist()}")
        df.rename(columns={'totalPrice': 'real_sales_revenue', 'storeId': 'store_id'}, inplace=True)
        logging.info(f"df columns after rename: {df.columns.tolist()}")
        df['real_sales_revenue'] = pd.to_numeric(df['real_sales_revenue'], errors='coerce').fillna(0)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df['timestamp'] = df['timestamp'].dt.floor('h')

        logging.info("Aggregating data to hourly statistics...")

        hourly_df = df.groupby([
            'timestamp', 'store_id', 'category_main', 'category_sub',
            'category_item', 'region', 'min_order_amount', 'avg_rating'
        ]).agg(
            real_sales_revenue=('real_sales_revenue', 'sum'),
            real_order_quantity=('store_id', 'count')
        ).reset_index()

        hourly_df['day_of_week'] = hourly_df['timestamp'].dt.dayofweek
        hourly_df['hour'] = hourly_df['timestamp'].dt.hour

        target_columns = [
            'timestamp', 'store_id', 'category_main', 'category_sub',
            'category_item', 'region', 'real_order_quantity', 'real_sales_revenue',
            'day_of_week', 'hour', 'min_order_amount', 'avg_rating'
        ]
        logging.info(f"hourly_df columns before final selection: {hourly_df.columns.tolist()}")
        hourly_df = hourly_df[target_columns]
        logging.info(f"hourly_df columns after final selection: {hourly_df.columns.tolist()}")

        hourly_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logging.info(f"Successfully preprocessed data and saved to {output_path}")

    @task()
    def load_data():
        """
        #### Load Data Task
        Load data from GCS CSV and convert to TimeSeriesDataFrame.
        """
        import pandas as pd
        from autogluon.timeseries import TimeSeriesDataFrame
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        BUCKET_NAME = os.environ.get("BUCKET_NAME")
        OBJECT_NAME = os.environ.get("OBJECT_NAME")

        if not BUCKET_NAME or not OBJECT_NAME:
            raise ValueError("BUCKET_NAME or OBJECT_NAME environment variables not set")

        logging.info("Starting load_data: Loading data from GCS")
        try:
            hook = GCSHook()
            local_path = "/tmp/forecast_data_featured.csv"
            hook.download(
                bucket_name=BUCKET_NAME, object_name=OBJECT_NAME, filename=local_path
            )
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Data file not found after download: {local_path}"
                )
            df = pd.read_csv(local_path)
            logging.info(f"Data loaded: {df.shape}")
            df = df.rename(
                columns={"store_id": "item_id", "real_order_quantity": "target"}
            )
            df = df.sort_values(["item_id", "timestamp"])
            data = TimeSeriesDataFrame.from_data_frame(df)
            logging.info(f"Data loaded: {data.shape}")
            logging.info(
                "Finished load_data: Data loaded and converted to TimeSeriesDataFrame"
            )
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
        return {
            "train_data": train_data.to_data_frame(),
            "test_data": test_data.to_data_frame(),
        }

    @task()
    def train_model(train_data_df):
        """
        #### Train Model Task
        Train the model using AutoGluon and upload to GCS.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import shutil
        import os
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        MODEL_DIR = "/tmp/autogluon_models"
        MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
        MODEL_OBJECT_NAME_PREFIX = os.environ.get(
            "MODEL_OBJECT_NAME_PREFIX"
        )  # e.g., 'autogluon_models/'

        if not MODEL_BUCKET_NAME or not MODEL_OBJECT_NAME_PREFIX:
            raise ValueError(
                "MODEL_BUCKET_NAME or MODEL_OBJECT_NAME_PREFIX environment variables not set"
            )

        # Clean up existing model directory
        if os.path.exists(MODEL_DIR):
            shutil.rmtree(MODEL_DIR)

        logging.info("Starting train_model: Training the model")
        try:
            # Convert pandas DataFrame back to TimeSeriesDataFrame
            train_data = TimeSeriesDataFrame.from_data_frame(train_data_df)

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
                time_limit=600,  # 300 seconds (5 minutes)
                enable_ensemble=False,
            )

            # Ensure predictor saved to MODEL_DIR
            predictor.save()

            # Upload model to GCS
            logging.info(
                f"Uploading model from {MODEL_DIR} to gs://{MODEL_BUCKET_NAME}/{MODEL_OBJECT_NAME_PREFIX}"
            )
            hook = GCSHook()
            for root, _, files in os.walk(MODEL_DIR):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    # Construct the GCS object name, preserving directory structure
                    relative_path = os.path.relpath(local_file_path, MODEL_DIR)
                    gcs_object_name = os.path.join(
                        MODEL_OBJECT_NAME_PREFIX, relative_path
                    )
                    hook.upload(
                        bucket_name=MODEL_BUCKET_NAME,
                        object_name=gcs_object_name,
                        filename=local_file_path,
                    )
                    logging.info(f"Uploaded {local_file_path} to {gcs_object_name}")

            logging.info("Model uploaded to GCS successfully.")

        except Exception as e:
            logging.error(f"Model training or upload failed: {e}")
            raise
        finally:
            # Clean up local model directory
            if os.path.exists(MODEL_DIR):
                shutil.rmtree(MODEL_DIR)
                logging.info(f"Cleaned up local model directory: {MODEL_DIR}")

        logging.info("Finished train_model: Model training and upload completed")
        # Return GCS path for downstream tasks
        return {
            "model_bucket_name": MODEL_BUCKET_NAME,
            "model_object_name_prefix": MODEL_OBJECT_NAME_PREFIX,
        }

    @task()
    def predict(model_info, train_data_df):  # Changed model_path to model_info
        """
        #### Predict Task
        Make predictions using the trained model from GCS.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import shutil
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        MODEL_BUCKET_NAME = model_info["model_bucket_name"]
        MODEL_OBJECT_NAME_PREFIX = model_info["model_object_name_prefix"]
        LOCAL_MODEL_PATH = "/tmp/downloaded_autogluon_models"

        logging.info("Starting predict: Making predictions")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(
                f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}"
            )
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(
                f"Downloading model from gs://{MODEL_BUCKET_NAME}/{MODEL_OBJECT_NAME_PREFIX} to {LOCAL_MODEL_PATH}"
            )
            hook = GCSHook()
            # List all blobs under the prefix and download them
            blobs = hook.list(
                bucket_name=MODEL_BUCKET_NAME, prefix=MODEL_OBJECT_NAME_PREFIX
            )
            for blob_name in blobs:
                if blob_name.endswith("/"):
                    continue
                # Construct local file path, preserving directory structure
                relative_path = os.path.relpath(blob_name, MODEL_OBJECT_NAME_PREFIX)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(
                    os.path.dirname(local_file_path), exist_ok=True
                )  # Create parent directories
                hook.download(
                    bucket_name=MODEL_BUCKET_NAME,
                    object_name=blob_name,
                    filename=local_file_path,
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
                logging.info(
                    f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}"
                )

    @task(multiple_outputs=True)
    def evaluate(model_info, test_data_df):  # Changed model_path to model_info
        """
        #### Evaluate Task
        Evaluate the model from GCS and log leaderboard.
        """
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import shutil
        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        MODEL_BUCKET_NAME = model_info["model_bucket_name"]
        MODEL_OBJECT_NAME_PREFIX = model_info["model_object_name_prefix"]
        LOCAL_MODEL_PATH = "/tmp/downloaded_autogluon_models_eval"  # Use a different path for evaluation

        logging.info("Starting evaluate: Evaluating the model")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(
                f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}"
            )
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(
                f"Downloading model from gs://{MODEL_BUCKET_NAME}/{MODEL_OBJECT_NAME_PREFIX} to {LOCAL_MODEL_PATH}"
            )
            hook = GCSHook()
            blobs = hook.list(
                bucket_name=MODEL_BUCKET_NAME, prefix=MODEL_OBJECT_NAME_PREFIX
            )
            for blob_name in blobs:
                if blob_name.endswith("/"):
                    continue
                relative_path = os.path.relpath(blob_name, MODEL_OBJECT_NAME_PREFIX)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                hook.download(
                    bucket_name=MODEL_BUCKET_NAME,
                    object_name=blob_name,
                    filename=local_file_path,
                )
                logging.info(f"Downloaded {blob_name} to {local_file_path}")

            # Convert pandas DataFrame back to TimeSeriesDataFrame
            test_data = TimeSeriesDataFrame.from_data_frame(test_data_df)

            # Load the AutoGluon TimeSeriesPredictor from the downloaded path
            predictor = TimeSeriesPredictor.load(LOCAL_MODEL_PATH)

            leaderboard = predictor.leaderboard(test_data)
            leaderboard.to_csv("/tmp/leaderboard.csv")

            logging.info("Finished evaluate: Evaluation completed")
            return {"leaderboard_path": "/tmp/leaderboard.csv"}
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise
        finally:
            # Clean up downloaded local model directory
            if os.path.exists(LOCAL_MODEL_PATH):
                shutil.rmtree(LOCAL_MODEL_PATH)
                logging.info(
                    f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}"
                )

    @task()
    def visualize(model_info, data_df, predictions):  # Changed model_path to model_info
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

        matplotlib.use("Agg")  # Use non-interactive backend

        MODEL_BUCKET_NAME = model_info["model_bucket_name"]
        MODEL_OBJECT_NAME_PREFIX = model_info["model_object_name_prefix"]
        LOCAL_MODEL_PATH = "/tmp/downloaded_autogluon_models_viz"  # Use a different path for visualization

        logging.info("Starting visualize: Generating visualization")

        # Clean up existing local model directory before download
        if os.path.exists(LOCAL_MODEL_PATH):
            shutil.rmtree(LOCAL_MODEL_PATH)
            logging.info(
                f"Cleaned up existing local model directory: {LOCAL_MODEL_PATH}"
            )
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

        try:
            # Download model from GCS
            logging.info(
                f"Downloading model from gs://{MODEL_BUCKET_NAME}/{MODEL_OBJECT_NAME_PREFIX} to {LOCAL_MODEL_PATH}"
            )
            hook = GCSHook()
            blobs = hook.list(
                bucket_name=MODEL_BUCKET_NAME, prefix=MODEL_OBJECT_NAME_PREFIX
            )
            for blob_name in blobs:
                if blob_name.endswith("/"):
                    continue
                relative_path = os.path.relpath(blob_name, MODEL_OBJECT_NAME_PREFIX)
                local_file_path = os.path.join(LOCAL_MODEL_PATH, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                hook.download(
                    bucket_name=MODEL_BUCKET_NAME,
                    object_name=blob_name,
                    filename=local_file_path,
                )
                logging.info(f"Downloaded {blob_name} to {local_file_path}")

            # Convert pandas DataFrame back to TimeSeriesDataFrame
            data = TimeSeriesDataFrame.from_data_frame(data_df)
            predictions_ts = TimeSeriesDataFrame.from_data_frame(predictions)

            # Randomly select a subset of item_ids for visualization
            item_ids_to_visualize = random.sample(
                list(data.item_ids), min(10, len(data.item_ids))
            )  # Randomly select up to 10 items for visualization

            # Load the AutoGluon TimeSeriesPredictor from the downloaded path
            predictor = TimeSeriesPredictor.load(LOCAL_MODEL_PATH)

            fig = predictor.plot(
                data=data,
                predictions=predictions_ts,
                item_ids=item_ids_to_visualize,  # Limit to randomly selected item_ids
                max_history_length=200,
            )
            fig.savefig("/tmp/forecast_plot.png")

            logging.info("Finished visualize: Visualization saved")
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            raise
        finally:
            # Clean up downloaded local model directory
            if os.path.exists(LOCAL_MODEL_PATH):
                shutil.rmtree(LOCAL_MODEL_PATH)
                logging.info(
                    f"Cleaned up downloaded local model directory: {LOCAL_MODEL_PATH}"
                )

    @task()
    def save_model(model_info):  # Changed model_path to model_info
        """
        #### Save Model Task
        Confirm model is saved to GCS.
        """
        logging.info("Starting save_model: Confirming model saving to GCS")
        model_bucket_name = model_info["model_bucket_name"]
        model_object_name_prefix = model_info["model_object_name_prefix"]
        logging.info(
            f"Model is expected to be available in GCS at: gs://{model_bucket_name}/{model_object_name_prefix}"
        )
        logging.info("Finished save_model: Model saving confirmation completed")

    # Build the flow
    data = load_data()
    # Skip preprocess_data as it's not doing any actual preprocessing
    train_test_result = train_test_split(data)
    train_data = train_test_result["train_data"]
    test_data = train_test_result["test_data"]
    model_info = train_model(train_data)  # Changed model_path to model_info
    predictions = predict(model_info, train_data)  # Changed model_path to model_info
    leaderboard_path = evaluate(
        model_info, test_data
    )  # Changed model_path to model_info
    visualize(
        model_info, test_data, predictions
    )  # Use test data for visualization # Changed model_path to model_info
    save_model(model_info)


# Invoke the DAG
chronos_train_dag_no_mlflow()