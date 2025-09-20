import os
import sys
import logging
import pandas as pd
import gcsfs

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Setup logging for the DAG file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _preprocess_data():
    """
    Function to run the preprocessing logic.
    """
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


with DAG(
    dag_id='preprocess_data_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['preprocessing', 'data'],
) as dag:
    preprocess_task = PythonOperator(
        task_id='run_preprocessing',
        python_callable=_preprocess_data,
    )