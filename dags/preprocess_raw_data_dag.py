import os
import sys
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the preprocessing.
    """
    raw_data_path = '../order-platform-msa-infer-pipeline/data/consumed_orders.csv'
    output_dir = './data'
    output_path = os.path.join(output_dir, 'train_data.csv')

    logging.info(f"Reading raw data from {raw_data_path}")
    try:
        # The raw CSV may not have a header, so we name the columns
        df = pd.read_csv(raw_data_path)
    except Exception as e:
        logging.error(f"Failed to read raw data file: {e}")
        raise

    logging.info("Preprocessing data...")

    # --- Preprocessing Logic --- #
    train_target_df = pd.read_csv('./data/example_train_data.csv')
    print("target-df")
    print(train_target_df.columns)
    print(df.columns)
    df.rename(columns={'totalPrice': 'real_sales_revenue', 'storeId': 'store_id'}, inplace=True)
    print(df.columns)
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
    hourly_df = hourly_df[target_columns]

    os.makedirs(output_dir, exist_ok=True)
    hourly_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logging.info(f"Successfully preprocessed data and saved to {output_path}")

if __name__ == "__main__":
    main()
