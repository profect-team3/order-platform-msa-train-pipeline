import pandas as pd
import json
from datetime import datetime

# Define the path to the CSV file
csv_file_path = "/Users/coldbrew_groom/Documents/order-platform-mlops/order-platform-msa-train-pipeline/data/example_infer_data.csv"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Take the first 7 days (7 * 24 hours = 168 rows)
# Assuming the CSV is sorted by timestamp and contains enough data
df_7days = df.head(7 * 24)

# Initialize the base request dictionary
fastapi_request_data = {
    "store_id": df_7days["store_id"].iloc[0],  # Get store_id from the first row
    "input_length": 24,
    "prediction_length": 6,
    "realDataItemList": []
}

# Populate realDataItemList
for index, row in df_7days.iterrows():
    # Convert timestamp to ISO 8601 format
    timestamp_obj = pd.to_datetime(row["timestamp"])
    iso_timestamp = timestamp_obj.isoformat(timespec='seconds')

    item = {
        "timestamp": iso_timestamp,
        "storeId": row["store_id"],
        "categoryMain": row["category_main"],
        "categorySub": row["category_sub"],
        "categoryItem": row["category_item"],
        "region": row["region"],
        "realOrderQuantity": row["real_order_quatity"],
        "realSalesRevenue": row["real_sales_revenue"],
        "dayOfWeek": row["day_of_week"],
        "hour": row["hour"],
        "minOrderAmount": row["min_order_amount"],
        "avgRating": row["avg_rating"]
    }
    fastapi_request_data["realDataItemList"].append(item)

# Print the JSON object
print(json.dumps(fastapi_request_data, indent=2, ensure_ascii=False))

# You can also save it to a file if needed
with open("fastapi_request.json", "w", encoding="utf-8") as f:
    json.dump(fastapi_request_data, f, indent=2, ensure_ascii=False)
