import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- 시뮬레이션 설정 ---
START_DATE = datetime(2024, 1, 1)
NUM_DAYS = 365  # 1년치 데이터

# --- 가게 프로필 (가게별 특성 정의) ---
STORE_PROFILES = {
    "store_001": {"category": "chicken", "region": "Gangnam", "min_order_amount": 15000, "base_orders": 10, "weekend_multiplier": 2.0, "peak_hours": {"dinner": (18, 21, 2.5)}, "trend": 0.005, "base_rating": 4.8},
    "store_002": {"category": "pizza", "region": "Gangnam", "min_order_amount": 18000, "base_orders": 12, "weekend_multiplier": 2.2, "peak_hours": {"dinner": (17, 22, 2.8)}, "trend": 0.003, "base_rating": 4.7},
    "store_003": {"category": "cafe", "region": "Pangyo", "min_order_amount": 11000, "base_orders": 15, "weekend_multiplier": 1.5, "peak_hours": {"morning": (8, 11, 2.0), "lunch": (12, 14, 2.5)}, "trend": 0.002, "base_rating": 4.9},
    "store_004": {"category": "korean", "region": "Mapo", "min_order_amount": 12000, "base_orders": 20, "weekend_multiplier": 1.4, "peak_hours": {"lunch": (11, 14, 3.0)}, "trend": -0.001, "base_rating": 4.6},
    "store_005": {"category": "chicken", "region": "Mapo", "min_order_amount": 14000, "base_orders": 8, "weekend_multiplier": 1.9, "peak_hours": {"dinner": (18, 22, 2.2)}, "trend": 0.01, "base_rating": 4.5}
}
AVG_PRICE_PER_ORDER = {"chicken": 22000, "pizza": 28000, "cafe": 12900, "korean": 18000}

def generate_data():
    all_orders = []
    current_time = START_DATE
    end_time = START_DATE + timedelta(days=NUM_DAYS)
    time_step = timedelta(hours=1)
    day_count = 0

    while current_time < end_time:
        for store_id, profile in STORE_PROFILES.items():
            # --- 1. 기본 주문량 및 패턴 계산 ---
            base_orders = profile["base_orders"]
            trend_multiplier = 1 + (day_count * profile["trend"])
            weekday = current_time.weekday()
            weekend_multiplier = profile["weekend_multiplier"] if weekday >= 4 else 1.0
            daily_multiplier = 1.0
            for start, end, multiplier in [v for k, v in profile.get("peak_hours", {}).items()]:
                if start <= current_time.hour <= end:
                    daily_multiplier = multiplier
                    break
            noise = np.random.normal(1.0, 0.2)
            order_count = int(base_orders/24 * trend_multiplier * weekend_multiplier * daily_multiplier * noise)
            order_count = max(0, order_count)

            # --- 2. 신규 피처 생성 ---
            # 2.1. 동적 평균 평점
            rating_noise = np.random.normal(0, 0.05)
            avg_rating = round(profile["base_rating"] + rating_noise + day_count * 0.001, 2)
            avg_rating = max(3.5, min(5.0, avg_rating))

            # 2.2. 결제/수령 방식 비율 시뮬레이션
            # (시간대에 따라 간단한 룰 기반으로 시뮬레이션)
            if 11 <= current_time.hour <= 13: # 점심시간
                delivery_ratio = 0.6
                simple_pay_ratio = 0.7
            elif 18 <= current_time.hour <= 20: # 저녁시간
                delivery_ratio = 0.85
                simple_pay_ratio = 0.6
            else: # 그 외 시간
                delivery_ratio = 0.7
                simple_pay_ratio = 0.5
            
            # 2.3. 매출액 계산
            sales_amount = int(order_count * profile["min_order_amount"] * np.random.normal(1.2, 0.1))

            # --- 3. 데이터 저장 ---
            all_orders.append({
                "timestamp": current_time,
                "store_id": store_id,
                "category": profile["category"],
                "region": profile["region"],
                "order_count": order_count,
                "sales_amount": sales_amount,
                "day_of_week": weekday,
                "hour": current_time.hour,
                # 신규 피처 추가
                "min_order_amount": profile["min_order_amount"],
                "avg_rating": avg_rating,
                "receipt_delivery_ratio": delivery_ratio * np.random.normal(1.0, 0.05),
                "receipt_take_out_ratio": (1-delivery_ratio) * np.random.normal(1.0, 0.05),
                "payment_simple_pay_ratio": simple_pay_ratio * np.random.normal(1.0, 0.05),
                "payment_credit_card_ratio": (1-simple_pay_ratio) * np.random.normal(1.0, 0.05),
            })
            
        current_time += time_step
        if current_time.hour == 0:
            day_count += 1

    df = pd.DataFrame(all_orders)

    # --- 4. 원-핫 인코딩 ---
    df = pd.get_dummies(df, columns=['region'], prefix='region')

    return df

if __name__ == "__main__":
    print("피처가 추가된 시계열 데이터 생성을 시작합니다...")
    forecast_df = generate_data()
    output_filename = "forecast_data_featured.csv"
    forecast_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"데이터 생성 완료! '{output_filename}' 파일에 저장되었습니다.")
    print("생성된 데이터 컬럼:")
    print(forecast_df.columns.tolist())
    print("\n생성된 데이터 샘플:")
    print(forecast_df.head())