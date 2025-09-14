import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm  # 진행 상황 트래킹

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 분리 ---
CONFIG = {
    "start_date": datetime(2025, 2, 1),
    "num_days": 360,
    "num_stores": 100,
    "output_dir": "./data",

    "regions": ["강남구", "서초구", "송파구", "마포구", "영등포구", "용산구", "성동구", "광진구", "중구", "종로구"],  # 한글 지역 리스트
    "region_multipliers": {  # 지역별 주문량 multiplier (인구/소득 기반)
        "강남구": 1.5,  # 부촌, 높음
        "서초구": 1.3,
        "송파구": 1.2,
        "마포구": 1.1,
        "영등포구": 1.0,
        "용산구": 0.9,
        "성동구": 0.8,
        "광진구": 0.7,
        "중구": 0.6,  # 도심, 중간
        "종로구": 0.5  # 낮음
    },
    "categories": {  # 3단계 카테고리 정의
        "음식": {
            "한식": ["비빔밥", "김치찌개", "불고기", "삼겹살"],
            "양식": ["피자", "파스타", "햄버거", "스테이크"],
            "중식": ["짜장면", "짬뽕", "탕수육", "마파두부"],
            "일식": ["초밥", "라멘", "돈까스", "우동"]
        },
        "음료": {
            "커피": ["아메리카노", "카페라떼", "카푸치노", "에스프레소"],
            "차": ["녹차", "홍차", "허브티", "밀크티"],
            "주스": ["오렌지주스", "사과주스", "포도주스", "토마토주스"]
        },
        "디저트": {
            "케이크": ["초코케이크", "치즈케이크", "생크림케이크", "티라미수"],
            "쿠키": ["초코칩쿠키", "오트밀쿠키", "마카다미아쿠키", "레이즌쿠키"],
            "아이스크림": ["바닐라", "초코", "딸기", "민트초코"]
        }
    }
}

def generate_store_profiles(num_stores):
    """가게 프로필을 동적으로 생성"""
    profiles = {}
    category_keys = list(CONFIG["categories"].keys())
    for i in range(num_stores):
        store_id = f"store_{i+1:03d}"
        main_cat = np.random.choice(category_keys)
        sub_cat = np.random.choice(list(CONFIG["categories"][main_cat].keys()))
        item = np.random.choice(CONFIG["categories"][main_cat][sub_cat])
        region = np.random.choice(CONFIG["regions"])
        min_order_amount = 0
        if main_cat == "음료":
            min_order_amount = np.random.randint(1000, 3000)
        elif main_cat == "음식":
            min_order_amount = np.random.randint(500, 1500)
        else:  # 디저트
            min_order_amount = np.random.randint(800, 2000)
        profiles[store_id] = {
            "category_main": main_cat,
            "category_sub": sub_cat,
            "category_item": item,
            "region": region,
            "min_order_amount": min_order_amount,
            "base_orders": np.random.randint(10, 21),  # 10-20개로 조정
            "weekend_multiplier": np.random.uniform(1.2, 2.5),
            "peak_hours": {
                "lunch": (11, 14, np.random.uniform(2.5, 4.0)),
                "dinner": (18, 21, np.random.uniform(3.0, 5.0)),
                "late_night": (22, 2, np.random.uniform(0.5, 1.5))  # 야식 피크
            },
            "trend": np.random.uniform(-0.01, 0.01),  # 감소하는 가게도 가능
            "base_rating": round(np.random.uniform(4.0, 5.0), 1),
            "event": {  # 이벤트: 랜덤 기간에 주문량 증가
                "start_day": np.random.randint(0, CONFIG["num_days"] - 60),  # 시작일
                "duration": 60,  # 2개월 (60일)
                "multiplier": np.random.uniform(1.5, 3.0)  # 증가 배수
            } if CONFIG["num_days"] >= 60 and np.random.random() < 0.2 else None  # 20% 확률로 이벤트
        }
    return profiles

def generate_data(num_stores=CONFIG["num_stores"]):
    """시뮬레이션 데이터 생성"""
    logging.info("데이터 생성을 시작합니다.")
    store_profiles = generate_store_profiles(num_stores)
    all_orders = []
    current_time = CONFIG["start_date"]
    end_time = CONFIG["start_date"] + timedelta(days=CONFIG["num_days"])
    time_step = timedelta(hours=1)
    day_count = 0
    daily_ratings = {} # 일별 평점을 저장할 딕셔너리

    total_iterations = (end_time - current_time) // time_step * num_stores
    with tqdm(total=total_iterations, desc="데이터 생성 진행") as pbar:
        while current_time < end_time:
            # 날짜가 바뀌면 모든 가게의 일별 평점 재계산
            if current_time.hour == 0:
                day_count += 1
                for store_id, profile in store_profiles.items():
                    rating_noise = np.random.normal(0, 0.05)
                    daily_avg_rating = round(profile["base_rating"] + rating_noise + day_count * 0.001, 2)
                    daily_avg_rating = max(3.5, min(5.0, daily_avg_rating))
                    daily_ratings[store_id] = daily_avg_rating

            for store_id, profile in store_profiles.items():
                # 주문량 계산
                base_orders = profile["base_orders"]
                trend_multiplier = 1 + (day_count * profile["trend"]) + np.random.normal(0, 0.005)  # 트렌드 노이즈 추가
                weekday = current_time.weekday()
                # 평일별 다양화: 월(0) 낮음, 화수목 중간, 금(4) 높음, 주말 높음
                if weekday == 0:  # 월요일
                    weekday_multiplier = np.random.uniform(0.7, 0.9) * np.random.normal(1.0, 0.1)  # 노이즈 추가
                elif weekday == 4:  # 금요일
                    weekday_multiplier = np.random.uniform(1.3, 1.6) * np.random.normal(1.0, 0.1)
                elif weekday >= 5:  # 주말
                    weekday_multiplier = profile["weekend_multiplier"] * np.random.normal(1.0, 0.1)
                else:  # 화수목
                    weekday_multiplier = np.random.uniform(0.9, 1.1) * np.random.normal(1.0, 0.1)
                daily_multiplier = 1.0
                # 기본 시간대 multiplier (낮 시간 주문 다양화)
                if 6 <= current_time.hour <= 10:  # 아침
                    daily_multiplier = np.random.uniform(0.3, 0.7) * np.random.normal(1.0, 0.2)
                elif 14 <= current_time.hour <= 17:  # 오후
                    daily_multiplier = np.random.uniform(0.5, 1.0) * np.random.normal(1.0, 0.2)
                elif 3 <= current_time.hour <= 5:  # 새벽
                    daily_multiplier = np.random.uniform(0.1, 0.4) * np.random.normal(1.0, 0.2)
                for peak_name, (start, end, multiplier) in profile.get("peak_hours", {}).items():
                    if peak_name == "late_night" and (start <= current_time.hour or current_time.hour <= end):
                        daily_multiplier = multiplier * np.random.normal(1.0, 0.2)
                    elif start <= current_time.hour <= end:
                        daily_multiplier = multiplier * np.random.normal(1.0, 0.2)
                        break
                noise = np.random.normal(1.0, 0.3)  # 노이즈 증가
                region_multiplier = CONFIG["region_multipliers"].get(profile["region"], 1.0)  # 지역 multiplier
                event_multiplier = 1.0
                if profile.get("event"):
                    event = profile["event"]
                    if event["start_day"] <= day_count <= event["start_day"] + event["duration"]:
                        event_multiplier = event["multiplier"]
                order_count = int(base_orders * trend_multiplier * weekday_multiplier * daily_multiplier * region_multiplier * event_multiplier * noise)
                order_count = max(0, order_count)

                # 피처 생성
                avg_rating = daily_ratings[store_id] # 일별 고정된 평점 사용

                # if 11 <= current_time.hour <= 13:
                #     delivery_ratio, simple_pay_ratio = 0.6, 0.7
                # elif 18 <= current_time.hour <= 20:
                #     delivery_ratio, simple_pay_ratio = 0.85, 0.6
                # else:
                #     delivery_ratio, simple_pay_ratio = 0.7, 0.5

                # 물가 상승 적용 (연간 3%)
                year_diff = current_time.year - CONFIG["start_date"].year
                inflation_multiplier = (1.03) ** year_diff

                sales_amount = int(order_count * (profile["min_order_amount"] * np.random.normal(1.0, 0.1)) * inflation_multiplier * np.random.normal(1.2, 0.2) * trend_multiplier)

                all_orders.append({
                    "timestamp": current_time,
                    "store_id": store_id,
                    "category_main": profile["category_main"],
                    "category_sub": profile["category_sub"],
                    "category_item": profile["category_item"],
                    "region": profile["region"],
                    "real_order_quantity": order_count,
                    "real_sales_revenue": sales_amount,
                    "day_of_week": weekday,
                    "hour": current_time.hour,
                    "min_order_amount": profile["min_order_amount"],
                    "avg_rating": avg_rating,
                    # "receipt_delivery_ratio": delivery_ratio * np.random.normal(1.0, 0.05),
                    # "receipt_take_out_ratio": (1 - delivery_ratio) * np.random.normal(1.0, 0.05),
                    # "payment_simple_pay_ratio": simple_pay_ratio * np.random.normal(1.0, 0.05),
                    # "payment_credit_card_ratio": (1 - simple_pay_ratio) * np.random.normal(1.0, 0.05),
                })
                pbar.update(1)

            current_time += time_step
            # day_count는 current_time.hour == 0 조건문 안으로 이동
    df = pd.DataFrame(all_orders)
    logging.info("데이터 생성 완료.")
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulated order data.")
    parser.add_argument("--num_stores", type=int, default=CONFIG["num_stores"], help="Number of stores to simulate.")
    parser.add_argument("--num_days", type=int, default=CONFIG["num_days"], help="Number of days to simulate.")
    parser.add_argument("--start_date", type=str, default=CONFIG["start_date"].strftime("%Y-%m-%d"), help="Start date for simulation (YYYY-MM-DD).")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"], help="Output directory for the generated CSV.")

    args = parser.parse_args()

    np.random.seed(42)  # 재현성

    # Update CONFIG with parsed arguments
    CONFIG["num_stores"] = args.num_stores
    CONFIG["num_days"] = args.num_days
    CONFIG["start_date"] = datetime.strptime(args.start_date, "%Y-%m-%d")
    CONFIG["output_dir"] = args.output_dir

    output_filename = f"train_D{args.num_days}_S{args.num_stores}.csv"
    
    forecast_df = generate_data(args.num_stores)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
    forecast_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"데이터 저장 완료: {output_path}")
    print("생성된 데이터 컬럼:", forecast_df.columns.tolist())
    print("샘플 데이터:\n", forecast_df.head())
