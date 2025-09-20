# Order Forecast - Training Pipeline

## 1. 프로젝트 개요

이 프로젝트는 주문량 예측 모델의 학습을 자동화하는 MLOps 파이프라인입니다. Google Cloud Composer(Apache Airflow)를 사용하여 데이터 전처리, 모델 학습, 그리고 학습된 모델 아티팩트를 Google Cloud Storage(GCS)에 저장하는 전 과정을 오케스트레이션합니다.

## 2. 주요 기능 및 책임

- **데이터 전처리**: Kafka로부터 수집되어 GCS에 저장된 Raw 데이터를 주기적으로 읽어와, 모델 학습에 적합한 시계열 데이터셋으로 가공합니다.
- **모델 학습**: AutoGluon의 시계열 예측(TimeSeriesPredictor) 기능과 Chronos 모델을 사용하여 주문량 예측 모델을 학습합니다.
- **아티팩트 저장**: 학습이 완료된 모델을 추론 파이프라인에서 사용할 수 있도록 GCS 버킷에 저장합니다.

## 3. 아키텍처

- **오케스트레이션**: Google Cloud Composer (Apache Airflow)
- **핵심 로직**: Airflow DAG (`dags/train_no_mlflow_9.16_final.py`)
- **데이터 소스**: Google Cloud Storage (GCS)
- **학습 프레임워크**: AutoGluon Time-Series (Chronos 모델)
- **아티팩트 저장소**: Google Cloud Storage (GCS)

## 4. 설정 및 설치

프로젝트의 의존성은 `pyproject.toml`에 정의되어 있으며, `uv`를 사용하여 설치할 수 있습니다.

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt
```

## 5. 사용법

### 5.1. 자동화된 DAG 실행

- 메인 파이프라인은 `dags/train_no_mlflow_9.16_final.py`에 정의된 Airflow DAG입니다.
- 이 DAG는 Google Cloud Composer 환경에 배포되어, 매주(`@weekly`) 자동으로 실행되도록 설정되어 있습니다.

### 5.2. 로컬 테스트

로컬 환경에서 학습 과정을 테스트하려면 `notebooks/local_train.py` 스크립트를 사용할 수 있습니다. 실행 전 필요한 환경 변수를 `.env` 파일에 설정해야 합니다.

```bash
# .env 파일에 환경 변수 설정 후 실행
python notebooks/local_train.py
```

## 6. 주요 환경 변수

Airflow DAG가 GCS 리소스에 접근하고 모델을 올바르게 저장하기 위해 다음 환경 변수들이 필요합니다.

- `BUCKET_NAME`: 학습 데이터가 저장된 GCS 버킷 이름
- `OBJECT_NAME`: 학습 데이터 파일의 GCS 내 경로
- `MODEL_BUCKET_NAME`: 학습된 모델 아티팩트를 저장할 GCS 버킷 이름
- `MODEL_OBJECT_NAME_PREFIX`: GCS 버킷 내에서 모델 아티팩트를 저장할 경로 (prefix)