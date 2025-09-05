# Order Platform MSA Train Pipeline 프로젝트 개요

이 프로젝트는 마이크로서비스 아키텍처(MSA)에서 머신러닝 모델 훈련을 위한 MLOps 파이프라인을 구축합니다. Airflow를 사용하여 워크플로우를 오케스트레이션하고, MLflow로 실험을 추적하며, DVC로 데이터 버전을 관리합니다.

## 주요 기능
- **워크플로우 오케스트레이션**: Airflow를 통해 모델 훈련 파이프라인을 자동화합니다.
- **실험 추적**: MLflow로 모델 훈련 결과를 기록하고 비교합니다.
- **데이터 관리**: DVC를 사용하여 데이터 파일의 버전을 Git처럼 관리합니다.
- **시뮬레이션 데이터 생성**: [notebooks/generate_simulated_data.py](notebooks/generate_simulated_data.py)에서 주문 데이터를 시뮬레이션하여 생성합니다.

## 프로젝트 구조
- `dags/`: Airflow DAG 파일 (예: [example_dag.py](dags/example_dag.py)).
- `data/`: DVC로 관리되는 데이터 파일 (예: forecast_data_featured.csv).
- `notebooks/`: Jupyter 노트북 (예: [generate_simulated_data.py](notebooks/generate_simulated_data.py)).
- `scripts/`: 훈련 및 유틸리티 스크립트.
- `config/`: 설정 파일 (예: airflow.cfg).
- `tests/`: 단위 테스트.
- `README.md`: 자세한 설치 및 사용법.

## 시작하기
1. 의존성 설치: `uv sync`
2. Airflow 시작: `uv run airflow standalone`
3. 데이터 생성: `uv run python notebooks/generate_simulated_data.py`
4. DVC 추적: `dvc add data/forecast_data_featured.csv`

## 실행 옵션
- **Docker Compose (권장)**: `docker-compose up --build` (Airflow: http://localhost:8000, MLflow: http://localhost:8001)
- **단일 Docker**: `docker build -t order-platform-mlops .` 후 `docker run -p 8000:8000 order-platform-mlops`
- **로컬**: `uv run airflow standalone`

## 기술 스택
- Python 3.13+
- Apache Airflow 3.x
- MLflow 3.x
- DVC 3.x
- uv (패키지 관리)
- Docker

이 파일을 루트 디렉토리에 저장하고, [README.md](README.md)를 참조하여