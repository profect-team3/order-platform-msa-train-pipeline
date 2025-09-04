# Order Platform MSA Train Pipeline

이 프로젝트는 마이크로서비스 아키텍처에서 머신러닝 모델 훈련을 위한 Airflow, MLflow, DVC를 사용하는 MLOps 파이프라인을 설정합니다.

## 프로젝트 구조

- `dags/`: 워크플로우 오케스트레이션을 위한 Airflow DAG
- `plugins/`: 사용자 정의 Airflow 플러그인
- `models/`: 훈련된 머신러닝 모델
- `data/`: DVC로 관리되는 데이터 파일
- `experiments/`: MLflow 실험 및 실행
- `scripts/`: 훈련 및 유틸리티 스크립트
- `config/`: 설정 파일
- `notebooks/`: 실험을 위한 Jupyter 노트북

## 설정

1. uv를 사용하여 의존성 설치:
   ```
   uv sync
   ```

2. Pre-commit hooks 설정 (코드 품질 자동 검사):
   ```
   uv run pre-commit install
   ```

3. DVC 초기화 (이미 완료됨):
   ```
   uv run dvc init
   ```

4. Airflow 시작:
   ```
   uv run airflow standalone
   ```

5. MLflow UI 시작:
   ```
   uv run mlflow ui
   ```

## 빠른 시작

### 옵션 1: Docker Compose로 실행 (전체 스택, 권장)

공식 Airflow 클러스터와 MLflow를 함께 실행:

```bash
./scripts/docker-compose-run.sh
```

또는 수동으로:

```bash
docker-compose up --build
```

- Airflow 웹 UI: http://localhost:8000
- MLflow UI: http://localhost:8001

**로그인 방법**:
- 사용자명: `admin`
- 비밀번호: 컨테이너 로그에서 확인
  ```bash
  docker logs order-platform-msa-train-pipeline-airflow-1 | grep -i password
  ```
  로그에서 "Password for user 'admin': [비밀번호]" 형식으로 표시됩니다.

**참고**: 이 설정은 PostgreSQL, Redis, CeleryExecutor를 포함한 완전한 Airflow 클러스터입니다.

**로그 확인**: Docker Desktop에서 각 서비스의 로그를 확인할 수 있습니다.

### 옵션 2: 단일 Docker로 실행 (Airflow만)

```bash
# 이미지 빌드
docker build -t order-platform-mlops .

# 컨테이너 실행
docker run -p 8000:8000 order-platform-mlops
```

http://localhost:8000에서 Airflow 웹 UI에 접속하세요.  
**참고**: 개발 편의를 위해 인증이 비활성화되어 있습니다 (모든 사용자가 관리자 권한). 로그인 필요 없음.

**관리자 계정**: 필요시 `admin` / `admin`으로 로그인 가능

### 옵션 3: 로컬에서 실행

1. 의존성 설치:
   ```bash
   uv sync
   ```

2. Airflow 시작:
   ```bash
   uv run airflow standalone
   ```

Airflow 웹 UI에 접속: http://localhost:8000  
**중요**: 인증이 완전히 비활성화되어 있습니다. 로그인 페이지가 나타나도 그냥 새로고침하거나 URL에 직접 접속하세요.

## 사용법

- `dags/` 디렉토리에 DAG 파일을 배치하세요.
- `scripts/`에서 훈련 스크립트를 실행하세요.
- MLflow로 실험을 추적하세요.
- DVC로 데이터 버전을 관리하세요.

## 기술

- **Airflow**: 워크플로우 오케스트레이션 (3.x 호환)
- **MLflow**: 실험 추적 및 모델 관리
- **DVC**: 데이터 버전 관리
- **uv**: 빠른 Python 패키지 관리자

## CI/CD

이 프로젝트는 지속적 통합을 위한 GitHub Actions와 개발을 위한 로컬 CI 스크립트를 포함합니다.

### GitHub Actions

CI 파이프라인은 `main`과 `dev` 브랜치의 모든 푸시와 풀 리퀘스트에서 실행됩니다. 다음을 포함합니다:
- uv를 사용한 의존성 설치
- ruff를 사용한 코드 린팅
- pytest를 사용한 단위 테스트
- 모델 훈련 실행
- Docker 이미지 빌드

### 로컬 개발

로컬에서 코드 품질 검사를 수행하려면:

```bash
# 린팅
uv run ruff check .

# 테스트 실행
uv run pytest

# 훈련 스크립트 실행
uv run python scripts/train.py
```

### Docker

Docker 이미지 빌드:

```bash
docker build -t order-platform-mlops .
```

컨테이너 실행:

```bash
docker run -p 8000:8000 order-platform-mlops
```

http://localhost:8000에서 Airflow 웹 UI에 접속하세요.

### Docker Compose

전체 MLOps 스택을 Docker Compose로 실행:

```bash
docker-compose up --build
```

또는 간단한 스크립트 사용:

```bash
./scripts/docker-compose-run.sh
```

서비스:
- **Airflow**: http://localhost:8000
- **MLflow**: http://localhost:8001

볼륨 마운트를 통해 로컬 파일 변경사항이 컨테이너에 반영됩니다.

## 문제 해결

**참고**: 401 에러가 발생하면 Airflow가 완전히 초기화될 때까지 몇 초 기다리거나 컨테이너 로그를 확인하세요:
```bash
docker logs <container_id>
```

인증이 여전히 실패하면 수동으로 사용자를 생성할 수 있습니다:
```bash
docker exec -it <container_id> uv run airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

**개발 참고**: 기본적으로 인증이 비활성화되어 개발이 쉽습니다. 인증이 필요하면 Dockerfile에서 `AIRFLOW__WEBSERVER__SIMPLE_AUTH_MANAGER_ALL_ADMINS` 환경 변수를 제거하세요.

## 데이터 버전 관리 (DVC)

이 프로젝트는 DVC(Data Version Control)를 사용하여 데이터 파일의 버전을 관리합니다. DVC는 Git처럼 데이터 변경 사항을 추적하고, 대용량 파일을 효율적으로 처리합니다.

### DVC 설치 및 초기화

1. DVC 설치 (uv 사용):
   ```bash
   uv add dvc
   ```

2. DVC 초기화 (프로젝트 루트에서):
   ```bash
   uv run dvc init
   ```
   - `.dvc/` 폴더와 `.dvcignore` 파일이 생성됩니다.
   - Git에 `.dvc` 폴더를 추가하세요: `git add .dvc`

### 데이터 파일 추적

1. 데이터 파일 추가:
   ```bash
   dvc add data/forecast_data_featured.csv
   ```
   - `data/forecast_data_featured.csv.dvc` 파일이 생성됩니다.
   - 원본 파일은 `.gitignore`에 추가되어 Git에서 제외됩니다.

2. Git 커밋:
   ```bash
   git add data/forecast_data_featured.csv.dvc .gitignore
   git commit -m "Add data file with DVC tracking"
   ```

### DVC 원격 저장소 설정 (선택)

대용량 데이터를 클라우드에 저장하려면 원격 저장소를 설정하세요:

1. Google Drive 예시:
   ```bash
   dvc remote add -d myremote gdrive://folder-id
   ```

2. AWS S3 예시:
   ```bash
   dvc remote add -d myremote s3://mybucket/path
   ```

3. 데이터 푸시:
   ```bash
   dvc push
   ```

4. 데이터 풀:
   ```bash
   dvc pull
   ```

### DVC 상태 및 변경 사항 확인

- 상태 확인:
  ```bash
  dvc status
  ```

- 변경 사항 비교:
  ```bash
  dvc diff
  ```

- 데이터 재현 (스크립트 실행):
  ```bash
  dvc repro
  ```

### 워크플로우 예시

1. 시뮬레이션 데이터 생성:
   ```bash
   uv run python notebooks/generate_simulated_data.py
   ```

2. DVC로 추적:
   ```bash
   dvc add data/forecast_data_featured.csv
   git add data/forecast_data_featured.csv.dvc
   git commit -m "Update simulated data"
   ```

3. 원격에 푸시:
   ```bash
   dvc push
   ```

DVC를 사용하면 데이터 변경 사항을 Git처럼 관리할 수 있어, 실험 재현성과 협업이 용이합니다.
