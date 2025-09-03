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

2. DVC 초기화 (이미 완료됨):
   ```
   uv run dvc init
   ```

3. Airflow 시작:
   ```
   uv run airflow standalone
   ```

4. MLflow UI 시작:
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

로컬에서 CI 파이프라인을 실행하려면:

```bash
./scripts/ci.sh
```

다음을 수행합니다:
- 의존성 설치/업데이트
- 린팅 실행
- 테스트 실행
- 훈련 스크립트 실행

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
