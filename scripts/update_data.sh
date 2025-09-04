#!/bin/bash

# 데이터 업데이트 스크립트
# 이 스크립트는 시뮬레이션 데이터를 생성하고 DVC로 버전을 관리합니다.

set -e  # 에러 발생 시 스크립트 중단

echo "시뮬레이션 데이터 생성 중..."
uv run python notebooks/generate_simulated_data.py

echo "DVC로 데이터 추적 중..."
dvc add data/forecast_data_featured.csv

echo "Git에 변경 사항 추가 중..."
git add data/forecast_data_featured.csv.dvc

echo "Git 커밋 중..."
git commit -m "Auto-update simulated data $(date +%Y-%m-%d_%H-%M-%S)"

echo "DVC 원격 저장소에 푸시 중..."
dvc push

echo "데이터 업데이트 완료!"
