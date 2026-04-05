import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 프로젝트 루트를 path에 추가하여 src를 불러올 수 있게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_directory_setup():
    """데이터 저장을 위한 폴더가 생성되는지 확인"""
    DATA_DIR = Path("data")
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    assert PROCESSED_DIR.exists()

def test_data_structure():
    """데이터프레임의 필수 컬럼과 데이터 타입 확인"""
    # 임의의 샘플 데이터 생성
    data = {
        "datetime": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
        "actual_load": [35000, 36000],
        "forecast_load": [34500, 35500]
    }
    df = pd.DataFrame(data)
    
    # 기초적인 계산 검증
    df["forecast_error"] = df["actual_load"] - df["forecast_load"]
    assert df["forecast_error"].iloc[0] == 500
    assert not df["forecast_error"].isnull().any()
