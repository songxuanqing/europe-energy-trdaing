
# Europe Energy Trading Analysis

## 프로젝트 목적

이 프로젝트는 **독일/유럽 전력 시장에서의 거래 마진 극대화**를 목표로, 시계열 전력 수요와 공급 데이터를 통합·분석하고, **예측 오차 기반 트레이딩 시그널(alpha signal)**을 생성하기 위해 진행되었습니다.
주요 목표는 다음과 같습니다:

1. **전력 수요 및 공급 예측 정확도 향상**

   * 실제 부하와 예측 부하 간 오차(forecast error)를 모델링하여, 가격 급등/급락 가능성을 사전에 탐지.
2. **재생에너지 및 전통 발전 비중 파악**

   * 풍력, 태양광, 화력 등 발전원별 데이터를 통합하여 전력 포트폴리오 분석.
3. **트레이딩 전략 생성**

   * 예측 오차와 가격 스파이크 간 상관관계를 활용하여 alpha signal 생성 → 거래 방향 결정.

---

## 데이터 전개 및 처리 이유

데이터는 다음과 같이 전개되었습니다:

| 데이터                    | 처리 및 의미                              |
| ---------------------- | ------------------------------------ |
| Actual consumption     | 시계열 전력 수요. forecast error 계산의 기준.    |
| Forecasted consumption | 시장 예상 부하. 실제 수요와 비교해 오차 모델링.         |
| Actual generation      | 발전원별 시간대 발전량. 재생에너지 비율과 공급 포트폴리오 분석. |
| Cross-border flows     | 인접 국가 전력 수급 영향. 순수입/수출 고려.           |
| Day-ahead prices       | 거래 마진 계산을 위한 기준 가격. 가격 스파이크 분석에 활용.  |
| Weather                | 풍력/태양광 발전량 및 부하 변동성 설명 변수.           |

**전개 방식:**

* CSV 로드 후 datetime 통합, NaN 처리, 선형 보간(interpolation), lag & rolling feature 생성.
* Lag (1h, 24h, 48h, 168h)와 rolling 통계(24h, 168h) 활용 → 시계열 의존성 반영.
* 공휴일, 주말, 시간대 등 시계열 관련 파생 feature 생성.

**의미:**

* 거래 전략에서 예측 오차가 가격 스파이크와 연결되므로, **lag/rolling을 통해 과거 패턴을 반영하는 feature engineering**이 중요.

---

## 사고 과정

1. **데이터 이해**

   * 전력 수요/공급, 가격, 기상 데이터의 시계열 패턴 확인.
2. **정상성 검사**

   * ADF/KPSS 테스트 → 부하 시계열 정상성 확보.
   * 차분(Differencing) 및 이동 평균 적용.
3. **Feature 설계**

   * 시계열 의존성(lag), 변동성(rolling), 재생에너지 비율, 가격/오차 스파이크 등을 통합.
4. **모델 선택**

   * **LightGBM Regressor** 사용: 비선형 관계 및 시계열 의존성을 효율적으로 학습.
5. **트레이딩 시그널 설계**

   * 예측 오차 기반 alpha signal 생성:

     * 오차 스파이크 → 거래 포지션 long/short 결정.
     * 방향성 정확도와 큰 오차 탐지 정확도 측정.

---

## 내가 한 액션

1. **데이터 전처리**

   * NaN 처리, 문자열 → 수치 변환, 선형 보간, datetime 통합.
   * 발전원별 합계, 재생에너지 비율 계산.
2. **Feature Engineering**

   * Lag, rolling, 시계열 파생 feature 생성.
   * 시간, 주말/공휴일, 월/요일 등의 외생 변수 추가.
3. **모델 학습**

   * LGBMRegressor 학습 → forecast error 예측.
   * train/test split: 시계열 순서 기반.
4. **결과 평가**

   * MAE, RMSE, MAPE 계산.
   * Direction Accuracy, Big Error Detection Accuracy 측정.
5. **트레이딩 시그널**

   * forecast error spike → alpha signal 생성.
   * alpha_position = alpha_signal * alpha_direction.
6. **시각화**

   * 실제 vs 예측 부하, 재생에너지 비율, 이동평균, 오차/가격 스파이크.

---

## 처리 결과

* **모델 성능**

  * MAE: 343.87 MWh
  * RMSE: 470.60 MWh
  * MAPE: 156.99%
  * Direction Accuracy: 95.7%
  * Big Error Detection Accuracy: 97.9%
* **상위 Feature 중요도**

  * `actual_load`, `forecast_load`, `forecast_error_lag_1`, `load_lag_1`, `hour`, `solar`
* **Alpha Signal 분석**

  * 예측 오차 스파이크 발생 시 가격 변동이 큰 확률 높음 → 트레이딩 포지션 근거.

---

## 해석 및 의미 (전력 거래 관점)

* **예측 오차 기반 트레이딩**

  * 예측보다 수요가 높거나 낮으면 day-ahead 가격이 변동 → 마진 기회 발생.
  * Alpha signal을 통해 **long/short 포지션 자동화** 가능.
* **재생에너지 비율 활용**

  * 풍력/태양광 비중이 높을수록 공급 변동성 증가 → 가격 스파이크 위험 상승.
* **Lag/rolling feature 활용**

  * 과거 부하 패턴 및 오차가 미래 가격 변동 예측에 핵심.
* **마진 최적화**

  * 모델 기반 예측과 alpha signal 적용 → 불필요한 거래 위험 감소, 전략적 포지션 확대 가능.

---

## 결과 파일

* `data/processed/processed_data.csv`: 통합 전처리 데이터
* `data/processed/forecast_error_predictions.csv`: 예측 오차 및 모델 예측 결과
* `data/processed/forecast_error_price_analysis.csv`: 가격 스파이크 분석 및 alpha signal

