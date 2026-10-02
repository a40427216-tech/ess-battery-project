# ESS 배터리 수명 예측

## 목적

초기 100사이클 데이터로 배터리 총수명을 예측하고, 다른 실험 배치에서의 성능과 ESS 운영 시사점을 분석한다.

## 프로젝트 개요

- 데이터: MIT–Stanford Battery Dataset (Severson et al., 2019).
- 학습: Batch 1 (`2017-05-12`) 36개 중 개발 28개, hold-out 8개.
- 평가: Batch 2 (`2018-02-20`) 39개; 추가 Batch 3 (`2018-04-12`) 40개.
- 태스크: Regression — 제공된 `cycle_life`(총수명, 사이클) 예측.

## 파일 구조

```text
.
├── data/          출처·가공 데이터·README.md
├── notebooks/     01_EDA.ipynb / 02_feature_engineering.ipynb / 03_modeling.ipynb
├── src/           preprocess.py / features.py / train.py / evaluate.py
├── scripts/       데이터 처리·결과 검증·문서 생성
├── models/        final_model.joblib
├── results/       model_performance.csv·셀별 예측·그래프
├── docs/          분석보고서(MD·PDF)·모델 탐색 근거
├── tests/         입력·검증 분리·미래 정보 누출 테스트
├── requirements.txt
└── README.md
```

## 환경 설정

Python 3.11 기준이다. 프로젝트 폴더에서 실행하며, 포함된 가공 데이터로 학습·평가를 재현할 수 있다.

```bash
git clone https://github.com/a40427216-tech/ess-battery-project.git
cd ess-battery-project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

원본 처리 방법은 [data/README.md](data/README.md)에 정리했다.

## EDA

### Cycle Life 분포

Batch 1·2·3의 수명 중앙값은 772.5·472.0·964.5사이클이다. Batch 2에는 500사이클 미만 셀이 28개(71.8%)로 많아 학습 배치와 분포가 다르다. 원본 139개 중 EOL 미관측·수명 결측·기존 노이즈 기준으로 115개를 분석했다.

### 열화 곡선 분석

초기 용량이 증가하는 셀도 있지만 말기에는 가속 열화가 관측된다. 전체 곡선의 구간별 선형 근사로 knee를 탐색했으며, 미래 knee와 전체 관측 길이는 예측 입력에서 제외했다.

### ΔQ(V) 곡선 분석

Cycle 100−10의 ΔQ(V)를 계산했다. 로그 분산과 로그 수명의 Pearson 상관은 Batch 1·2·3에서 −0.844·−0.918·−0.805다. 장수명 셀의 ΔQ 변화가 상대적으로 작지만, 높은 상관만으로 외부 예측 성능을 보장할 수는 없다.

### 충전 속도(C-rate)와 수명의 관계

Batch 1의 8C(15%)–3.6C와 8C(35%)–3.6C 정책의 평균 수명은 각각 1,008·608사이클이다(각 n=2). 최대 C-rate가 같아도 SOC 구간과 전류 패턴에 따라 수명이 다르며, 소표본 비교로 인과관계를 단정하기 어렵다.

### 추가 확인한 내용

Batch 1의 Tavg–Tmax 상관은 0.953, ΔQ 로그 분산–중심화 최솟값 상관은 −0.944다. 중복 정보가 커서 규제 모델을 비교했다.

## Modeling

### 피처 엔지니어링 전략

최종 입력은 ΔQ 로그 분산, 중심화 ΔQ 최솟값, 2사이클 QD, 2–100사이클 QD 기울기의 4개다. 저항·온도·충전 시간·정책·실측 전류를 추가한 피처 집합도 비교했다. 결측 대체와 표준화는 각 학습 fold 안에서 적합하고, 타깃은 log10 변환 후 학습한다.

### 모델 선택 및 근거

- 후보: 중앙값 기준, ΔQ 단일 선형 기준, Ridge, ElasticNet, RBF SVR, 얕은 Random Forest의 17개 조합.
- 검증: 개발 28개에 정책 그룹 기준 바깥 5-fold·안쪽 4-fold CV; hold-out 8개는 별도 평가.
- 최종: **ElasticNet / discharge** (`alpha=0.01`, `l1_ratio=0.5`). 개발 CV 평균 MAPE가 가장 낮았고, QD 기울기 계수는 0으로 축소됐다.

모델 선택에는 개발 데이터만 사용했다. 동일한 개발 28개 적합 모델을 모든 평가 집단에 적용했다. 후보 선택에 사용한 CV는 낙관적일 수 있고, DAY 1 EDA에서 외부 배치도 살펴보았다.

## 성능 결과

| 구분 | MAPE(%) / Gap(%p) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 7.72 | 정책 그룹 5-fold 평균 |
| Valid (Batch 1 Hold-out) | 7.21 | 별도 8개 셀 |
| Test (Batch 2) | 43.45 | 필수 평가, 39개 셀 |
| Gap (Train–Valid) | -0.51 | Valid − CV |
| Gap (Valid–Test) | +36.25 | Batch 2 − Valid |
| Gap (Target–Test) | +34.35 | Batch 2 − 9.1 |
| Test (Batch 3) | 11.71 | 추가 평가, 40개 셀 |
| Gap (Batch 2–Batch 3) | -31.74 | Batch 3 − Batch 2 |
| Gap (Target–Test, Batch 3) | +2.61 | Batch 3 − 9.1 |

CV는 fold별 MAPE 평균(표준편차 2.56%)이다. Gap은 뒤 평가−앞 평가이며, 목표 비교는 Test−9.1%다. 논문과 평가 조건이 달라 목표 Gap은 참고 비교다. Batch 2에서 목표에 도달하지 못했다.

## 오류 분석

- Batch 2: 39개 중 37개를 과대 예측했다(평균 편향 +211.1사이클). 초기 QD 분포와 수명 상관의 배치 차이, 좁은 학습 수명 범위가 원인 후보다. ΔQ 단일 선형 기준의 MAPE 29.90%가 최종 모델보다 낮았으며, 외부 결과를 보고 모델을 교체하지 않았다.
- Batch 3: 1,000사이클 초과 19개 셀을 모두 과소 예측했다. b3c38의 실제 수명은 1,935사이클, 예측은 약 1,052사이클이다. 장수명 학습 표본을 확보하고 새 검증 집단에서 개선 효과를 확인할 필요가 있다.

## ESS 도메인 해석

수명 예측은 BMS 상태 정보와 함께 EMS의 점검·교체 우선순위 판단을 보완할 수 있다. Batch 2의 과대 예측은 교체 시점을 늦출 수 있으므로 평균 오차와 오차 방향을 함께 봐야 한다. 전류·SOC 정책은 PCS 충전 제어와 I²R 손실에 연결되지만, 이번 분석만으로 최적 정책을 결정하기는 어렵다.

실제 ESS 적용에는 부분 충·방전, 달력 열화, 센서 오차, 셀 간 불균형 및 팩 운전 조건의 추가 검증이 필요하다. 총수명 예측은 현재 SOH·출력 한계 SOP와 구분하고, 전압·온도 보호 제어를 대신하지 않는다.

### 후기

SKALA 수업에서 처음으로 전기전자공학 전공과 연결해 과제를 진행해 특히 흥미로웠다. 전압·전류·용량·저항·온도가 실제 수명 분석에 활용되는 과정을 살펴보며, 전공 지식과 데이터 분석을 함께 사용하는 경험을 했다. 앞으로도 전공과 관련된 데이터를 분석할 기회가 더 많아졌으면 좋겠다.

## 참고문헌

- [Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy.](https://www.nature.com/articles/s41560-019-0356-8)
- [MIT–Stanford Battery Dataset — Kaggle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

## 팀 구성

김진형(울산캠퍼스 1반, 개인 과제): EDA, 피처 엔지니어링, 모델 개발, 성능 평가, 오류 분석 및 보고서 작성.

[상세 분석보고서](docs/DAY2_분석보고서.md) · [피처 정의와 전체 모델 비교](docs/모델_탐색_및_피처_정의.md)
