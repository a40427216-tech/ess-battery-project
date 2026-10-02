# 데이터 출처와 처리 기준

과제 지정 [Kaggle MIT–Stanford 배터리 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 다음 세 원본을 사용했다.

| 배치 | 파일 날짜 | 원본 셀 | 분석 셀 |
| --- | --- | --- | --- |
| Batch 1 | 2017-05-12 | 46 | 36 |
| Batch 2 | 2018-02-20 | 47 | 39 |
| Batch 3 | 2018-04-12 | 46 | 40 |

원본 파일명·다운로드 URL·바이트 수·SHA-256은 `source_manifest.json`, 포함된 가공 자료의 해시는 `processed_manifest.json`에 기록했다. 대용량 원본은 저장소에서 제외하고 가공 자료를 포함했다.

## 포함 자료

- `processed/cells_raw.csv`: 원본 셀 ID·정책·제공 수명·마지막 관측 메타데이터.
- `processed/cells_analysis.csv`: 제외 기준을 적용한 분석 코호트.
- `processed/summary_raw.csv.gz`, `summary_analysis.csv.gz`: 전체 사이클 요약. 전체 곡선과 knee의 EDA에 사용한다.
- `processed/early_curves.npz`: 실제 2·10·100사이클 원시 신호와 원본 Vdlin 격자.
- `processed/extraction_audit.json`: 원시 최대 Qd와 요약 QD의 사이클 정렬 대조.

모델 피처는 `src.preprocess.load_inputs()`가 먼저 2–100사이클로 자른 요약 자료에서 계산한다. 전체 관측 길이·마지막 QD·레이블·셀/배치 ID는 모델의 허용 입력 목록에서 제외한다.

## 제외 기준과 한계

공칭 1.1Ah의 80%는 0.88Ah다. EOL 미관측 기록을 판별할 때만 마지막 QD 0.885Ah를 허용 기준으로 사용하고, 정답은 제공된 `cycle_life`를 유지한다.

Batch 1의 10개 미완료 셀을 제외했다. 이 중 첫 5개는 원논문에서 2017-06-30 후속 기록을 연결한 셀이지만 과제 파일에 해당 연장 기록이 없으므로 논문 수명값으로 덮어쓰지 않았다. 이 처리 때문에 장수명 학습 정보가 제한된다.

Batch 2의 수명 결측 8개와 Batch 3의 공개 저자 로더상 노이즈 6개를 제외했다. 수명 결측과 노이즈 사유는 일부 겹친다. 정확한 셀과 이유는 [셀별 제외 기록](../results/cell_exclusions.csv) 및 [품질 점검 결과](../results/quality_audit.json)에 있다. 분석 결과는 이 코호트에 조건부이며 모든 원본 셀이나 새 현장을 대표하지 않는다.

Batch 2는 지정 2018-02-20이다. 논문의 2017-06-30 파일을 대체 평가 배치로 사용하지 않았다. Vdlin의 실제 격자는 3.5→2.0V, 1,000점이다. 중심화는 ΔQ의 상수 오프셋만 제거하며 전체 정렬 문제를 해결하는 보정은 아니다.

## 원본부터 재현

프로젝트 루트에서 실행한다. 다운로드 없이 가공 자료로 재현하려면 README의 `python -m src.train`부터 실행하면 된다.

```bash
python scripts/download_data.py
python scripts/extract_data.py
python scripts/prepare_analysis.py
python scripts/analyze_data.py
python scripts/record_data_manifest.py
python -m src.train
python -m src.evaluate
```

가공 입력의 해시가 바뀌면 피처 생성 단계에서 중단한다. 원본 재처리를 의도적으로 수행한 경우에만 `record_data_manifest.py`로 새 가공 자료의 출처 기록을 갱신한다.

데이터의 논문 출처: Severson et al. (2019), *Nature Energy*, 4, 383–391. 파일을 다른 연구에 활용할 때 원본 배포 조건과 출처를 확인한다. 글꼴의 배포 조건은 `references/NanumGothic-OFL.txt`에 포함했다.
