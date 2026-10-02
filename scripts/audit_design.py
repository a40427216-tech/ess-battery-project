"""Audit split and tuning details without changing model selection."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.preprocess import load_features,validate_input_manifest
from src.train import candidate_specs
from scripts.build_documents import table


def audit():
    features,split=load_features(ROOT)
    validate_input_manifest(ROOT)
    selection=json.loads((ROOT/'results/selection.json').read_text())
    registry=candidate_specs()
    expected=[{'name':x['name'],'feature_set':x['feature_set'],'grid':x['grid']} for x in registry]
    assert expected==selection['candidate_grids'],'Candidate search space changed after selection'
    by_id=features.set_index('cell_id',drop=False)
    contexts=[(f'outer_fold_{i}',f['train_cells']) for i,f in enumerate(split['cv'],1)]
    contexts.append(('final_development_tuning',split['development_cells']))
    records=[]
    for context,ids in contexts:
        frame=by_id.loc[ids]
        cv=GroupKFold(n_splits=min(4,frame.charging_policy.nunique()))
        for i,(a,b) in enumerate(cv.split(frame,frame.cycle_life,frame.charging_policy),1):
            train,valid=frame.iloc[a],frame.iloc[b]
            assert set(train.cell_id).isdisjoint(valid.cell_id)
            assert set(train.charging_policy).isdisjoint(valid.charging_policy)
            assert set(train.cell_id)|set(valid.cell_id)==set(ids)
            assert (train.batch==1).all() and (valid.batch==1).all()
            assert (set(train.cell_id)|set(valid.cell_id)).isdisjoint(split['holdout_cells'])
            records.append({'context':context,'fold':i,'train_cells':train.cell_id.tolist(),
                            'valid_cells':valid.cell_id.tolist(),
                            'n_train_groups':train.charging_policy.nunique(),
                            'n_valid_groups':valid.charging_policy.nunique()})
    (ROOT/'results/inner_cv_audit.json').write_text(json.dumps({
        'status':'passed','method':'Reconstructed from the exact GroupKFold splitter and frozen input order used by src.train.tune',
        'inner_folds':records,'holdout_or_external_cells_in_inner_cv':False,
        'candidate_grids_match_frozen_selection':True},ensure_ascii=False,indent=2))
    grids=[]
    for family in dict.fromkeys(x['family'] for x in registry):
        spec=next(x for x in registry if x['family']==family)
        tuning=' / '.join(f"{k.split('__')[-1]}: {v}" for k,v in spec['grid'].items()) or '튜닝 없음'
        fixed={'Dummy':'원래 총수명 중앙값','Linear':'log10 타깃, 기본 선형회귀',
               'Ridge':'log10 타깃, L2 규제','ElasticNet':'max_iter=100000, tol=1e-7, seed=42',
               'SVR':'RBF kernel, log10 타깃','RandomForest':'120 trees, seed=42, n_jobs=1'}[family]
        grids.append({'계열':family,'개발 데이터 내부 탐색 범위':tuning,'고정 설정':fixed})
    dictionary=pd.DataFrame([
        ['delta_q_logvar','log10(Ah² / 1Ah²)','log10(max(Var(Q100−Q10, ddof=1), 1e-15))','10·100'],
        ['delta_q_min','Ah','min(ΔQ−median(ΔQ))','10·100'],
        ['QD_2','Ah','summary.QDischarge의 실제 2번 사이클 값','2'],
        ['QD_slope','Ah/cycle','유효 QD에 대한 실제 사이클 번호의 최소제곱 기울기','2–100'],
        ['IR_mean','Ω','IR>0인 초기 내부 저항 평균','2–100'],
        ['IR_change_100_2','Ω','IR100−IR2; 비양수 IR은 결측','2·100'],
        ['Tavg_mean','°C','−20≤Tavg≤100인 초기 평균 온도의 평균','2–100'],
        ['chargetime_mean','min','양수 충전 시간 평균','2–100'],
        ['c_rate_1','C','충전 정책 문자열에서 첫 C-rate 추출','사전 정책'],
        ['c_rate_2','C','충전 정책 문자열에서 둘째 C-rate 추출','사전 정책'],
        ['switch_soc','%','첫 단계가 끝나는 SOC 경계','사전 정책'],
        ['charge_I_rms_c10','A','양의 충전 구간의 시간 가중 RMS; 시간은 분→초 환산','10'],
        ['heat_proxy_c10_J','J','충전 구간 ∫I²dt × 초기 평균 IR; 실제 총발열과 구분','10'],
    ],columns=['피처','단위','계산·처리','관측 범위'])
    comparison=pd.read_csv(ROOT/'results/cv_comparison.csv')
    full=table(comparison[['model','n_features','CV MAPE mean (%)','CV MAPE std (%)']].rename(columns={
        'model':'모델','n_features':'피처 수','CV MAPE mean (%)':'평균 MAPE(%)','CV MAPE std (%)':'표준편차(%)'}))
    ridge=comparison.query('family == "Ridge"')[['feature_set','n_features','CV MAPE mean (%)']]
    (ROOT/'docs/모델_탐색_및_피처_정의.md').write_text(f'''# 피처 정의와 모델 탐색의 상세 근거

## 입력 피처의 단위와 계산

{table(dictionary)}

QD 기울기 계산에서 0<QD<1.5Ah만 유효하다. 결측 대체값과 표준화 통계는 각 학습 fold에서 적합한다. 총수명·배치 ID·셀 ID·관측 길이·마지막 QD·미래 knee는 입력 목록에 없다. 정답은 제공된 cycle_life를 그대로 사용하며, 입력 셀은 분석 코호트의 조건을 충족한다.

ΔQ의 상수 중심화는 분산에 영향을 주지 않는다. 모델에 전달하는 전압 정렬은 제공된 실제 Vdlin 격자 기준이다. 초기 IR 결측은 B1 0개/B2 6개/B3 0개이며 최종 방전 피처 모델은 IR을 사용하지 않는다. IR을 쓰는 후보는 해당 학습 fold의 중앙값으로 대체한다.

## 하이퍼파라미터 탐색 범위

{table(pd.DataFrame(grids))}

모든 후보의 안쪽 평가 지표는 원래 사이클 단위의 MAPE다. 바깥 5-fold에서 모델·피처 조합을 비교하고, 각 안쪽 4-fold에서 위 격자를 탐색한다. 최종 28개 개발 데이터 튜닝도 같은 그룹 분리와 같은 격자를 사용한다.

## 전체 17개 후보 비교

{full}

## 같은 모델 계열에서의 피처 추가 비교

{table(ridge.rename(columns={'feature_set':'피처 집합','n_features':'입력 수','CV MAPE mean (%)':'Ridge CV MAPE(%)'}))}

Ridge 계열에서 방전 피처는 단일 분산보다 평균 CV 오차가 낮았지만, state·policy 피처 추가는 개선되지 않았다. current는 discharge와 비슷했다. 모델별 정규화 강도는 각각 안쪽 CV에서 선택했으므로 한 개의 고정 적합 모델에 대한 인과 효과 비교가 아니라 피처 집합별 학습 절차 비교다.

ElasticNet / discharge의 7.72%가 최저지만 Ridge / discharge의 8.25%와 차이는 약 0.53%p다. fold 표준편차와 소표본을 고려하면 통계적으로 확실한 우월성을 입증한 것은 아니다. 사전 선택 규칙인 개발 CV 평균 최솟값을 따랐다.

## 안쪽 교차검증 대조

바깥 5개 훈련 집합과 최종 개발 집합의 안쪽 그룹 분리 총 {len(records)}개를 실제 splitter·입력 순서로 다시 구성했다. 모든 안쪽 fold에서 셀·충전 정책 중복이 없고 hold-out·외부 셀이 들어가지 않는다. 상세 ID는 `../results/inner_cv_audit.json`에 있다.
''',encoding='utf-8')
    print(f'Audit passed: 17 frozen candidate grids and {len(records)} inner group folds; feature dictionary completed.')


if __name__=='__main__':
    audit()
