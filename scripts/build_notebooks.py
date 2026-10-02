"""Create three concise notebooks with real computations and frozen model replay."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


SETUP = '''from pathlib import Path
import sys, json
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import display, Image, Markdown
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 4)
def show_plot(name):
    display(Image(filename=str(ROOT / f'results/figures/{name}.png')))
'''


def write(name, cells):
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(notebook, ROOT/'notebooks'/name)


def build():
    write('01_EDA.ipynb', [
        md('''# 01. EDA — 指定 배터리 데이터의 특성
울산캠퍼스 1반 김진형 · 회귀 과제

Batch 1: 2017-05-12 / 필수 Batch 2: 2018-02-20 / 추가 Batch 3: 2018-04-12.
전체 열화와 knee는 사후 설명이며 초기 예측 입력에서 제외한다.'''.replace('指定','지정')),
        code(SETUP),
        md('## 데이터 품질과 제외 기준'),
        code("audit = json.loads((ROOT/'results/quality_audit.json').read_text())\nprint(audit['raw_counts'], '→', audit['analysis_counts'])\ndisplay(pd.read_csv(ROOT/'results/cell_exclusions.csv')[['cell_id','batch','reasons']])"),
        md('## Q1. 수명 분포와 배치 차이\nBatch 2에는 학습 분포보다 짧은 수명의 셀이 많다. 불완료 기록 제외에 따른 장수명 선택 편향도 고려한다.'),
        code("display(pd.read_csv(ROOT/'results/batch_statistics.csv'))\nshow_plot('01_cycle_life')"),
        md('## Q2. 전체·초기 열화와 탐색적 knee\n초기 용량의 증가와 말기 가속 열화를 구분한다. 미래 knee는 입력이 아니다.'),
        code("display(pd.read_csv(ROOT/'results/knee_estimates.csv').groupby('batch')[['knee','knee_fraction']].median())\nshow_plot('02_degradation')\nshow_plot('03_knee')"),
        md('## Q3. ΔQ100−10(V)\n제공 전압 격자와 실제 사이클 번호를 사용한다. 상관은 예측 성능 지표와 다르다.'),
        code("f = pd.read_csv(ROOT/'results/features.csv')\nfor batch, part in f.groupby('batch'):\n    print(f'Batch {batch}: r={part.delta_q_logvar.corr(np.log10(part.cycle_life)):.4f}')\nshow_plot('04_delta_q')\nshow_plot('05_delta_q_life')"),
        md('## Q4. 정책과 실측 충전 전류\n같은 최대 전류라도 유지 SOC 구간과 실측 패턴이 다르다. 정책별 n이 작아 인과 효과를 단정하지 않는다.'),
        code("display(pd.read_csv(ROOT/'results/policy_statistics.csv'))\nshow_plot('06_policy')\nshow_plot('09_current_patterns')"),
        md('## Q5. 강한 신호와 다중공선성\n중복 신호에는 대표 변수와 규제를 사용하고 모델 비교로 추가 가치를 확인한다.'),
        code("display(pd.read_csv(ROOT/'results/feature_correlations.csv'))\nshow_plot('08_correlations')"),
    ])
    write('02_feature_engineering.ipynb', [
        md('# 02. 초기 피처 구현과 누수 점검\n셀 내부 피처는 2–100사이클만 사용한다. 결측 대체·표준화는 학습 fold에서만 적합한다.'),
        code(SETUP),
        md('## 구현된 피처 집합'),
        code("from src.features import FEATURE_SETS, build_features\nfrom src.preprocess import load_inputs\ndisplay(pd.DataFrame([{'집합':name,'입력 수':len(cols),'피처':', '.join(cols)} for name,cols in FEATURE_SETS.items()]))"),
        md('## 초기 신호에서 피처 다시 계산\n전체 분석 코호트의 같은 타깃을 유지하고, 계산 결과를 저장 피처와 비교한다.'),
        code("cells, summary, curves = load_inputs(ROOT)\nprint('입력 summary 최대 사이클:',summary.cycle.max())\ncomputed, _, _ = build_features(cells, summary, curves)\ncurves.close()\nsaved = pd.read_csv(ROOT/'results/features.csv').set_index('cell_id')\ncols = sorted(set().union(*map(set,FEATURE_SETS.values())))\nnp.testing.assert_allclose(computed.set_index('cell_id').loc[saved.index,cols],saved[cols],rtol=1e-10,atol=1e-12,equal_nan=True)\nprint('초기 피처 재계산 일치')\ndisplay(computed[['cell_id','batch',*FEATURE_SETS['discharge']]].head())"),
        md('## 결측 처리와 중복·분포 차이'),
        code("display(saved.groupby('batch')[cols].agg(lambda x:x.isna().sum()))\ndisplay(pd.read_csv(ROOT/'results/multicollinearity.csv').head(10))\nshow_plot('13_feature_drift')"),
        md('## 고정 분리 검증\n개발/hold-out 및 모든 CV fold 사이의 셀·정책 중복을 거부한다.'),
        code("from src.preprocess import load_features, validate_split\nf, split = load_features(ROOT)\nassert validate_split(f,split)\nprint('개발:',len(split['development_cells']),'hold-out:',len(split['holdout_cells']))\nprint('정책:',split['n_development_groups'],'/',split['n_holdout_groups'])"),
    ])
    write('03_modeling.ipynb', [
        md('''# 03. 회귀 모델 비교, 평가와 오류 분석
후보 학습과 중첩 그룹 튜닝은 `python -m src.train`으로 재현한다.
이 노트북은 실제 저장 결과와 선택 기록을 읽고, 저장 모델을 다시 불러와 외부 예측을 재계산한다.
외부 평가 결과로 모델을 교체하지 않는다.'''),
        code(SETUP),
        md('## 개발 데이터의 후보 비교'),
        code("comparison = pd.read_csv(ROOT/'results/cv_comparison.csv')\ndisplay(comparison)\nshow_plot('10_cv_candidates')"),
        md('## 최종 선택과 하이퍼파라미터'),
        code("selection = json.loads((ROOT/'results/selection.json').read_text())\nprint(selection['selected_model'],selection['best_params'])\nprint(selection['features'],'학습 셀:',selection['n_fit'])\nassert set(selection['fit_cells']) == set(json.loads((ROOT/'results/split_design.json').read_text())['development_cells'])"),
        md('## 저장 모델 재평가\n훈련 당시의 전처리 통계를 유지한 동일 모델로 hold-out·Batch 2·Batch 3 예측을 재현한다.'),
        code("import joblib\nfrom src.train import metrics\nfrom src.preprocess import load_features\nmodel = joblib.load(ROOT/'models/final_model.joblib')\nf,_ = load_features(ROOT);by_id=f.set_index('cell_id')\npred = pd.read_csv(ROOT/'results/predictions.csv')\npred = pred.loc[pred.model == selection['selected_model']]\nrows=[]\nfor subset,part in pred.groupby('subset'):\n    replay=model['model'].predict(by_id.loc[part.cell_id,model['features']])\n    np.testing.assert_allclose(replay,part.predicted,rtol=1e-10,atol=1e-8)\n    rows.append({'구분':subset,**metrics(part.actual,replay)})\ndisplay(pd.DataFrame(rows))\nshow_plot('11_model_parity')"),
        md('## 과제 요구 성능표와 Gap\nMAPE는 %이고 Gap은 %p다. 양수 Gap은 뒤 평가의 오차 증가를 뜻한다.'),
        code("display(pd.read_csv(ROOT/'results/model_performance.csv'))\ndisplay(pd.read_csv(ROOT/'results/mape_bootstrap.csv'))"),
        md('## 단일 피처 기준과 비교\nBatch 2에서는 CV로 고른 최종 모델보다 단일 피처 기준이 낫다. 이 실패도 결과에 포함한다.'),
        code("display(pd.read_csv(ROOT/'results/evaluation_metrics.csv'))\nshow_plot('14_baseline_comparison')"),
        md('## 오류 사례, 피처 이동과 계수\nB2 과대 예측·B3 장수명 과소 예측을 구분한다. 물리 해석은 인과 효과가 아닌 검증할 가설이다.'),
        code("display(pd.read_csv(ROOT/'results/largest_errors.csv').head(6))\ndisplay(pd.read_csv(ROOT/'results/error_groups.csv'))\ndisplay(pd.read_csv(ROOT/'results/model_coefficients.csv'))\nshow_plot('12_model_errors')\nshow_plot('15_model_coefficients')"),
        md('## 전기전자공학 시사점과 후기'),
        code("text=(ROOT/'docs/DAY2_분석보고서.md').read_text()\nstart=text.index('## 10.');end=text.index('## 12.')\ndisplay(Markdown(text[start:end].replace('<!-- PAGEBREAK -->','')))"),
    ])
    print('Three workflow notebooks created.')


if __name__=='__main__':
    build()
