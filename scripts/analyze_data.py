"""Cell-level EDA for all three batches. No predictive model is fitted for DAY 1."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
RESULTS = ROOT / 'results'
FIG = RESULTS / 'figures'
COLORS = {1: '#2463A6', 2: '#E67E22', 3: '#288C79'}
NOMINAL = 1.1
EOL = 0.8 * NOMINAL

def setup():
    FIG.mkdir(parents=True, exist_ok=True)
    font = ROOT / 'references/NanumGothic-Regular.ttf'
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': 180})

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def policy_values(policy):
    match = re.search(r'([\d.]+)C\(([\d.]+)%\)-([\d.]+)C', policy)
    if not match:
        return np.nan, np.nan, np.nan, np.nan
    c1, soc, c2 = map(float, match.groups())
    # The named stages cover 0–80% SOC; after 80% all cells use a common CC-CV step.
    f = soc / 100
    effective = 0.8 / (f / c1 + (0.8 - f) / c2) if 0 < f <= 0.8 else np.nan
    return c1, soc, c2, effective

def knee_estimate(frame):
    s = frame.loc[(frame['QD'] > 0) & (frame['QD'] < 1.5)].sort_values('cycle')
    x = s['cycle'].to_numpy()
    y = s['QD'].rolling(11, center=True, min_periods=1).median().to_numpy()
    if len(x) < 120:
        return None
    base = np.column_stack([np.ones(len(x)), x])
    linear = np.linalg.lstsq(base, y, rcond=None)[0]
    sse_linear = np.sum((y - base @ linear)**2)
    candidates = np.unique(np.quantile(x, np.linspace(0.2, 0.9, 75)))
    best = None
    for k in candidates:
        if (x < k).sum() < 30 or (x >= k).sum() < 30:
            continue
        design = np.column_stack([base, np.maximum(0, x-k)])
        b = np.linalg.lstsq(design, y, rcond=None)[0]
        sse = np.sum((y - design @ b)**2)
        before, after = b[1], b[1] + b[2]
        if after >= 0 or after >= before:
            continue
        if best is None or sse < best['sse']:
            best = {'knee': float(k), 'slope_before': float(before), 'slope_after': float(after),
                    'sse': float(sse), 'improvement': float(1-sse/sse_linear) if sse_linear else 0}
    return best if best and best['improvement'] >= 0.2 else None

def interpolate_discharge(curves, cell_id, cycle, voltage):
    prefix = f'{cell_id}_c{cycle}_'
    i, v, q, t = (curves[prefix + field] for field in ['I', 'V', 'Qd', 't'])
    # Longest contiguous negative-current region avoids brief resistance pulses.
    mask = (i < -0.1) & np.isfinite(v) & np.isfinite(q) & np.isfinite(t)
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    runs = [r for r in runs if len(r) > 10]
    if not runs:
        return np.full(len(voltage), np.nan)
    run = max(runs, key=lambda r: t[r[-1]]-t[r[0]])
    # Some Batch 2 raw Qd points reset to exactly zero at low voltage while
    # current remains negative. They are not valid cumulative discharge charge.
    # Exclude those structural zeros before interpolation; retain the original
    # data and audit their count separately.
    run = run[q[run] > 1e-8]
    d = pd.DataFrame({'V': v[run], 'Q': q[run]}).groupby('V')['Q'].median().sort_index()
    out = np.interp(voltage, d.index.to_numpy(), d.to_numpy(), left=np.nan, right=np.nan)
    return out

def build_features(cells, summary, curves):
    rows, dq_arrays, quality = [], {}, []
    for _, cell in cells.iterrows():
        cid = cell.cell_id
        s = summary.loc[summary.cell_id == cid].sort_values('cycle')
        early = s.loc[(s.cycle >= 2) & (s.cycle <= 100)].copy()
        valid_q = early.QD.where((early.QD > 0) & (early.QD < 1.5))
        early.loc[early.IR <= 0, 'IR'] = np.nan
        # Unrealistic temperatures are missing sensor readings, not cool cells.
        for field in ['Tavg', 'Tmin', 'Tmax']:
            early.loc[(early[field] < -20) | (early[field] > 100), field] = np.nan
        row = cell.to_dict()
        row.update(dict(zip(['c_rate_1', 'switch_soc', 'c_rate_2', 'effective_c_rate'], policy_values(cell.charging_policy))))
        row['QD_mean'] = valid_q.mean()
        row['QD_std'] = valid_q.std()
        row['IR_mean'] = early.IR.mean()
        row['Tavg_mean'] = early.Tavg.mean()
        row['Tmax_mean'] = early.Tmax.mean()
        row['temperature_span'] = (early.Tmax - early.Tmin).mean()
        row['chargetime_mean'] = early.chargetime.where(early.chargetime > 0).mean()
        good = valid_q.notna()
        row['QD_slope'] = np.polyfit(early.loc[good, 'cycle'], valid_q[good], 1)[0] if good.sum() > 3 else np.nan
        qd_by_cycle = s.set_index('cycle')['QD']
        row['QD_2'] = qd_by_cycle.get(2, np.nan)
        row['QD_100'] = qd_by_cycle.get(100, np.nan)
        row['QD_change_100_2'] = row['QD_100'] - row['QD_2']
        row['QD_ratio_100_2'] = row['QD_100']/row['QD_2'] if row['QD_2'] > 0 else np.nan
        ir_by_cycle = early.set_index('cycle')['IR']
        row['IR_change_100_2'] = ir_by_cycle.get(100, np.nan) - ir_by_cycle.get(2, np.nan)
        prefix = f'{cid}_c10_'
        current, time = curves[prefix+'I'], curves[prefix+'t']
        positive = (current > 0.1) & np.isfinite(current) & np.isfinite(time)
        dt = np.diff(time)*60
        valid_segments = positive[:-1] & positive[1:] & (dt > 0)
        i2dt = ((current[:-1]**2 + current[1:]**2)/2*dt)[valid_segments].sum()
        duration = dt[valid_segments].sum()
        row['charge_I_rms_c10'] = float(np.sqrt(i2dt/duration)) if duration > 0 else np.nan
        row['charge_I_max_c10'] = float(np.max(current[positive])) if positive.any() else np.nan
        # Engineering proxy, not measured total heat. t is in minutes.
        row['heat_proxy_c10_J'] = float(i2dt*row['IR_mean'])
        for field in ['delta_q_logvar', 'delta_q_min', 'delta_q_mean', 'raw_delta_q_logvar']:
            row[field] = np.nan
        k10, k100 = f'{cid}_c10_Qdlin', f'{cid}_c100_Qdlin'
        if k10 in curves and k100 in curves:
            q10, q100 = curves[k10], curves[k100]
            if len(q10) == len(q100) and len(q10) > 20:
                delta = q100 - q10
                if np.isfinite(delta).all():
                    # Median-centering removes scalar capacity origin differences.
                    centered = delta - np.median(delta)
                    dq_arrays[cid] = {'raw': delta, 'centered': centered}
                    row['delta_q_logvar'] = np.log10(max(np.var(delta, ddof=1), 1e-15))
                    row['delta_q_min'] = float(np.min(centered))
                    row['delta_q_mean'] = float(np.mean(centered))
                    quality.append({'cell_id': cid, 'q10_first': float(q10[0]), 'q100_first': float(q100[0]),
                                    'delta_q_offset': float(np.median(delta)), 'finite': True})
            grid = np.linspace(2.1, 3.2, 500)
            raw10 = interpolate_discharge(curves, cid, 10, grid)
            raw100 = interpolate_discharge(curves, cid, 100, grid)
            rd = raw100 - raw10
            if np.isfinite(rd).all():
                row['raw_delta_q_logvar'] = float(np.log10(max(np.var(rd, ddof=1), 1e-15)))
        rows.append(row)
    return pd.DataFrame(rows), dq_arrays, pd.DataFrame(quality)

def analyze():
    setup()
    cells = pd.read_csv(DATA/'cells_analysis.csv')
    summary = pd.read_csv(DATA/'summary_analysis.csv.gz')
    curves = np.load(DATA/'early_curves.npz')
    features, dq, quality = build_features(cells, summary, curves)
    features['log_cycle_life'] = np.log10(features.cycle_life)
    features.to_csv(RESULTS/'cell_features.csv', index=False)
    quality.to_csv(RESULTS/'delta_q_quality.csv', index=False)
    stats_rows = []
    for batch, group in features.groupby('batch'):
        y = group.cycle_life
        stats_rows.append({'batch': int(batch), 'n': len(y), 'mean': y.mean(), 'median': y.median(), 'std': y.std(),
                           'min': y.min(), 'q25': y.quantile(.25), 'q75': y.quantile(.75), 'max': y.max(),
                           'short_n': int((y < 500).sum()), 'short_pct': 100*(y < 500).mean(),
                           'long_n': int((y > 1000).sum()), 'long_pct': 100*(y > 1000).mean(),
                           'capacity_increase_pct': 100*(group.QD_ratio_100_2 > 1).mean(),
                           'median_capacity_change_pct': 100*(group.QD_ratio_100_2.median()-1),
                           'n_policy': group.charging_policy.nunique()})
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(RESULTS/'batch_statistics.csv', index=False)
    features.nsmallest(8, 'cycle_life').to_csv(RESULTS/'shortest_cells.csv', index=False)
    # Q1: same bins, density rather than sample-size-driven counts.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4))
    for batch, group in features.groupby('batch'):
        axes[0].hist(group.cycle_life, bins=np.arange(100, 2401, 150), histtype='step', linewidth=2,
                     density=True, label=f'Batch {batch} (n={len(group)})', color=COLORS[batch])
    axes[0].axvline(500, color='#666', ls='--', lw=.8)
    axes[0].axvline(1000, color='#666', ls='--', lw=.8)
    axes[0].set(xlabel='총수명 (사이클)', ylabel='확률 밀도', title='배치별 수명 분포')
    axes[0].legend(fontsize=8)
    box = axes[1].boxplot([g.cycle_life for _, g in features.groupby('batch')], tick_labels=['Batch 1', 'Batch 2', 'Batch 3'], patch_artist=True)
    for b, patch in enumerate(box['boxes'], 1):
        patch.set_facecolor(COLORS[b]); patch.set_alpha(.5)
    axes[1].set(ylabel='총수명 (사이클)', title='중앙값과 분산 비교')
    fig.tight_layout(); save(fig, '01_cycle_life')
    # Q2: complete trajectories are EDA only and are never predictive inputs.
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6))
    norm = plt.Normalize(features.cycle_life.min(), features.cycle_life.max())
    for _, cell in features.iterrows():
        s = summary.loc[(summary.cell_id == cell.cell_id) & (summary.QD > 0) & (summary.QD < 1.5)]
        color = plt.cm.viridis(norm(cell.cycle_life))
        col = int(cell.batch)-1
        axes[0, col].plot(s.cycle, s.QD, color=color, alpha=.7, lw=.6)
        e = s.loc[s.cycle <= 100]
        axes[1, col].plot(e.cycle, e.QD, color=color, alpha=.7, lw=.7)
    for col in range(3):
        axes[0, col].axhline(EOL, color='#C44', ls='--', lw=.8)
        axes[0, col].set(title=f'Batch {col+1}: 전체 열화', xlabel='사이클', ylabel='QD (Ah)', ylim=(.82, 1.13))
        axes[1, col].set(title='초기 100사이클', xlabel='사이클', ylabel='QD (Ah)', xlim=(2,100), ylim=(.94,1.13))
    fig.tight_layout(); save(fig, '02_degradation')
    knees=[]
    fig, axes=plt.subplots(1,3,figsize=(11.6,3.8))
    for b in range(1,4):
        group=features.loc[features.batch==b]
        median=group.cycle_life.median()
        representative=group.loc[(group.cycle_life-median).abs().idxmin()]
        s=summary.loc[summary.cell_id==representative.cell_id]
        s=s.loc[(s.QD>0)&(s.QD<1.5)]
        ax=axes[b-1];ax.plot(s.cycle,s.QD,lw=1,color=COLORS[b])
        k=knee_estimate(s)
        if k:
            ax.axvline(k['knee'],ls='--',color='#B34D4D',label=f"추정 knee: {k['knee']:.0f}")
        ax.axhline(EOL,ls=':',color='#777')
        ax.set(title=f'Batch {b}: {representative.cell_id}',xlabel='사이클',ylabel='QD (Ah)');ax.legend(fontsize=8)
    for _,cell in features.iterrows():
        s=summary.loc[(summary.cell_id==cell.cell_id)&(summary.cycle<=cell.cycle_life)]
        k=knee_estimate(s)
        if k: knees.append({'cell_id':cell.cell_id,'batch':int(cell.batch),'cycle_life':cell.cycle_life,**k,'knee_fraction':k['knee']/cell.cycle_life})
    pd.DataFrame(knees).to_csv(RESULTS/'knee_estimates.csv',index=False)
    fig.tight_layout();save(fig,'03_knee')
    # Q3: stored uniform grid comes from the paper, not the introductory 2–3.6V range.
    fig,axes=plt.subplots(1,3,figsize=(11.6,4))
    for _,cell in features.iterrows():
        if cell.cell_id not in dq:continue
        delta=dq[cell.cell_id]['centered']; voltage=curves[f'{cell.cell_id}_Vdlin'] if f'{cell.cell_id}_Vdlin' in curves else np.linspace(3.5,2,len(delta))
        group='단수명 <500' if cell.cycle_life<500 else ('장수명 >1000' if cell.cycle_life>1000 else '중간 수명')
        color={'단수명 <500':'#C64B43','장수명 >1000':'#2463A6','중간 수명':'#B4BAC1'}[group]
        ax=axes[int(cell.batch)-1];ax.plot(voltage,delta,color=color,alpha=.75 if group!='중간 수명' else .35,lw=1 if group!='중간 수명' else .6)
        ax.set(title=f'Batch {int(cell.batch)}',xlabel='전압 (V)',ylabel='중앙값을 뺀 ΔQ (Ah)')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0],[0],color=c,label=l) for l,c in [('단수명 <500','#C64B43'),('장수명 >1000','#2463A6'),('중간 수명','#B4BAC1')]],fontsize=8)
    fig.tight_layout();save(fig,'04_delta_q')
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.8))
    for b,g in features.groupby('batch'):
        v=g[['delta_q_logvar','log_cycle_life']].dropna();r=pearsonr(v.iloc[:,0],v.iloc[:,1]).statistic
        ax=axes[int(b)-1];ax.scatter(g.delta_q_logvar,g.cycle_life,color=COLORS[b],s=25,alpha=.8)
        ax.set(xlabel='log10 Var(ΔQ)',ylabel='총수명 (사이클)',title=f'Batch {int(b)}: log–log r={r:.3f}',yscale='log')
    fig.tight_layout();save(fig,'05_delta_q_life')
    # Q4: all policy results remain in CSV, figures show extremes with n and SD.
    policies=features.groupby(['batch','charging_policy']).cycle_life.agg(['mean','std','count']).reset_index()
    policies.to_csv(RESULTS/'policy_statistics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11.6,5))
    for b in range(1,4):
        g=policies.loc[policies.batch==b].sort_values('mean')
        selected=pd.concat([g.head(4),g.tail(4)]).drop_duplicates('charging_policy').sort_values('mean')
        ax=axes[b-1];y=np.arange(len(selected))
        ax.barh(y,selected['mean'],xerr=selected['std'].fillna(0),color=COLORS[b],alpha=.7,capsize=2)
        ax.set_yticks(y,[f'{p} (n={int(n)})' for p,n in zip(selected.charging_policy,selected['count'])],fontsize=7)
        ax.set(xlabel='평균 총수명 ± SD',title=f'Batch {b}: 수명 상·하위 정책')
    fig.tight_layout();save(fig,'06_policy')
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.8))
    for b,g in features.groupby('batch'):
        ax=axes[int(b)-1];ax.scatter(g.effective_c_rate,g.cycle_life,color=COLORS[b],s=25,alpha=.8)
        v=g[['effective_c_rate','cycle_life']].dropna();r=spearmanr(v.iloc[:,0],v.iloc[:,1]).statistic
        ax.set(xlabel='0–80% SOC 유효 C-rate',ylabel='총수명 (사이클)',title=f'Batch {int(b)}: Spearman ρ={r:.3f}')
    fig.tight_layout();save(fig,'07_c_rate')
    # Q5: per-batch correlations and training-batch multicollinearity.
    names=['delta_q_logvar','delta_q_min','QD_2','QD_slope','IR_mean','IR_change_100_2','Tavg_mean','Tmax_mean','chargetime_mean','effective_c_rate']
    correlation=[]
    for b,g in features.groupby('batch'):
        for name in dict.fromkeys(names+['delta_q_mean','QD_change_100_2','raw_delta_q_logvar','charge_I_rms_c10','charge_I_max_c10','heat_proxy_c10_J']):
            v=g[[name,'cycle_life','log_cycle_life']].replace([np.inf,-np.inf],np.nan).dropna()
            if len(v)>3 and v[name].nunique()>1:
                correlation.append({'batch':int(b),'feature':name,'n':len(v),
                    'pearson_life':float(pearsonr(v[name],v.cycle_life).statistic),
                    'pearson_loglife':float(pearsonr(v[name],v.log_cycle_life).statistic),
                    'spearman_life':float(spearmanr(v[name],v.cycle_life).statistic)})
    corr=pd.DataFrame(correlation);corr.to_csv(RESULTS/'feature_correlations.csv',index=False)
    matrix=corr.pivot(index='feature',columns='batch',values='pearson_loglife').reindex(names)
    fig,axes=plt.subplots(1,2,figsize=(11.6,5.4),gridspec_kw={'width_ratios':[.9,1.6]})
    im=axes[0].imshow(matrix.to_numpy(),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    axes[0].set_xticks(range(3),['Batch 1','Batch 2','Batch 3']);axes[0].set_yticks(range(len(names)),names,fontsize=8)
    axes[0].set_title('피처와 log10(총수명)의 Pearson r')
    for j in range(3):
        for i in range(len(names)):
            val=matrix.iloc[i,j];axes[0].text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(val)>.6 else 'black')
    train=features.loc[features.batch==1,names];mc=train.corr();mc.to_csv(RESULTS/'batch1_feature_correlations.csv')
    axes[1].imshow(mc.to_numpy(),cmap='RdBu_r',vmin=-1,vmax=1)
    axes[1].set_xticks(range(len(names)),names,rotation=55,ha='right',fontsize=7);axes[1].set_yticks(range(len(names)),names,fontsize=7)
    axes[1].set_title('Batch 1 피처 간 상관: 다중공선성 점검')
    for i in range(len(names)):
        for j in range(len(names)):
            val=mc.iloc[i,j];axes[1].text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=6,color='white' if abs(val)>.6 else 'black')
    fig.colorbar(im,ax=axes,fraction=.018,pad=.03);save(fig,'08_correlations')
    pairs=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if abs(mc.loc[a,b])>.85:pairs.append({'feature_a':a,'feature_b':b,'r':float(mc.loc[a,b])})
    pd.DataFrame(pairs).to_csv(RESULTS/'multicollinearity.csv',index=False)
    drift=[]
    for b in (2,3):
        for name in ['cycle_life']+names:
            a=features.loc[features.batch==1,name].dropna();other=features.loc[features.batch==b,name].dropna()
            if len(a) and len(other):
                k=ks_2samp(a,other);drift.append({'batch':b,'feature':name,'median_batch1':a.median(),'median_other':other.median(),'ks_stat':k.statistic,'p_value':k.pvalue})
    pd.DataFrame(drift).to_csv(RESULTS/'batch_drift.csv',index=False)
    current_rows=[]
    for b,g in features.groupby('batch'):
        v=g[['charge_I_rms_c10','QD_slope']].dropna()
        current_rows.append({'batch':int(b),'n':len(v),'r_current_vs_QD_slope':float(pearsonr(v.iloc[:,0],v.iloc[:,1]).statistic)})
    pd.DataFrame(current_rows).to_csv(RESULTS/'current_degradation_correlations.csv',index=False)
    b2=features.loc[features.batch==2].copy()
    b2['structure']=np.where(b2.charging_policy.str.contains('newstructure'),'newstructure','regular')
    b2['policy_core']=b2.charging_policy.str.replace('-newstructure','',regex=False)
    b2.groupby(['policy_core','structure']).cycle_life.agg(['mean','count','std']).reset_index().to_csv(RESULTS/'batch2_structure_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11.6,3.8))
    b1=features.loc[features.batch==1]
    for policy in ['8C(15%)-3.6C','8C(35%)-3.6C']:
        representative=b1.loc[b1.charging_policy==policy].iloc[0]
        cid=representative.cell_id;ii=curves[f'{cid}_c10_I'];tt=curves[f'{cid}_c10_t']
        last_positive=np.flatnonzero(ii>.1)[-1]
        axes[0].plot(tt[:last_positive+1]-tt[0],ii[:last_positive+1]/NOMINAL,label=f'{policy}: {cid}')
    axes[0].set(xlabel='사이클 10 충전 경과 시간 (분)',ylabel='전류 / 공칭 용량 (C)',title='동일한 8C라도 지속 SOC 구간이 다름');axes[0].legend(fontsize=8)
    for b,g in features.groupby('batch'):
        axes[1].scatter(g.charge_I_rms_c10,g.QD_slope*1000,color=COLORS[b],label=f'Batch {int(b)}',s=24,alpha=.75)
    axes[1].set(xlabel='사이클 10 충전 전류 RMS (A)',ylabel='초기 QD 기울기 (mAh/사이클)',title='전류 패턴과 초기 용량 변화');axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'09_current_patterns')
    overview={'batches':stats.to_dict('records'),'feature_correlations':correlation,
              'multicollinearity':pairs,'n_knee':len(knees),'n_features':len(features),
              'invalid_summary_QD':int(((summary.QD<=0)|(~np.isfinite(summary.QD))).sum())}
    (RESULTS/'eda_overview.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    print(stats.to_string(index=False));print(corr.to_string(index=False));print('DONE',len(features),'cells',flush=True)

if __name__ == '__main__':
    analyze()
