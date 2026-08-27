"""
run_all_results.py
==================
Generates ALL results needed for the Kitchener Climate Report.

Run this ONE script in Colab and it produces every file the report needs.

REQUIRED INPUT FILES (must be in same folder):
  Water_Main_Breaks.csv
  Water_Mains.csv
  modelling_dataset.csv
  projections_annual.csv          ← you already have this

OUTPUT FILES PRODUCED:
  model_results_both_splits.csv   → Table 1 (model performance)
  coefficients_ci_splitA.csv      → Table 2 (CI coefficients, robustness)
  coefficients_ci_splitB.csv      → Table 2 (CI coefficients, PRIMARY)
  coefficients_di_splitA.csv      → DI coefficients
  coefficients_di_splitB.csv      → DI coefficients
  predictions_ci_splitA.csv       → Figure 3 data
  predictions_ci_splitB.csv       → Figure 3 data (PRIMARY)
  predictions_di_splitA.csv       → Figure 3 data
  predictions_di_splitB.csv       → Figure 3 data (PRIMARY)
  di_pipe_level_coefficients.csv  → DI pipe-level paragraph + Table 8
  di_pipe_level_summary.csv       → DI pipe-level paragraph + Figure 4
  design_life_results.csv         → Table 4 (design life)
  design_life_pipe_predictions.csv→ design life supporting data

Run:
  pip install statsmodels scipy pandas numpy
  python run_all_results.py
"""

import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import scipy.stats as stats
warnings.filterwarnings('ignore')

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

def p(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def fit_nb(df, features, offset='km_primary'):
    X = sm.add_constant(df[features])
    y = df['break_count']
    off = np.log(df[offset].clip(lower=0.001))
    return GLM(y, X, family=families.NegativeBinomial(alpha=1.0),
               offset=off).fit(maxiter=200, disp=False)

def pred_nb(mod, df, features, offset='km_primary', alpha=0.10):
    X   = sm.add_constant(df[features], has_constant='add')
    off = np.log(df[offset].clip(lower=0.001))
    mu  = mod.predict(X, offset=off)
    a   = mod.scale
    lo  = stats.nbinom.ppf(alpha/2,   n=1/a, p=1/(1+a*mu))
    hi  = stats.nbinom.ppf(1-alpha/2, n=1/a, p=1/(1+a*mu))
    return mu.values, lo, hi

def annual_rate_mae(pred_df, km_col='km_primary'):
    ann = pred_df.groupby('year').agg(
        actual=('break_count','sum'),
        predicted=('predicted','sum'),
        km=(km_col,'mean')).reset_index()
    ann['act_rate']  = ann['actual']    / ann['km']
    ann['pred_rate'] = ann['predicted'] / ann['km']
    return round(float(np.mean(np.abs(ann['act_rate']-ann['pred_rate']))),4), ann

def mase(actual, predicted, baseline_mae):
    return round(float(np.mean(np.abs(actual-predicted)))/baseline_mae, 3)

def coverage90(actual, lo, hi):
    return round(float(np.mean((actual>=lo)&(actual<=hi)))*100, 1)


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
p("STEP 1: Loading data")

d  = pd.read_csv(os.path.join(script_dir, 'modelling_dataset.csv'))
b  = pd.read_csv(os.path.join(script_dir, 'Water_Main_Breaks.csv'),
                 encoding='utf-8-sig', low_memory=False)
m  = pd.read_csv(os.path.join(script_dir, 'Water_Mains.csv'),
                 encoding='utf-8-sig', low_memory=False)
pr = pd.read_csv(os.path.join(script_dir, 'projections_annual.csv'))

b['dt'] = pd.to_datetime(b['Incident date'], errors='coerce')
b['yr'] = b['dt'].dt.year
m['iy'] = pd.to_datetime(m['INSTALLATION_DATE'], errors='coerce').dt.year

print(f"  modelling_dataset: {d.shape}")
print(f"  breaks: {len(b):,}  |  mains: {len(m):,}  |  projections: {len(pr):,}")


# ── 2. NEGATIVE BINOMIAL COHORT MODELS — BOTH SPLITS ─────────────────────────
p("STEP 2: Fitting negative binomial models (Split A and B)")

# Features selected by stepwise AIC (confirmed from notebook)
CI_FEAT = ['FI', 'Tmean_lag1', 'FD_lag1', 'cos_month']
DI_A_FEAT = ['FI', 'Tmean_lag2', 'FTC_lag1', 'FTC_lag3', 'FD_lag2']
DI_B_FEAT = ['FI', 'FI_lag2', 'FD_lag2']

results_rows = []

for mat, feat_A, feat_B in [('CI', CI_FEAT, CI_FEAT),
                              ('DI', DI_A_FEAT, DI_B_FEAT)]:
    md = d[d['material']==mat].copy()

    # Split A
    trA  = md[md['year']<=2016]
    vaA  = md[(md['year']>=2017)&(md['year']<=2021)]
    teA  = md[md['year']>=2022]
    trvaA= md[md['year']<=2021]

    # M0 baseline for Split A
    m0A = trA.groupby('month_num')['break_count'].mean()
    m0_mae_teA = float(np.mean(np.abs(teA['break_count'].values -
                                       teA['month_num'].map(m0A).values)))

    # Fit Split A — refit on train+val for test evaluation
    modA = fit_nb(trvaA, feat_A)
    pA, loA, hiA = pred_nb(modA, teA, feat_A)
    predA = teA.copy(); predA['predicted']=pA; predA['lower_90']=loA; predA['upper_90']=hiA
    predA['residual'] = teA['break_count'].values - pA
    mae_rate_A, _ = annual_rate_mae(predA)
    results_rows.append({'material':mat,'split':'A','model':'M2_stepwise',
        'train_years':'1997-2016','test_years':'2022-2025',
        'MAE_monthly':round(float(np.mean(np.abs(teA['break_count'].values-pA))),3),
        'MAE_annual':round(float(np.mean(np.abs(
            teA.groupby('year')['break_count'].sum().values -
            pd.Series(pA,index=teA.index).groupby(teA['year']).sum().values))),1),
        'MAE_rate':mae_rate_A,
        'MASE':mase(teA['break_count'].values, pA, m0_mae_teA),
        'Coverage_90':coverage90(teA['break_count'].values, loA, hiA),
        'Bias':round(float(np.mean(pA-teA['break_count'].values)),3)})

    # Save coefficients Split A
    coef_A = pd.DataFrame({'feature':modA.params.index,
        'coefficient':modA.params.values.round(5),
        'std_error':modA.bse.values.round(5),
        'p_value':modA.pvalues.values.round(4),
        'significant':modA.pvalues.values<0.05})
    coef_A.to_csv(os.path.join(script_dir,
        f'coefficients_{mat.lower()}_splitA.csv'), index=False)

    # Save predictions Split A
    predA[['material','month_str','year','month_num','break_count',
           'km_primary','predicted','lower_90','upper_90','residual']
          ].to_csv(os.path.join(script_dir,
        f'predictions_{mat.lower()}_splitA.csv'), index=False)

    print(f"  {mat} Split A: MAE_rate={mae_rate_A}  "
          f"MASE={results_rows[-1]['MASE']}")

    # Split B
    trB = md[md['year']<=2020]
    teB = md[md['year']>=2021]

    # M0 baseline for Split B
    m0B = trB.groupby('month_num')['break_count'].mean()
    m0_mae_teB = float(np.mean(np.abs(teB['break_count'].values -
                                       teB['month_num'].map(m0B).values)))

    modB = fit_nb(trB, feat_B)
    pB, loB, hiB = pred_nb(modB, teB, feat_B)
    predB = teB.copy(); predB['predicted']=pB; predB['lower_90']=loB; predB['upper_90']=hiB
    predB['residual'] = teB['break_count'].values - pB
    mae_rate_B, _ = annual_rate_mae(predB)
    results_rows.append({'material':mat,'split':'B','model':'M2_stepwise',
        'train_years':'1997-2020','test_years':'2021-2025',
        'MAE_monthly':round(float(np.mean(np.abs(teB['break_count'].values-pB))),3),
        'MAE_annual':round(float(np.mean(np.abs(
            teB.groupby('year')['break_count'].sum().values -
            pd.Series(pB,index=teB.index).groupby(teB['year']).sum().values))),1),
        'MAE_rate':mae_rate_B,
        'MASE':mase(teB['break_count'].values, pB, m0_mae_teB),
        'Coverage_90':coverage90(teB['break_count'].values, loB, hiB),
        'Bias':round(float(np.mean(pB-teB['break_count'].values)),3)})

    # Save coefficients Split B
    coef_B = pd.DataFrame({'feature':modB.params.index,
        'coefficient':modB.params.values.round(5),
        'std_error':modB.bse.values.round(5),
        'p_value':modB.pvalues.values.round(4),
        'significant':modB.pvalues.values<0.05})
    coef_B.to_csv(os.path.join(script_dir,
        f'coefficients_{mat.lower()}_splitB.csv'), index=False)

    # Save predictions Split B
    predB[['material','month_str','year','month_num','break_count',
           'km_primary','predicted','lower_90','upper_90','residual']
          ].to_csv(os.path.join(script_dir,
        f'predictions_{mat.lower()}_splitB.csv'), index=False)

    print(f"  {mat} Split B: MAE_rate={mae_rate_B}  "
          f"MASE={results_rows[-1]['MASE']}")

# Save model results
results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(script_dir,'model_results_both_splits.csv'), index=False)
print(f"\n  Saved: model_results_both_splits.csv")
print(results_df[['material','split','MAE_rate','MASE','Coverage_90','Bias']
                 ].to_string(index=False))


# ── 3. DI PIPE-LEVEL ANALYSIS ─────────────────────────────────────────────────
p("STEP 3: DI pipe-level model (age + prior breaks)")

di_mains = m[(m['MATERIAL']=='DI')&(m['Shape__Length']>0)&(m['iy'].notna())].copy()
bm = b[(b['Type of Asset Broken']=='MAIN')&
       (b.yr>=1997)&(b.yr<=2025)&
       (b['Asset Material']=='DI')].copy()
bm['aid'] = pd.to_numeric(bm['Related Asset ID'], errors='coerce')
bm_s = bm.sort_values(['aid','dt'])
bm_s['prior'] = bm_s.groupby('aid').cumcount()

linked = bm_s[bm_s['aid'].isin(di_mains['WATMAINID'])].merge(
    di_mains[['WATMAINID','iy','Shape__Length']],
    left_on='aid', right_on='WATMAINID', how='left')
linked['age'] = linked['yr'] - linked['iy']
linked = linked[(linked['age']>=0)&(linked['age']<=120)].copy()

brk_cnt = linked.groupby(['WATMAINID','yr']).size().reset_index(name='breaks')
prior_lu = linked.sort_values(['WATMAINID','yr']).groupby(
    ['WATMAINID','yr'])['prior'].min().reset_index()

rows = []
for _, pipe in di_mains.iterrows():
    pid = pipe['WATMAINID']; iy = int(pipe['iy'])
    length_km = pipe['Shape__Length']/1000.0
    for yr in range(max(1997,iy), 2026):
        age = yr - iy
        br = brk_cnt[(brk_cnt['WATMAINID']==pid)&(brk_cnt['yr']==yr)]
        n_b = int(br['breaks'].values[0]) if len(br) else 0
        pb = prior_lu[(prior_lu['WATMAINID']==pid)&(prior_lu['yr']<yr)]
        pn = int(pb['prior'].max()+1) if len(pb) else 0
        rows.append({'pipe_id':pid,'year':yr,'age':age,
                     'prior_breaks':pn,'breaks':n_b,'length_km':length_km})

panel = pd.DataFrame(rows)
panel = panel[(panel['age']>=0)&(panel['length_km']>0)].copy()
panel['log_length']       = np.log(panel['length_km'])
panel['prior_breaks_cap'] = panel['prior_breaks'].clip(upper=5)
panel['age_sq']           = panel['age']**2 / 1000

X   = sm.add_constant(panel[['age','age_sq','prior_breaks_cap']])
mod_di_pipe = GLM(panel['breaks'], X,
                  family=families.NegativeBinomial(alpha=1.0),
                  offset=panel['log_length']).fit(maxiter=200, disp=False)

c_const = mod_di_pipe.params['const']
c_age   = mod_di_pipe.params['age']
c_agesq = mod_di_pipe.params['age_sq']
c_prior = mod_di_pipe.params['prior_breaks_cap']

print(f"  AIC: {mod_di_pipe.aic:.1f}")
print(f"  Coefficients:")
for f, c, p_v in zip(mod_di_pipe.params.index,
                      mod_di_pipe.params.values,
                      mod_di_pipe.pvalues.values):
    print(f"    {f:22s}  coef={c:+.5f}  p={p_v:.4f}")

# Rate multipliers
print(f"\n  Rate multipliers by prior break count:")
multipliers = []
for pb in range(6):
    mult = round(float(np.exp(c_prior * pb)), 2)
    print(f"    {pb} prior breaks: {mult}x")
    multipliers.append({'prior_breaks':pb,'rate_multiplier':mult})

# Peak age
peak_age = round(-c_age * 1000 / (2*c_agesq), 1)
print(f"\n  Age at peak rate: {peak_age} years")

# Save coefficients
coef_pipe = pd.DataFrame({'feature':mod_di_pipe.params.index,
    'coefficient':mod_di_pipe.params.values.round(5),
    'std_error':mod_di_pipe.bse.values.round(5),
    'p_value':mod_di_pipe.pvalues.values.round(4),
    'significant':mod_di_pipe.pvalues.values<0.05})
coef_pipe.to_csv(os.path.join(script_dir,'di_pipe_level_coefficients.csv'), index=False)

# Save summary
summary_pipe = pd.DataFrame([
    {'finding':'Prior break multiplier per additional break',
     'value':round(float(np.exp(c_prior)),3),
     'note':'Each prior break multiplies rate by this factor'},
    {'finding':'Age at peak break rate (years)',
     'value':peak_age,
     'note':'Beyond this age, survivorship bias dominates'},
    {'finding':'Total pipe-years in panel','value':len(panel),
     'note':'Sample size'},
    {'finding':'Total DI breaks in panel','value':int(panel['breaks'].sum()),
     'note':'Total events'},
    {'finding':'Model AIC','value':round(mod_di_pipe.aic,1),'note':''},
] + [{'finding':f'{r["prior_breaks"]} prior breaks → rate multiplier',
      'value':r['rate_multiplier'],'note':''} for r in multipliers])
summary_pipe.to_csv(os.path.join(script_dir,'di_pipe_level_summary.csv'), index=False)

print(f"\n  Saved: di_pipe_level_coefficients.csv")
print(f"  Saved: di_pipe_level_summary.csv")


# ── 4. DESIGN LIFE ANALYSIS ───────────────────────────────────────────────────
p("STEP 4: DI design life analysis (75 years, horizon 2050)")

DESIGN_LIFE_DI = 75
START_YEAR     = 2025
HORIZON_YEAR   = 2050
KM_DI          = 322.05
HIST_DI_RATE   = 0.0663

di = di_mains.copy()
di['age_2025']   = START_YEAR - di['iy']
di['dl_year']    = di['iy'] + DESIGN_LIFE_DI
di['length_km'] = di['Shape__Length'] / 1000.0

prior_2025 = panel[panel['year']==2025][['pipe_id','prior_breaks']].rename(
    columns={'prior_breaks':'pb_2025'})
di = di.merge(prior_2025, left_on='WATMAINID', right_on='pipe_id', how='left')
di['pb_2025'] = di['pb_2025'].fillna(0).clip(upper=5).astype(int)
di_sim = di[(di['dl_year'] > START_YEAR) & (di['age_2025'] >= 0)].copy()

print(f"  DI pipes in simulation: {len(di_sim):,}")
print(f"  Of which reach DL before 2050: "
      f"{((di_sim['dl_year']>START_YEAR)&(di_sim['dl_year']<=HORIZON_YEAR)).sum()}")

def pipe_rate_fn(age, prior):
    return np.exp(c_const + c_age*age + c_agesq*age**2/1000 + c_prior*min(prior,5))

# Historical baseline rate per pipe in 2025
baseline_rates = di_sim.apply(
    lambda r: pipe_rate_fn(r['age_2025'], r['pb_2025']), axis=1)
baseline_mean = baseline_rates.mean()

dl_results  = []
pipe_preds  = []

scenarios = [('historical', None, None)] + [
    (f"{gcm}_{ssp}", gcm, ssp)
    for gcm in ['CanESM5','MIROC6']
    for ssp in ['ssp126','ssp245','ssp370','ssp585']
]

for (label, gcm, ssp) in scenarios:
    if label == 'historical':
        clim_mult = {yr: 1.0 for yr in range(2026, HORIZON_YEAR+1)}
    else:
        sub = pr[(pr['model']==gcm)&(pr['scenario']==ssp)]
        base_rate = HIST_DI_RATE
        clim_mult = dict(zip(sub['year'],
                              sub['rate_di'].values / base_rate))

    scen_rows = []
    for _, pipe in di_sim.iterrows():
        age_now  = int(pipe['age_2025'])
        prior_now= int(pipe['pb_2025'])
        dl_yr    = int(pipe['dl_year'])
        sim_yrs  = int(min(dl_yr, HORIZON_YEAR) - START_YEAR)
        if sim_yrs <= 0:
            continue

        years_arr    = np.arange(1, sim_yrs+1)
        yrs_cal      = START_YEAR + years_arr
        ages_arr     = age_now + years_arr
        log_rates    = (c_const + c_age*ages_arr +
                        c_agesq*ages_arr**2/1000 + c_prior*min(prior_now,5))
        base_rates_yr= np.exp(log_rates)
        clim_arr     = np.array([clim_mult.get(int(y), 1.0) for y in yrs_cal])
        rates_yr     = base_rates_yr * clim_arr

        hazard = 1 - np.exp(-rates_yr * pipe['length_km'])
        surv   = np.concatenate([[1.0], np.cumprod(1-hazard)])
        p_fail_yr = surv[:-1] * hazard

        mask_dl = yrs_cal <= dl_yr
        prob_before_dl  = float(p_fail_yr[mask_dl].sum())
        prob_by_horizon = float(p_fail_yr.sum())

        if prob_by_horizon > 0:
            mean_fail_yr = float(np.sum(yrs_cal*p_fail_yr)/prob_by_horizon)
            years_lost   = dl_yr - mean_fail_yr
        else:
            mean_fail_yr = np.nan
            years_lost   = np.nan

        scen_rows.append({'scenario':label,
            'pipe_id':pipe['WATMAINID'],
            'age_2025':age_now,'dl_year':dl_yr,
            'p_fail_before_dl':round(prob_before_dl,4),
            'p_fail_by_2050':round(prob_by_horizon,4),
            'mean_failure_year':round(mean_fail_yr,1) if not np.isnan(mean_fail_yr) else np.nan,
            'years_lost':round(years_lost,1) if not np.isnan(years_lost) else np.nan})

    scen_df = pd.DataFrame(scen_rows)
    pipe_preds.append(scen_df)

    pct_before_dl = scen_df['p_fail_before_dl'].mean()*100
    pct_by_2050   = scen_df['p_fail_by_2050'].mean()*100
    mean_yrs_lost = scen_df['years_lost'].mean()

    print(f"  {label:<25}  %fail before DL={pct_before_dl:.2f}%  "
          f"yrs lost={mean_yrs_lost:.1f}")

    dl_results.append({'scenario':label,
        'pct_fail_before_dl':round(pct_before_dl,2),
        'pct_fail_by_2050':round(pct_by_2050,2),
        'mean_years_lost':round(mean_yrs_lost,2)})

dl_df = pd.DataFrame(dl_results)
dl_df.to_csv(os.path.join(script_dir,'design_life_results.csv'), index=False)

pipe_pred_all = pd.concat(pipe_preds, ignore_index=True)
pipe_pred_all.to_csv(os.path.join(script_dir,
    'design_life_pipe_predictions.csv'), index=False)

print(f"\n  Saved: design_life_results.csv")
print(f"  Saved: design_life_pipe_predictions.csv")


# ── 5. PRINT COMPLETE RESULTS SUMMARY ────────────────────────────────────────
p("COMPLETE RESULTS SUMMARY FOR REPORT")

print("\n  TABLE 1: Model Performance")
print(results_df[['material','split','train_years','test_years',
                   'MAE_rate','MASE','Coverage_90','Bias']].to_string(index=False))

print("\n  TABLE 2: CI Coefficients (Split B)")
ci_coef = pd.read_csv(os.path.join(script_dir,'coefficients_ci_splitB.csv'))
print(ci_coef.to_string(index=False))

print("\n  TABLE 3: Projection summary (CI, end of century)")
proj30 = pd.read_csv(os.path.join(script_dir,'projections_30yr_summary.csv'))
eoc = proj30[proj30['window']=='2071-2100']
print(eoc[['model','scenario','mean_rate_ci','pct_change_ci',
           'mean_rate_di','pct_change_di']].to_string(index=False))

print("\n  DI PIPE-LEVEL KEY NUMBERS:")
print(f"    Prior break multiplier: {float(np.exp(c_prior)):.2f}x per break")
print(f"    Age peaks at:           {peak_age} years")
print(f"    p-value (prior_breaks): {mod_di_pipe.pvalues['prior_breaks_cap']:.4f}")

print("\n  TABLE 4: Design Life Results")
print(dl_df.to_string(index=False))

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
print("""
Files produced:
  model_results_both_splits.csv    → Table 1
  coefficients_ci_splitA/B.csv    → Table 2
  coefficients_di_splitA/B.csv    → DI model
  predictions_ci/di_splitA/B.csv  → Figure 3
  di_pipe_level_coefficients.csv  → DI pipe-level
  di_pipe_level_summary.csv       → DI pipe-level + Figure 4
  design_life_results.csv         → Table 4
  design_life_pipe_predictions.csv→ supporting data

Next: run make_figures.py to generate the 4 figures.
""")
