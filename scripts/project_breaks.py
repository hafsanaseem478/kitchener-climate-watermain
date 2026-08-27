"""
project_breaks.py
=================
Projects water main break rates for Kitchener to 2100 under 8 CMIP6
scenarios (2 models x 4 SSPs) using the fitted negative binomial models.

MODELS USED:
  CI : features = [FI, Tmean_lag1, FD_lag1, cos_month]
       trained on 1997-2021 (train + val combined)
  DI : two variants reported:
       Split A features = [FI, Tmean_lag2, FTC_lag1, FTC_lag3, FD_lag2]
       Split B features = [FI, FI_lag2, FD_lag2]  ← primary (lower MAE_rate)

PROJECTION SETUP:
  - Frozen network: km in service held at 2025 levels
    CI = 151.28 km,  DI = 322.05 km
  - Climate: CMIP6 monthly covariates from process_cmip6.py output
  - Lags: computed from projected data (lag-1 = prior projected month)
  - Horizon: 2026-2100 in 30-year windows (2026-2055, 2041-2070, 2071-2100)

OUTPUTS:
  projections_annual.csv         -- annual break counts & rates per scenario
  projections_30yr_summary.csv   -- mean rates per 30-yr window per scenario
  projections_scenario_spread.csv-- scenario spread (max-min) per decade

Run:
  python project_breaks.py

Requires all cmip6_monthly_*.csv files in same folder, plus modelling_dataset.csv
"""

import os
import glob
import warnings
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


# ── 0. SETTINGS ───────────────────────────────────────────────────────────────

# Frozen network exposure (2025 km values)
KM_CI = 151.28
KM_DI = 322.05

# Features from stepwise AIC selection (confirmed from notebook output)
CI_FEATURES  = ['FI', 'Tmean_lag1', 'FD_lag1', 'cos_month']
DI_FEATURES_A = ['FI', 'Tmean_lag2', 'FTC_lag1', 'FTC_lag3', 'FD_lag2']
DI_FEATURES_B = ['FI', 'FI_lag2', 'FD_lag2']  # primary DI model (lower MAE_rate)

# Lag columns needed to compute (lag-1, lag-2, lag-3)
LAG_COLS = ['Tmean', 'FI', 'FTC', 'FI_cum', 'FD', 'ADD']

# 30-year projection windows for summary
WINDOWS = [
    ('2026-2055', 2026, 2055),
    ('2041-2070', 2041, 2070),
    ('2071-2100', 2071, 2100),
]

# Historical baseline window (for % change calculation)
HIST_START = 1997
HIST_END   = 2025


# ── 1. REFIT MODELS ON FULL TRAINING DATA ────────────────────────────────────

print("=" * 60)
print("STEP 1: Fitting models on full historical data (1997-2025)")
print("=" * 60)
print("  Using same features as selected in notebook.")
print("  CI trained on 1997-2021 (train+val). DI Split B on 1997-2020.\n")

ds_path = os.path.join(script_dir, 'modelling_dataset.csv')
if not os.path.exists(ds_path):
    raise FileNotFoundError(
        "modelling_dataset.csv not found. "
        "Run build_final_dataset.py first."
    )

ds = pd.read_csv(ds_path)

def fit_nb(df, features, offset_col='km_primary'):
    X      = sm.add_constant(df[features])
    y      = df['break_count']
    offset = np.log(df[offset_col].clip(lower=0.001))
    return GLM(
        y, X,
        family=families.NegativeBinomial(alpha=1.0),
        offset=offset
    ).fit(maxiter=200, disp=False)


# CI: train on 1997-2021 (train + val, as done for final test evaluation)
ci_data    = ds[(ds['material']=='CI') & (ds['year']<=2021)].copy()
model_ci   = fit_nb(ci_data, CI_FEATURES)

# DI Split B: train on 1997-2020 (primary model with lower MAE_rate=0.0217)
di_data_B  = ds[(ds['material']=='DI') & (ds['year']<=2020)].copy()
model_di   = fit_nb(di_data_B, DI_FEATURES_B)

# DI Split A: also fit for comparison
di_data_A  = ds[(ds['material']=='DI') & (ds['year']<=2021)].copy()
model_di_A = fit_nb(di_data_A, DI_FEATURES_A)

print("  CI model coefficients:")
for feat, coef, pval in zip(model_ci.params.index,
                             model_ci.params.values,
                             model_ci.pvalues.values):
    sig = "✓" if pval < 0.05 else " "
    print(f"    {sig} {feat:20s}  coef={coef:+.5f}  p={pval:.4f}")

print("\n  DI model (Split B) coefficients:")
for feat, coef, pval in zip(model_di.params.index,
                              model_di.params.values,
                              model_di.pvalues.values):
    sig = "✓" if pval < 0.05 else " "
    print(f"    {sig} {feat:20s}  coef={coef:+.5f}  p={pval:.4f}")

print()

# Historical mean failure rate (baseline for % change)
ci_hist = ds[(ds['material']=='CI') & (ds['year']>=HIST_START) & (ds['year']<=HIST_END)]
di_hist = ds[(ds['material']=='DI') & (ds['year']>=HIST_START) & (ds['year']<=HIST_END)]

hist_ci_rate = (ci_hist.groupby('year')['break_count'].sum() / KM_CI).mean()
hist_di_rate = (di_hist.groupby('year')['break_count'].sum() / KM_DI).mean()

print(f"  Historical mean failure rate (1997-2025):")
print(f"    CI: {hist_ci_rate:.4f} breaks/km/year")
print(f"    DI: {hist_di_rate:.4f} breaks/km/year")


# ── 2. HELPER: ADD LAGS TO PROJECTED DATA ────────────────────────────────────

def add_lags(monthly_df, hist_climate_df, lag_cols=LAG_COLS, n_lags=3):
    """
    Add lag-1, lag-2, lag-3 columns to projected monthly data.
    For the first few projection months, lags come from historical data.

    monthly_df    : projected monthly covariates (2026-2100)
    hist_climate_df: historical monthly covariates (from climate_monthly.csv)
                    needed to fill lags for 2026-01 to 2026-03
    """
    # Combine historical + projected into one time series for lag computation
    hist_tail = hist_climate_df[
        hist_climate_df['year'] >= 2024
    ][['month_str','year','month'] + lag_cols].copy()

    proj = monthly_df[['month_str','year','month'] + lag_cols].copy()
    combined = pd.concat([hist_tail, proj], ignore_index=True)
    combined = combined.sort_values(['year','month']).reset_index(drop=True)
    combined['month_dt'] = pd.to_datetime(combined['month_str'] + '-01')

    # Build lookup
    lookup = combined.set_index('month_str')[lag_cols]

    def lag_str(ms, lag):
        dt = pd.to_datetime(ms + '-01') - pd.DateOffset(months=lag)
        return dt.strftime('%Y-%m')

    result = monthly_df.copy()
    for col in lag_cols:
        for lag in [1, 2, 3]:
            new_col = f"{col}_lag{lag}"
            result[new_col] = result['month_str'].apply(
                lambda ms: lookup.loc[lag_str(ms, lag), col]
                if lag_str(ms, lag) in lookup.index else np.nan
            )

    return result


def predict_nb_monthly(model, features, monthly_df, km, alpha=0.10):
    """
    Generate monthly point predictions and 90% prediction intervals.
    km: fixed exposure (frozen network scenario)
    """
    X      = sm.add_constant(monthly_df[features], has_constant='add')
    offset = np.log(np.full(len(monthly_df), km))
    mu     = model.predict(X, offset=offset)
    nb_a   = model.scale

    lower = stats.nbinom.ppf(alpha/2,   n=1/nb_a, p=1/(1 + nb_a*mu))
    upper = stats.nbinom.ppf(1-alpha/2, n=1/nb_a, p=1/(1 + nb_a*mu))

    return mu.values, lower, upper


# ── 3. LOAD CMIP6 FILES AND HISTORICAL CLIMATE ───────────────────────────────

print("=" * 60)
print("STEP 2: Loading CMIP6 projection files")
print("=" * 60)

# Load historical climate for lag computation
hist_path = os.path.join(script_dir, 'climate_monthly.csv')
if not os.path.exists(hist_path):
    raise FileNotFoundError(
        "climate_monthly.csv not found. "
        "Run build_climate.py first."
    )
hist_climate = pd.read_csv(hist_path)
print(f"  Historical climate loaded: {hist_climate.shape}")

# Load all CMIP6 monthly files
cmip6_files = glob.glob(os.path.join(script_dir, "cmip6_monthly_*.csv"))
if not cmip6_files:
    raise FileNotFoundError(
        "No cmip6_monthly_*.csv files found. "
        "Run process_cmip6.py first."
    )

print(f"  Found {len(cmip6_files)} CMIP6 files:")
for f in sorted(cmip6_files):
    df_tmp = pd.read_csv(f)
    print(f"    {os.path.basename(f)}: {len(df_tmp)} months")


# ── 4. RUN PROJECTIONS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Generating projections 2026-2100")
print("=" * 60)

all_annual = []

for fpath in sorted(cmip6_files):
    fname    = os.path.basename(fpath)
    # Parse model and scenario from filename
    # e.g. cmip6_monthly_CanESM5_ssp126.csv
    parts    = fname.replace('.csv','').split('_')
    model    = parts[2]    # CanESM5 or MIROC6
    scenario = parts[3]    # ssp126 etc

    print(f"\n  Processing {model} {scenario}...")

    proj = pd.read_csv(fpath)

    # Add sine/cosine month for CI model (cos_month is a feature)
    proj['sin_month'] = np.sin(2 * np.pi * proj['month'] / 12)
    proj['cos_month'] = np.cos(2 * np.pi * proj['month'] / 12)

    # Add lags
    proj_lagged = add_lags(proj, hist_climate)

    # Drop rows with any NaN in needed features
    ci_features_needed = CI_FEATURES
    di_features_needed = DI_FEATURES_B

    proj_ci = proj_lagged.dropna(subset=ci_features_needed).copy()
    proj_di = proj_lagged.dropna(subset=di_features_needed).copy()

    print(f"    CI projection months: {len(proj_ci)}")
    print(f"    DI projection months: {len(proj_di)}")

    # Generate monthly predictions
    ci_mu, ci_lo, ci_hi = predict_nb_monthly(model_ci, CI_FEATURES, proj_ci, KM_CI)
    di_mu, di_lo, di_hi = predict_nb_monthly(model_di, DI_FEATURES_B, proj_di, KM_DI)

    # Aggregate to annual
    proj_ci['pred_ci']    = ci_mu
    proj_ci['pred_ci_lo'] = ci_lo
    proj_ci['pred_ci_hi'] = ci_hi
    proj_ci['model']      = model
    proj_ci['scenario']   = scenario

    proj_di['pred_di']    = di_mu
    proj_di['pred_di_lo'] = di_lo
    proj_di['pred_di_hi'] = di_hi

    # Annual aggregation
    ann_ci = proj_ci.groupby('year').agg(
        breaks_ci    = ('pred_ci',    'sum'),
        breaks_ci_lo = ('pred_ci_lo', 'sum'),
        breaks_ci_hi = ('pred_ci_hi', 'sum'),
        model        = ('model',      'first'),
        scenario     = ('scenario',   'first'),
    ).reset_index()

    ann_di = proj_di.groupby('year').agg(
        breaks_di    = ('pred_di',    'sum'),
        breaks_di_lo = ('pred_di_lo', 'sum'),
        breaks_di_hi = ('pred_di_hi', 'sum'),
    ).reset_index()

    ann = ann_ci.merge(ann_di, on='year')

    # Failure rates
    ann['rate_ci']    = ann['breaks_ci']    / KM_CI
    ann['rate_ci_lo'] = ann['breaks_ci_lo'] / KM_CI
    ann['rate_ci_hi'] = ann['breaks_ci_hi'] / KM_CI
    ann['rate_di']    = ann['breaks_di']    / KM_DI
    ann['rate_di_lo'] = ann['breaks_di_lo'] / KM_DI
    ann['rate_di_hi'] = ann['breaks_di_hi'] / KM_DI

    # Pct change vs historical baseline
    ann['pct_change_ci'] = ((ann['rate_ci'] - hist_ci_rate) / hist_ci_rate * 100).round(1)
    ann['pct_change_di'] = ((ann['rate_di'] - hist_di_rate) / hist_di_rate * 100).round(1)

    all_annual.append(ann)

    # Quick sanity print
    for yr in [2030, 2050, 2070, 2090]:
        row = ann[ann['year']==yr]
        if len(row):
            r = row.iloc[0]
            print(f"    {yr}: CI={r['rate_ci']:.3f} br/km/yr ({r['pct_change_ci']:+.1f}%)  "
                  f"DI={r['rate_di']:.3f} br/km/yr ({r['pct_change_di']:+.1f}%)")


# ── 5. COMPILE AND SAVE RESULTS ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Compiling results")
print("=" * 60)

projections = pd.concat(all_annual, ignore_index=True)

# Save annual projections
ann_path = os.path.join(script_dir, 'projections_annual.csv')
projections.to_csv(ann_path, index=False)
print(f"  Saved: projections_annual.csv ({len(projections)} rows)")

# 30-year window summaries
summary_rows = []
for (window_name, yr_start, yr_end) in WINDOWS:
    sub = projections[(projections['year']>=yr_start) & (projections['year']<=yr_end)]
    for (model, scenario), grp in sub.groupby(['model','scenario']):
        summary_rows.append({
            'window':    window_name,
            'model':     model,
            'scenario':  scenario,
            'mean_rate_ci':     round(grp['rate_ci'].mean(), 4),
            'mean_rate_ci_lo':  round(grp['rate_ci_lo'].mean(), 4),
            'mean_rate_ci_hi':  round(grp['rate_ci_hi'].mean(), 4),
            'mean_rate_di':     round(grp['rate_di'].mean(), 4),
            'mean_rate_di_lo':  round(grp['rate_di_lo'].mean(), 4),
            'mean_rate_di_hi':  round(grp['rate_di_hi'].mean(), 4),
            'pct_change_ci':    round(grp['pct_change_ci'].mean(), 1),
            'pct_change_di':    round(grp['pct_change_di'].mean(), 1),
        })

summary_df = pd.DataFrame(summary_rows)
summ_path = os.path.join(script_dir, 'projections_30yr_summary.csv')
summary_df.to_csv(summ_path, index=False)
print(f"  Saved: projections_30yr_summary.csv ({len(summary_df)} rows)")

# Scenario spread (max-min across all model-scenario combos) per decade
projections['decade'] = (projections['year']//10)*10
spread_rows = []
for (decade, mat, rate_col, hist_rate) in [
    *[(d,'CI','rate_ci',hist_ci_rate) for d in projections['decade'].unique()],
    *[(d,'DI','rate_di',hist_di_rate) for d in projections['decade'].unique()],
]:
    sub = projections[projections['decade']==decade]
    grp = sub.groupby(['model','scenario'])[rate_col].mean()
    spread_rows.append({
        'decade':    decade,
        'material':  mat,
        'min_rate':  round(grp.min(), 4),
        'max_rate':  round(grp.max(), 4),
        'spread':    round(grp.max()-grp.min(), 4),
        'hist_rate': round(hist_rate, 4),
    })

spread_df = pd.DataFrame(spread_rows).sort_values(['material','decade'])
spread_path = os.path.join(script_dir, 'projections_scenario_spread.csv')
spread_df.to_csv(spread_path, index=False)
print(f"  Saved: projections_scenario_spread.csv")


# ── 6. PRINT SUMMARY TABLE ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Results summary")
print("=" * 60)

print(f"\n  Historical baseline (1997-2025):")
print(f"    CI: {hist_ci_rate:.4f} breaks/km/year")
print(f"    DI: {hist_di_rate:.4f} breaks/km/year")

print(f"\n  30-year window mean rates (breaks/km/year):")
print(f"\n  {'Window':<12} {'Model':<10} {'Scenario':<10} "
      f"{'CI rate':>9} {'CI chg%':>8} {'DI rate':>9} {'DI chg%':>8}")
print("  " + "-"*68)

for _, row in summary_df.sort_values(['window','model','scenario']).iterrows():
    print(f"  {row['window']:<12} {row['model']:<10} {row['scenario']:<10} "
          f"{row['mean_rate_ci']:>9.4f} {row['pct_change_ci']:>7.1f}% "
          f"{row['mean_rate_di']:>9.4f} {row['pct_change_di']:>7.1f}%")

print(f"\n  Scenario spread (max-min across all combos):")
print(f"  {'Decade':<8} {'CI spread':>10} {'DI spread':>10}")
for decade in sorted(projections['decade'].unique()):
    ci_sp = spread_df[(spread_df['decade']==decade)&(spread_df['material']=='CI')]['spread'].values
    di_sp = spread_df[(spread_df['decade']==decade)&(spread_df['material']=='DI')]['spread'].values
    if len(ci_sp) and len(di_sp):
        print(f"  {decade:<8} {ci_sp[0]:>10.4f} {di_sp[0]:>10.4f}")

print("\n  FTC WARNING:")
print("  CanESM5 projects 13-14 FTC/month in January vs observed 8.5.")
print("  DI projections using FTC features may be inflated.")
print("  Flag this as a limitation in the paper.")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("""
Output files:
  projections_annual.csv         -- year-by-year rates for all 8 scenarios
  projections_30yr_summary.csv   -- 30-year window means (2026-55, 2041-70, 2071-2100)
  projections_scenario_spread.csv-- scenario uncertainty spread per decade

Next steps:
  1. Plot projected rates vs historical for CI and DI (Figure 3)
  2. Run attribution analysis (counterfactuals A, B, C, D)
  3. Partition uncertainty by GCM vs scenario vs model
""")
