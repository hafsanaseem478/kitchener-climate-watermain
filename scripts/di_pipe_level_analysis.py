
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
warnings.filterwarnings('ignore')

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


# Input data lives in the repository's data directory.
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

# 1. LOAD DATA 

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

b = pd.read_csv(os.path.join(data_dir, 'Water_Main_Breaks.csv'),
                encoding='utf-8-sig', low_memory=False)
m = pd.read_csv(os.path.join(data_dir, 'Water_Mains.csv'),
                encoding='utf-8-sig', low_memory=False)

b['dt'] = pd.to_datetime(b['Incident date'], errors='coerce')
b['yr'] = b['dt'].dt.year
m['iy'] = pd.to_datetime(m['INSTALLATION_DATE'], errors='coerce').dt.year

bm = b[(b['Type of Asset Broken']=='MAIN') &
       (b.yr >= 1997) & (b.yr <= 2025) &
       (b['Asset Material'] == 'DI')].copy()

mm = m[(m['MATERIAL'] == 'DI') &
       (m['Shape__Length'] > 0) &
       (m['iy'].notna())].copy()

print(f"  DI breaks (1997-2025): {len(bm):,}")
print(f"  DI pipes in inventory: {len(mm):,}")


#  2. BUILD PIPE-YEAR PANEL 

print("\n" + "=" * 60)
print("STEP 2: Building pipe-year panel")
print("=" * 60)
print("  Each row = one DI pipe x one year")

# Sort breaks to compute prior break count
bm_s = bm.sort_values(['Related Asset ID', 'dt']).copy()
bm_s['aid'] = pd.to_numeric(bm_s['Related Asset ID'], errors='coerce')
bm_s['prior_breaks'] = bm_s.groupby('aid').cumcount()

# Link breaks to inventory
linked = bm_s[bm_s['aid'].isin(mm['WATMAINID'])].merge(
    mm[['WATMAINID', 'iy', 'Shape__Length']],
    left_on='aid', right_on='WATMAINID', how='left')
linked['age'] = linked['yr'] - linked['iy']
linked = linked[(linked['age'] >= 0) & (linked['age'] <= 120)].copy()

# Break counts and prior breaks per pipe per year
brk_by_pipe_yr = linked.groupby(['WATMAINID', 'yr']).size().reset_index(name='breaks')
prior_by_pipe_yr = (linked.sort_values(['WATMAINID', 'yr'])
                    .groupby(['WATMAINID', 'yr'])['prior_breaks']
                    .min().reset_index())

# Build full pipe-year panel
rows = []
for _, pipe in mm.iterrows():
    pid = pipe['WATMAINID']
    iy = int(pipe['iy'])
    length_km = pipe['Shape__Length'] / 1000.0

    for yr in range(max(1997, iy), 2026):
        age = yr - iy
        b_row = brk_by_pipe_yr[
            (brk_by_pipe_yr['WATMAINID'] == pid) &
            (brk_by_pipe_yr['yr'] == yr)]
        n_breaks = int(b_row['breaks'].values[0]) if len(b_row) else 0
        pb_row = prior_by_pipe_yr[
            (prior_by_pipe_yr['WATMAINID'] == pid) &
            (prior_by_pipe_yr['yr'] < yr)]
        prior_n = int(pb_row['prior_breaks'].max() + 1) if len(pb_row) else 0
        rows.append({'pipe_id': pid, 'year': yr, 'age': age,
                     'prior_breaks': prior_n, 'breaks': n_breaks,
                     'length_km': length_km})

panel = pd.DataFrame(rows)
print(f"  Panel rows: {len(panel):,}")
print(f"  Total DI breaks in panel: {panel['breaks'].sum()}")
print(f"  Age range: {panel['age'].min()} to {panel['age'].max()} years")
print(f"  Zero-break rows: {100*(panel['breaks']==0).mean():.1f}%")


#  3. FIT MODEL 

print("\n" + "=" * 60)
print("STEP 3: Fitting negative binomial pipe-level model")
print("=" * 60)
print("  breaks ~ age + age² + prior_breaks")
print("  offset = log(pipe length in km)")

panel_fit = panel[panel['length_km'] > 0].copy()
panel_fit['log_length'] = np.log(panel_fit['length_km'])
panel_fit['prior_breaks_cap'] = panel_fit['prior_breaks'].clip(upper=5)
panel_fit['age_sq'] = panel_fit['age'] ** 2 / 1000

X = sm.add_constant(panel_fit[['age', 'age_sq', 'prior_breaks_cap']])
y = panel_fit['breaks']
offset = panel_fit['log_length']

glm = GLM(
    y,
    X,
    family=families.NegativeBinomial(alpha=1.0),
    offset=offset
)

model = glm.fit(
    maxiter=200,
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": panel_fit["pipe_id"]}
)

print(f"  AIC: {model.aic:.1f}")
print(f"  Deviance: {model.deviance:.1f}")
print(f"\n  Coefficients:")
for feat, coef, se, pval in zip(model.params.index,
                                  model.params.values,
                                  model.bse.values,
                                  model.pvalues.values):
    sig = '✓' if pval < 0.05 else ' '
    print(f"    {sig} {feat:22s}  coef={coef:+.5f}  se={se:.5f}  p={pval:.4f}")


# ── 4. INTERPRET COEFFICIENTS ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Practical interpretation")
print("=" * 60)

c_const = model.params['const']
c_age   = model.params['age']
c_agesq = model.params['age_sq']
c_prior = model.params['prior_breaks_cap']

# Rate multipliers at various ages
print("\n  Rate multiplier vs a 30-year-old pipe (with 0 prior breaks):")
baseline_30 = c_const + c_age*30 + c_agesq*30**2/1000
for age in [30, 40, 50, 60, 70]:
    log_r = c_const + c_age*age + c_agesq*age**2/1000
    mult = np.exp(log_r - baseline_30)
    print(f"    Age {age}: rate multiplier = {mult:.3f}x")

# Prior breaks effect
print("\n  Effect of prior breaks (holding age constant):")
for pb in range(6):
    mult = np.exp(c_prior * pb)
    print(f"    {pb} prior breaks: rate multiplier = {mult:.3f}x")


# ── 5. SAVE OUTPUTS ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Saving outputs")
print("=" * 60)

# Coefficients table
coef_df = pd.DataFrame({
    'feature':      model.params.index,
    'coefficient':  model.params.values.round(5),
    'std_error':    model.bse.values.round(5),
    'z_score':      model.tvalues.values.round(3),
    'p_value':      model.pvalues.values.round(4),
    'significant':  model.pvalues.values < 0.05,
})
coef_path = os.path.join(script_dir, 'di_pipe_level_coefficients.csv')
coef_df.to_csv(coef_path, index=False)
print(f"  Saved: di_pipe_level_coefficients.csv")

# Summary of key findings
summary_rows = [
    {'finding': 'Prior break multiplier per additional break',
     'value':   round(np.exp(c_prior), 3),
     'note':    'Each prior break multiplies future break rate by this factor'},
    {'finding': 'Age effect at mean age (32 yrs) per decade',
     'value':   round(np.exp((c_age + 2*c_agesq*32/1000)*10), 3),
     'note':    'Rate multiplier per extra decade of age at current mean age'},
    {'finding': 'Age at which rate is maximum',
     'value':   round(-c_age * 1000 / (2*c_agesq), 1),
     'note':    'Peak of the age-break curve (in years)'},
    {'finding': 'Total pipe-years in panel',
     'value':   len(panel_fit),
     'note':    'Sample size for the model'},
    {'finding': 'Total DI breaks in panel',
     'value':   panel_fit['breaks'].sum(),
     'note':    'Total events observed'},
    {'finding': 'Model AIC',
     'value':   round(model.aic, 1),
     'note':    'Lower is better'},
]
summary_df = pd.DataFrame(summary_rows)
summ_path = os.path.join(script_dir, 'di_pipe_level_summary.csv')
summary_df.to_csv(summ_path, index=False)
print(f"  Saved: di_pipe_level_summary.csv")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

