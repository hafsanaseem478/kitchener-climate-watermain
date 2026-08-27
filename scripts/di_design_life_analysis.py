"""
di_design_life_analysis.py
==========================
DESIGN LIFE ANALYSIS for DI pipes — 2025 to 2050 horizon.

Question:
  Does climate change cause DI pipes to fail before their 75-year design life?

Method (survival simulation, pipe by pipe):
  1. For each DI pipe currently in service (2025):
     - Compute its remaining life = min(75 - current age, 25 years to 2050)
  2. Under each climate scenario:
     - Get projected annual break rate for the DI cohort in each year
     - Adjust for the individual pipe's condition (age, prior breaks)
     - Compute probability of first failure each year
     - Simulate whether pipe fails before reaching design life or 2050
  3. Aggregate: what % of pipes fail before design life under each scenario?
  4. Report: average years of service life lost per failing pipe

INPUTS:
  Water_Mains.csv
  Water_Main_Breaks.csv
  projections_annual.csv          (from project_breaks.py)
  di_pipe_level_coefficients.csv  (from di_pipe_level_analysis.py)

OUTPUTS:
  design_life_results.csv           -- per-scenario summary
  design_life_pipe_predictions.csv  -- per-pipe failure predictions

Run:
  python di_design_life_analysis.py

Note: DI = 75-year design life
      Horizon = 2050 (25 years from 2025)
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# ── 0. SETTINGS ───────────────────────────────────────────────────────────────

DESIGN_LIFE_DI = 75      # years
HORIZON_YEAR   = 2050    # cut-off for analysis
START_YEAR     = 2025
DI_KM_2025     = 322.05  # total DI km in service in 2025 (from earlier work)

# Historical baseline break rate (breaks/km/year) — for reference scenario
HIST_DI_RATE = 0.0663

# Analytical survival calculation — no Monte Carlo needed


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

m    = pd.read_csv(os.path.join(script_dir, 'Water_Mains.csv'),
                   encoding='utf-8-sig', low_memory=False)
b    = pd.read_csv(os.path.join(script_dir, 'Water_Main_Breaks.csv'),
                   encoding='utf-8-sig', low_memory=False)
proj = pd.read_csv(os.path.join(script_dir, 'projections_annual.csv'))
coef = pd.read_csv(os.path.join(script_dir, 'di_pipe_level_coefficients.csv'))

m['iy'] = pd.to_datetime(m['INSTALLATION_DATE'], errors='coerce').dt.year
b['dt'] = pd.to_datetime(b['Incident date'], errors='coerce')
b['yr'] = b['dt'].dt.year

# Filter to active DI pipes
di = m[(m['MATERIAL'] == 'DI') &
       (m['Shape__Length'] > 0) &
       (m['iy'].notna())].copy()

print(f"  Active DI pipes: {len(di):,}")
print(f"  Total DI length: {di['Shape__Length'].sum()/1000:.1f} km")


# ── 2. COMPUTE PER-PIPE ATTRIBUTES ───────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Computing per-pipe attributes")
print("=" * 60)

di['age_2025']   = START_YEAR - di['iy']
di['dl_year']    = di['iy'] + DESIGN_LIFE_DI
di['length_km'] = di['Shape__Length'] / 1000.0

# Remaining years to design life, capped at horizon
di['years_to_dl'] = di['dl_year'] - START_YEAR
di['years_to_horizon'] = HORIZON_YEAR - START_YEAR
di['sim_years'] = np.minimum(di['years_to_dl'], di['years_to_horizon']).clip(lower=0)

# Count prior breaks for each pipe (from 1997-2025 history)
bm = b[(b['Type of Asset Broken']=='MAIN') &
       (b.yr >= 1997) & (b.yr <= 2025) &
       (b['Asset Material'] == 'DI')].copy()
bm['aid'] = pd.to_numeric(bm['Related Asset ID'], errors='coerce')
prior_counts = bm.groupby('aid').size().reset_index(name='prior_breaks_2025')
di = di.merge(prior_counts, left_on='WATMAINID', right_on='aid', how='left')
di['prior_breaks_2025'] = di['prior_breaks_2025'].fillna(0).astype(int)

# Categorize pipes
already_past = (di['dl_year'] <= START_YEAR).sum()
reaches_in_window = ((di['dl_year'] > START_YEAR) & (di['dl_year'] <= HORIZON_YEAR)).sum()
still_within = (di['dl_year'] > HORIZON_YEAR).sum()

print(f"  Already past design life in 2025: {already_past} pipes")
print(f"  Reaches design life 2025-2050:    {reaches_in_window} pipes")
print(f"  Still within design life in 2050: {still_within} pipes")

# Exclude pipes already past DL (they cannot 'fail before DL' by definition)
di_sim = di[di['dl_year'] > START_YEAR].copy()
print(f"\n  Pipes included in simulation: {len(di_sim):,}")


# ── 3. LOAD COEFFICIENTS FROM SCRIPT 1 ───────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Loading pipe-level coefficients")
print("=" * 60)

# Get the age/prior_breaks adjustment coefficients from Script 1 output
c_const = coef[coef['feature']=='const']['coefficient'].values[0]
c_age   = coef[coef['feature']=='age']['coefficient'].values[0]
c_agesq = coef[coef['feature']=='age_sq']['coefficient'].values[0]
c_prior = coef[coef['feature']=='prior_breaks_cap']['coefficient'].values[0]

print(f"  const:            {c_const:+.5f}")
print(f"  age:              {c_age:+.5f}")
print(f"  age_sq:           {c_agesq:+.5f}")
print(f"  prior_breaks_cap: {c_prior:+.5f}")


def pipe_baseline_rate(age, prior_breaks):
    """
    Baseline rate per km per year from pipe-level model.
    (Before applying climate multiplier.)
    """
    log_rate = (c_const +
                c_age    * age +
                c_agesq  * age**2 / 1000 +
                c_prior  * min(prior_breaks, 5))
    return np.exp(log_rate)


# Reference: mean baseline rate at 2025 across all pipes
di_sim['baseline_rate_2025'] = di_sim.apply(
    lambda r: pipe_baseline_rate(r['age_2025'], r['prior_breaks_2025']), axis=1)
mean_baseline_2025 = di_sim['baseline_rate_2025'].mean()
print(f"\n  Mean pipe-level baseline rate 2025: {mean_baseline_2025:.5f} breaks/km/yr")


# ── 4. RUN SIMULATION FOR EACH SCENARIO ──────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Simulating failures under each climate scenario")
print("=" * 60)
print(f"  Method: analytical survival calculation per pipe")
print(f"          For each year: hazard = 1 - exp(-rate * length)")
print(f"          Survival S(t) = product of (1 - hazard) up to year t")
print(f"          Expected P(fail before DL) = sum of P(first fail in year k)")

# Historical baseline scenario: use HIST_DI_RATE for every year
# Then each SSP scenario uses that year's projected rate

# For projected scenarios, we need year-by-year rate 2026-2050
scenarios = ['historical'] + sorted(
    [(m, s) for m, s in proj[['model','scenario']].drop_duplicates().values.tolist()],
    key=lambda x: (x[0], x[1])
)

results = []
pipe_predictions = []

for scen in scenarios:
    if scen == 'historical':
        scenario_label = 'historical'
        scenario_rates = {yr: HIST_DI_RATE for yr in range(2026, HORIZON_YEAR + 1)}
    else:
        model_gcm, ssp = scen
        scenario_label = f"{model_gcm}_{ssp}"
        # Get projected DI rate per year
        s = proj[(proj['model']==model_gcm) & (proj['scenario']==ssp)]
        scenario_rates = dict(zip(s['year'], s['rate_di']))

    print(f"\n  Scenario: {scenario_label}")

    # Climate multiplier for each year vs 2025 baseline
    # (Rate divided by historical baseline gives the climate factor)
    climate_mult = {yr: scenario_rates.get(yr, HIST_DI_RATE) / HIST_DI_RATE
                    for yr in range(2026, HORIZON_YEAR + 1)}

    scen_pipe_results = []

    for _, pipe in di_sim.iterrows():
        pid       = pipe['WATMAINID']
        age_now   = int(pipe['age_2025'])
        prior_now = int(pipe['prior_breaks_2025'])
        sim_years = int(pipe['sim_years'])
        dl_year   = int(pipe['dl_year'])

        if sim_years <= 0:
            continue

        # ANALYTICAL survival calculation (no Monte Carlo loop needed)
        # For each year k, compute hazard rate and survival probability
        # P(first fail in year yr) = S(yr-1) * (1 - exp(-rate_yr * length))

        years_arr    = np.arange(1, sim_years + 1)
        yrs_calendar = START_YEAR + years_arr           # 2026, 2027...
        ages_arr     = age_now + years_arr

        # Pipe-level base rates for each future year
        log_rates = (c_const +
                     c_age    * ages_arr +
                     c_agesq  * ages_arr**2 / 1000 +
                     c_prior  * min(prior_now, 5))
        base_rates = np.exp(log_rates)

        # Climate multipliers
        clim_arr = np.array([climate_mult.get(int(y), 1.0) for y in yrs_calendar])
        rates_arr = base_rates * clim_arr

        # Hazard per year (probability of failure in year k given survival to k-1)
        hazard = 1 - np.exp(-rates_arr * pipe['length_km'])

        # Survival function
        surv = np.concatenate([[1.0], np.cumprod(1 - hazard)])   # S(0)=1
        # P(first failure in year k) = S(k-1) * hazard(k)
        p_fail_each_year = surv[:-1] * hazard

        # Expected probability of failing before design life
        mask_before_dl = yrs_calendar <= dl_year
        prob_fail_before_dl = p_fail_each_year[mask_before_dl].sum()
        prob_fail_by_horizon = p_fail_each_year.sum()

        # Expected failure year (among those that fail)
        if prob_fail_by_horizon > 0:
            mean_failure_year = float(np.sum(yrs_calendar * p_fail_each_year) /
                                       prob_fail_by_horizon)
            years_lost = dl_year - mean_failure_year
        else:
            mean_failure_year = np.nan
            years_lost = np.nan

        scen_pipe_results.append({
            'scenario':       scenario_label,
            'pipe_id':        pid,
            'age_2025':       age_now,
            'dl_year':        dl_year,
            'prior_breaks':   prior_now,
            'length_km':      round(pipe['length_km'], 4),
            'p_fail_before_dl':   round(prob_fail_before_dl, 3),
            'p_fail_by_2050':     round(prob_fail_by_horizon, 3),
            'mean_failure_year':  round(mean_failure_year, 1) if not np.isnan(mean_failure_year) else np.nan,
            'years_lost':         round(years_lost, 1) if not np.isnan(years_lost) else np.nan,
        })

    scen_df = pd.DataFrame(scen_pipe_results)
    pipe_predictions.append(scen_df)

    # Aggregate for this scenario
    total_pipes = len(scen_df)
    # Expected % pipes failing before DL (probability-weighted)
    pct_fail_before_dl = scen_df['p_fail_before_dl'].mean() * 100
    pct_fail_by_2050   = scen_df['p_fail_by_2050'].mean() * 100
    mean_years_lost    = scen_df['years_lost'].mean()
    total_km_at_risk   = (scen_df['p_fail_before_dl'] * scen_df['length_km']).sum()

    print(f"    Expected % of DI pipes failing before design life: {pct_fail_before_dl:.1f}%")
    print(f"    Expected % of DI pipes failing by 2050:            {pct_fail_by_2050:.1f}%")
    print(f"    Mean years of service life lost (failing pipes):   {mean_years_lost:.1f} years")

    results.append({
        'scenario':            scenario_label,
        'total_pipes':         total_pipes,
        'pct_fail_before_dl':  round(pct_fail_before_dl, 2),
        'pct_fail_by_2050':    round(pct_fail_by_2050, 2),
        'mean_years_lost':     round(mean_years_lost, 2),
        'total_km_at_risk':    round(total_km_at_risk, 2),
    })


# ── 5. SAVE OUTPUTS ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Saving outputs")
print("=" * 60)

results_df = pd.DataFrame(results)
res_path = os.path.join(script_dir, 'design_life_results.csv')
results_df.to_csv(res_path, index=False)
print(f"  Saved: design_life_results.csv")

pipe_pred_df = pd.concat(pipe_predictions, ignore_index=True)
pp_path = os.path.join(script_dir, 'design_life_pipe_predictions.csv')
pipe_pred_df.to_csv(pp_path, index=False)
print(f"  Saved: design_life_pipe_predictions.csv ({len(pipe_pred_df)} rows)")


# ── 6. FINAL SUMMARY ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Final summary")
print("=" * 60)
print(f"\n  DI design life: {DESIGN_LIFE_DI} years")
print(f"  Analysis horizon: {START_YEAR} to {HORIZON_YEAR}")
print()
print("  RESULTS BY SCENARIO:")
print(f"  {'Scenario':<25} {'% fail before DL':>18} {'% fail by 2050':>18} {'Yrs lost':>10}")
print("  " + "-"*72)
for _, r in results_df.iterrows():
    print(f"  {r['scenario']:<25} {r['pct_fail_before_dl']:>16.2f}%  "
          f"{r['pct_fail_by_2050']:>16.2f}%  {r['mean_years_lost']:>9.1f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("""
PAPER FRAMING:

  "Under historical climate, an expected [X.X]% of DI pipes are projected
   to experience their first failure before reaching their 75-year design
   life within the 2025-2050 planning horizon. Under SSP5-8.5, this
   fraction changes to [Y.Y]%, with failing pipes losing an average of
   [Z.Z] years of expected service life. This analysis assumes climate
   and pipe-level ageing effects act multiplicatively and independently,
   and treats prior break counts as frozen at 2025 values."

CAVEATS TO STATE:
  - Assumes multiplicative independence of climate and ageing effects
  - Prior break counts frozen at 2025 (no self-excitation dynamics)
  - Right-censored (many DI pipes have not yet reached design life)
  - Uses cohort-level climate projection scaled to pipe level
""")
