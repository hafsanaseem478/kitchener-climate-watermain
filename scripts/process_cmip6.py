"""
process_cmip6.py
================
Reads all 24 CMIP6 NetCDF files (2 models x 4 scenarios x 3 variables),
calculates the same monthly climate covariates as the historical data,
and saves one CSV per model-scenario combination.

WHAT WE KNOW FROM FILE INSPECTION:
  - Units: degC (already Celsius, no conversion needed)
  - Single point: lat=43.46N lon=-80.54W (Kitchener)
  - Time: days since 1950-01-01, gregorian calendar
  - Shape: (29585, 1, 1) — one value per day
  - Missing value: 32767 — masked automatically
  - Variable names: tasmax, tasmin, pr

OUTPUT FILES (one per model-scenario):
  cmip6_monthly_CanESM5_ssp126.csv
  cmip6_monthly_CanESM5_ssp245.csv
  cmip6_monthly_CanESM5_ssp370.csv
  cmip6_monthly_CanESM5_ssp585.csv
  cmip6_monthly_MIROC6_ssp126.csv
  cmip6_monthly_MIROC6_ssp245.csv
  cmip6_monthly_MIROC6_ssp370.csv
  cmip6_monthly_MIROC6_ssp585.csv

COVARIATES PRODUCED (identical to historical climate_monthly.csv):
  Tmean, Tmin, Tmax, ADD, FI, FD, FTC, FI_cum, TI, TD, TIG, TDG

Run:
  pip install netCDF4 numpy pandas
  python process_cmip6.py

Put all .nc files in the same folder as this script.
"""

import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 0. SETTINGS ───────────────────────────────────────────────────────────────

# Projection period — historical overlap excluded
PROJ_START_YEAR = 2026
PROJ_END_YEAR   = 2100

# Historical overlap period — used for bias check
HIST_START_YEAR = 1995
HIST_END_YEAR   = 2014

MISSING_VALUE = 32767.0

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# ── 1. FIND ALL NC FILES ──────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Finding NC files")
print("=" * 60)

nc_files = glob.glob(os.path.join(script_dir, "*.nc"))

if not nc_files:
    raise FileNotFoundError(
        "No .nc files found in current folder.\n"
        "Put all CMIP6 .nc files in the same folder as this script."
    )

print(f"  Found {len(nc_files)} NC files:")
for f in sorted(nc_files):
    print(f"    {os.path.basename(f)}")


# ── 2. PARSE FILENAMES INTO GROUPS ───────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Grouping files by model and scenario")
print("=" * 60)

# File naming pattern (underscores replacing + and spaces):
# tasmax_day_MBCn_PCIC-Blend_MIROC6_historical_ssp370_r1i1p1f1_gn_...nc
# We need to extract: variable (tasmax/tasmin/pr), model (CanESM5/MIROC6),
# scenario (ssp126/ssp245/ssp370/ssp585)

groups = {}  # key = (model, scenario), value = {var: filepath}

for f in nc_files:
    base = os.path.basename(f).lower()

    # Variable
    if base.startswith('tasmax'):
        var = 'tasmax'
    elif base.startswith('tasmin'):
        var = 'tasmin'
    elif base.startswith('pr'):
        var = 'pr'
    else:
        print(f"  WARNING: Unknown variable in {base}, skipping")
        continue

    # Model
    if 'canesm5' in base:
        model = 'CanESM5'
    elif 'miroc6' in base:
        model = 'MIROC6'
    else:
        print(f"  WARNING: Unknown model in {base}, skipping")
        continue

    # Scenario
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        if ssp in base:
            scenario = ssp
            break
    else:
        print(f"  WARNING: Unknown scenario in {base}, skipping")
        continue

    key = (model, scenario)
    if key not in groups:
        groups[key] = {}
    groups[key][var] = f

print(f"  Groups found: {len(groups)}")
for (model, scenario), vars_dict in sorted(groups.items()):
    vars_present = sorted(vars_dict.keys())
    status = "✓ complete" if len(vars_present) == 3 else f"INCOMPLETE: {vars_present}"
    print(f"    {model} {scenario}: {status}")


# ── 3. HELPER: READ NC FILE ───────────────────────────────────────────────────

def read_nc_timeseries(filepath, varname):
    """
    Read a PCIC CanDCS-M6 NetCDF file.
    Returns a DataFrame with columns: date, <varname>
    Handles missing values (32767) and flattens the (time,1,1) shape.
    Auto-detects the actual variable name in the file in case it differs.
    """
    ds = nc.Dataset(filepath)

    # Auto-detect the climate variable (not time/lat/lon)
    coord_vars = {'time', 'lat', 'lon', 'latitude', 'longitude'}
    data_vars = [v for v in ds.variables if v not in coord_vars]
    actual_varname = data_vars[0] if data_vars else varname

    # Time: days since 1950-01-01
    time_var  = ds.variables['time']
    time_vals = time_var[:].data
    base_date = datetime(1950, 1, 1)
    dates = [base_date + timedelta(days=float(int(t))) for t in time_vals]

    # Variable data — shape (time, 1, 1)
    data = ds.variables[actual_varname][:].data.flatten()

    # Mask missing values
    data = np.where(np.abs(data - MISSING_VALUE) < 1, np.nan, data)

    ds.close()

    # Always return with the requested varname as column
    df = pd.DataFrame({'date': dates, varname: data})
    df['date'] = pd.to_datetime(df['date'])
    return df


# ── 4. HELPER: CALCULATE MONTHLY COVARIATES ──────────────────────────────────

def calc_gradients(tmean_series):
    """TIG and TDG: max rate of temperature change between consecutive days."""
    t = tmean_series.dropna().values
    if len(t) < 2:
        return np.nan, np.nan
    diffs = np.diff(t)
    return float(np.max(diffs)), float(np.max(-diffs))


def calc_fi_cumulative(daily_df, year, month):
    """
    Cumulative freezing index since start of freeze season (Oct previous year).
    """
    if month < 10:
        season_start = pd.Timestamp(year=year-1, month=10, day=1)
    else:
        season_start = pd.Timestamp(year=year, month=10, day=1)
    season_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    mask = (daily_df['date'] >= season_start) & (daily_df['date'] <= season_end)
    fi_cum = daily_df.loc[mask, 'tasmin'].apply(
        lambda x: min(x, 0) if pd.notna(x) else 0
    ).sum()
    return round(fi_cum, 2)


def build_monthly_covariates(daily_df, year_start, year_end):
    """
    Given a daily DataFrame with columns: date, tasmax, tasmin, pr
    Returns monthly covariate DataFrame for year_start to year_end.
    """
    # Filter to requested period
    df = daily_df[
        (daily_df['date'].dt.year >= year_start) &
        (daily_df['date'].dt.year <= year_end)
    ].copy()

    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Tmean approximation from tasmax and tasmin
    # PCIC files provide tasmax and tasmin but not tmean directly
    # Standard approximation: Tmean = (Tmax + Tmin) / 2
    df['Tmean'] = (df['tasmax'] + df['tasmin']) / 2

    monthly_rows = []

    for (yr, mo), grp in df.groupby(['year', 'month']):
        grp = grp.copy().reset_index(drop=True)

        # Temperature level
        Tmean = grp['Tmean'].mean()
        Tmin  = grp['tasmin'].mean()
        Tmax  = grp['tasmax'].mean()
        ADD   = (grp['tasmax'] - grp['tasmin']).mean()

        # Frost
        FI  = grp['Tmean'].apply(lambda x: min(x,0) if pd.notna(x) else 0).sum()
        FD  = int((grp['Tmean'] < 0).sum())
        FTC = int(((grp['tasmin'] < 0) & (grp['tasmax'] > 0)).sum())

        # Thaw
        TI  = grp['Tmean'].apply(lambda x: max(x,0) if pd.notna(x) else 0).sum()
        TD  = int((grp['Tmean'] > 0).sum())

        # Gradients
        TIG, TDG = calc_gradients(grp['Tmean'])

        # Cumulative FI since freeze season start
        FI_cum = calc_fi_cumulative(df, yr, mo)

        monthly_rows.append({
            'year':     yr,
            'month':    mo,
            'month_str': f"{yr}-{mo:02d}",
            'Tmean':    round(Tmean, 3) if pd.notna(Tmean) else np.nan,
            'Tmin':     round(Tmin,  3) if pd.notna(Tmin)  else np.nan,
            'Tmax':     round(Tmax,  3) if pd.notna(Tmax)  else np.nan,
            'ADD':      round(ADD,   3) if pd.notna(ADD)    else np.nan,
            'FI':       round(FI,    2),
            'FD':       FD,
            'FTC':      FTC,
            'FI_cum':   FI_cum,
            'TI':       round(TI, 2),
            'TD':       TD,
            'TIG':      round(TIG, 3) if pd.notna(TIG) else np.nan,
            'TDG':      round(TDG, 3) if pd.notna(TDG) else np.nan,
        })

    return pd.DataFrame(monthly_rows).sort_values(['year','month']).reset_index(drop=True)


# ── 5. PROCESS EACH GROUP ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Processing each model-scenario combination")
print("=" * 60)

all_summaries = []

for (model, scenario), vars_dict in sorted(groups.items()):

    print(f"\n  --- {model} {scenario} ---")

    # Check all three variables present
    missing_vars = [v for v in ['tasmax','tasmin','pr'] if v not in vars_dict]
    if missing_vars:
        print(f"  SKIPPING — missing variables: {missing_vars}")
        continue

    # Read all three variables
    print(f"  Reading tasmax...")
    df_max  = read_nc_timeseries(vars_dict['tasmax'], 'tasmax')
    print(f"  Reading tasmin...")
    df_min  = read_nc_timeseries(vars_dict['tasmin'], 'tasmin')
    print(f"  Reading pr...")
    df_pr   = read_nc_timeseries(vars_dict['pr'], 'pr')

    # Merge on date
    daily = df_max.merge(df_min, on='date').merge(df_pr, on='date')
    print(f"  Daily rows: {len(daily)} | date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # Check for missing values
    for col in ['tasmax','tasmin','pr']:
        n_miss = daily[col].isna().sum()
        if n_miss > 0:
            print(f"  WARNING: {col} has {n_miss} missing values")

    # Build monthly covariates for projection period
    print(f"  Calculating monthly covariates ({PROJ_START_YEAR}-{PROJ_END_YEAR})...")
    monthly_proj = build_monthly_covariates(daily, PROJ_START_YEAR, PROJ_END_YEAR)
    monthly_proj['model']    = model
    monthly_proj['scenario'] = scenario

    expected_months = (PROJ_END_YEAR - PROJ_START_YEAR + 1) * 12
    print(f"  Monthly rows: {len(monthly_proj)} (expect {expected_months})")

    # Build monthly covariates for historical overlap (for bias check)
    # Note: some files may start after HIST_END_YEAR — handle gracefully
    daily_min_yr = daily['date'].dt.year.min()
    if daily_min_yr <= HIST_END_YEAR:
        monthly_hist = build_monthly_covariates(daily, HIST_START_YEAR, HIST_END_YEAR)
        monthly_hist['model']    = model
        monthly_hist['scenario'] = scenario
        out_hist = os.path.join(script_dir, f"cmip6_hist_{model}_{scenario}.csv")
        monthly_hist.to_csv(out_hist, index=False)
    else:
        print(f"  Note: file starts in {daily_min_yr}, no historical overlap available for bias check")

    # Quick sanity check on projection data
    jan_fi = monthly_proj[monthly_proj['month']==1]['FI'].mean()
    jul_tmp = monthly_proj[monthly_proj['month']==7]['Tmean'].mean()
    print(f"  Sanity: Jan mean FI={jan_fi:.1f} (expect negative), Jul Tmean={jul_tmp:.1f} (expect warm)")

    # Save projection file
    out_proj = os.path.join(script_dir, f"cmip6_monthly_{model}_{scenario}.csv")
    monthly_proj.to_csv(out_proj, index=False)
    print(f"  Saved: {os.path.basename(out_proj)}")

    # (historical overlap file saved inside the if-block above)

    # Summary stats
    all_summaries.append({
        'model': model,
        'scenario': scenario,
        'proj_months': len(monthly_proj),
        'jan_fi_mean': round(jan_fi, 1),
        'jul_tmean': round(jul_tmp, 1),
        'jan_ftc_mean': round(monthly_proj[monthly_proj['month']==1]['FTC'].mean(), 1),
    })


# ── 6. BIAS CHECK ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Bias check — GCM historical vs observed (1995-2014)")
print("=" * 60)
print("  Comparing GCM historical run against ECCC observed climate.")
print("  Large bias means the model systematically over/underpredicts.")
print("  For FTC especially — threshold crossings are sensitive to bias.\n")

# Load historical observed monthly data
obs_path = os.path.join(script_dir, 'climate_monthly.csv')
if os.path.exists(obs_path):
    obs = pd.read_csv(obs_path)
    obs_hist = obs[(obs['year']>=HIST_START_YEAR) & (obs['year']<=HIST_END_YEAR)]
    obs_jan = obs_hist[obs_hist['month']==1]

    print(f"  Observed (ECCC) 1995-2014:")
    print(f"    Jan Tmean mean: {obs_jan['Tmean'].mean():.1f}°C")
    print(f"    Jan FI mean:    {obs_jan['FI'].mean():.1f} deg-C-days")
    print(f"    Jan FTC mean:   {obs_jan['FTC'].mean():.1f} cycles")
    print()

    # Compare each GCM
    hist_files = glob.glob(os.path.join(script_dir, "cmip6_hist_*.csv"))
    for hf in sorted(hist_files):
        gcm = pd.read_csv(hf)
        gcm_jan = gcm[gcm['month']==1]
        model   = gcm['model'].iloc[0]
        scenario= gcm['scenario'].iloc[0]
        bias_tmean = gcm_jan['Tmean'].mean() - obs_jan['Tmean'].mean()
        bias_fi    = gcm_jan['FI'].mean()    - obs_jan['FI'].mean()
        bias_ftc   = gcm_jan['FTC'].mean()   - obs_jan['FTC'].mean()
        print(f"  {model} {scenario} January bias:")
        print(f"    Tmean: {bias_tmean:+.2f}°C  FI: {bias_fi:+.1f}  FTC: {bias_ftc:+.1f}")
        flag = " ← FLAG" if abs(bias_ftc) > 3 else ""
        print(f"    FTC bias{flag}")
else:
    print("  climate_monthly.csv not found — skipping bias check.")
    print("  Put climate_monthly.csv in the same folder to enable this check.")


# ── 7. SUMMARY ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Summary")
print("=" * 60)

summary_df = pd.DataFrame(all_summaries)
if len(summary_df) > 0:
    print(summary_df.to_string(index=False))

print("\n  Output files:")
out_files = glob.glob(os.path.join(script_dir, "cmip6_monthly_*.csv"))
for f in sorted(out_files):
    df = pd.read_csv(f)
    print(f"    {os.path.basename(f)}: {len(df)} rows")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("""
Next steps:
  1. Check the bias report above — if FTC bias > 3 cycles/month, flag in paper
  2. Load the cmip6_monthly_*.csv files in the projection script
  3. Add lags to projected covariates (same as modelling_dataset.csv)
  4. Feed into the fitted negative binomial model to get projected break rates
  5. Run attribution scenarios A, B, C, D
""")
