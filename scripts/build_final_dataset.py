"""
build_final_dataset.py
======================
Merges the monthly break panel (monthly_panel_primary.csv) with the
monthly climate covariates (climate_monthly.csv) to produce the final
modelling dataset.

PRECIPITATION NOTE:
  We use temperature covariates only. Precipitation is excluded because
  the KITCHENER/WATERLOO station (2010-2025) measures TOTAL_PRECIPITATION
  inconsistently compared to earlier stations, introducing a hidden
  station-change artefact in the precipitation series. Temperature data
  is consistent across all stations for the full 1997-2025 window.

FINAL COVARIATES:
  From panel   : material, break_count, km_primary, failure_rate,
                 sin_month, cos_month, time_index
  From climate : Tmean, Tmin, Tmax, ADD,
                 FI, FD, FTC, FI_cum,
                 TI, TD, TIG, TDG,
                 missing_days (quality flag)

LAGGED COVARIATES ADDED:
  Each temperature covariate at t-1, t-2, t-3 months.
  Frost penetrates to pipe depth weeks after surface air temperature drops,
  so lags are physically motivated and important for CI pipes.

OUTPUT:
  modelling_dataset.csv  -- one row per material x month, 1997-2025
                            1044 rows x ~45 columns

Run:
  python build_final_dataset.py

Requires: pandas, numpy
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 0. SETTINGS ───────────────────────────────────────────────────────────────

PANEL_FILE   = "monthly_panel_primary.csv"
CLIMATE_FILE = "climate_monthly.csv"
OUTPUT_FILE  = "modelling_dataset.csv"

# Temperature covariates to keep (no precipitation)
CLIMATE_COLS = [
    'Tmean', 'Tmin', 'Tmax', 'ADD',
    'FI', 'FD', 'FTC', 'FI_cum',
    'TI', 'TD', 'TIG', 'TDG',
    'missing_days'
]

# Which covariates to also add as lags (t-1, t-2, t-3)
# Physically motivated: frost at pipe depth lags surface temperature
LAG_COLS = ['Tmean', 'FI', 'FTC', 'FI_cum', 'FD', 'ADD']
LAG_MONTHS = [1, 2, 3]


# ── 1. LOAD FILES ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading files")
print("=" * 60)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

panel   = pd.read_csv(os.path.join(script_dir, PANEL_FILE))
climate = pd.read_csv(os.path.join(script_dir, CLIMATE_FILE))

print(f"  Panel   : {panel.shape}  ({PANEL_FILE})")
print(f"  Climate : {climate.shape}  ({CLIMATE_FILE})")


# ── 2. PREPARE CLIMATE FILE FOR MERGE ────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Preparing climate file")
print("=" * 60)

# Keep only the columns we need
climate_keep = ['month_str'] + CLIMATE_COLS
climate_slim = climate[climate_keep].copy()

print(f"  Climate columns kept: {len(climate_keep)}")
print(f"  {climate_keep}")


# ── 3. MERGE ON month_str ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Merging panel with climate")
print("=" * 60)

# Panel covers 1997-2025, climate covers 1996-2025
# Left join: keep all panel rows, match climate where available
# The 1996 climate year is only needed for lag calculation (added below)
merged = panel.merge(climate_slim, on='month_str', how='left')

print(f"  Rows after merge: {len(merged)} (expect 1044)")
print(f"  Columns after merge: {len(merged.columns)}")

# Check all climate rows matched
unmatched = merged['Tmean'].isna().sum()
print(f"  Rows with no climate match: {unmatched} (expect 0)")


# ── 4. ADD LAGGED COVARIATES ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Adding lagged climate covariates")
print("=" * 60)

# To build lags correctly we need the 1996 climate data too
# because 1997-01 lag-1 = 1996-12, lag-2 = 1996-11, lag-3 = 1996-10
climate_full = climate[['month_str'] + LAG_COLS].copy()
climate_full['month_dt'] = pd.to_datetime(climate_full['month_str'] + '-01')
climate_full = climate_full.sort_values('month_dt').reset_index(drop=True)

# Build a lookup: month_str -> lagged values
lag_lookup = climate_full.set_index('month_str')[LAG_COLS]

def get_lag_month_str(month_str, lag):
    """Return the month_str that is 'lag' months before the given month_str."""
    dt = pd.to_datetime(month_str + '-01') - pd.DateOffset(months=lag)
    return dt.strftime('%Y-%m')

# Add lags to merged dataset
for col in LAG_COLS:
    for lag in LAG_MONTHS:
        new_col = f"{col}_lag{lag}"
        merged[new_col] = merged['month_str'].apply(
            lambda ms: lag_lookup.loc[get_lag_month_str(ms, lag), col]
            if get_lag_month_str(ms, lag) in lag_lookup.index else np.nan
        )
        n_null = merged[new_col].isna().sum()
        status = "OK" if n_null == 0 else f"WARNING: {n_null} nulls"
        print(f"  Added {new_col:<20} {status}")


# ── 5. FINAL COLUMN ORDER ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Organising final column order")
print("=" * 60)

# Identity columns
id_cols = ['material', 'month_str', 'year', 'month_num']

# Target variable
target = ['break_count', 'km_primary', 'failure_rate']

# Pipe / network features
pipe_cols = ['sin_month', 'cos_month', 'time_index']

# Current-month climate
climate_current = [c for c in CLIMATE_COLS if c != 'missing_days']

# Lagged climate
lag_col_names = [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_MONTHS]

# Quality flag
quality = ['missing_days']

final_cols = id_cols + target + pipe_cols + climate_current + lag_col_names + quality

# Keep only columns that exist
final_cols = [c for c in final_cols if c in merged.columns]
dataset = merged[final_cols].copy()

print(f"  Final columns: {len(dataset.columns)}")
for col in dataset.columns:
    print(f"    {col}")


# ── 6. SANITY CHECKS ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Sanity checks")
print("=" * 60)

# Check 1: shape
print(f"\n  Shape: {dataset.shape}")
print(f"  Expected rows: 1044 (3 materials x 348 months)")

# Check 2: nulls
print(f"\n  Null counts:")
nulls = dataset.isnull().sum()
null_cols = nulls[nulls > 0]
if len(null_cols) == 0:
    print("    None — all columns complete")
else:
    print(null_cols.to_string())

# Check 3: materials and date range
print(f"\n  Materials: {dataset['material'].unique().tolist()}")
print(f"  Date range: {dataset['month_str'].min()} to {dataset['month_str'].max()}")

# Check 4: failure rate by material
print(f"\n  Mean failure rate (breaks/km/month) by material:")
print(dataset.groupby('material')['failure_rate'].mean().round(5).to_string())

# Check 5: climate signal visible
print(f"\n  Mean FI by material x season (CI should show strong winter signal):")
dataset['winter'] = dataset['month_num'].isin([12,1,2])
print(dataset.groupby(['material','winter'])['FI'].mean().round(1).to_string())
dataset.drop(columns=['winter'], inplace=True)

# Check 6: lag columns look right
print(f"\n  Spot check: Jan 1997 CI row")
jan97 = dataset[(dataset['month_str']=='1997-01')&(dataset['material']=='CI')]
print(jan97[['month_str','Tmean','Tmean_lag1','Tmean_lag2','Tmean_lag3']].to_string(index=False))
print("  (lag1 = Dec 1996, lag2 = Nov 1996, lag3 = Oct 1996 — should be cold)")

# Check 7: time_index
print(f"\n  Time index range: {dataset['time_index'].min()} to {dataset['time_index'].max()}")
print(f"  (expect 0 to 347)")

# Check 8: missing_days flag
months_flagged = dataset[dataset['missing_days']>5]['month_str'].unique()
print(f"\n  Months with >5 gap-filled climate days (flag in paper):")
print(f"    {sorted(set(months_flagged))}")


# ── 7. SAVE ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: Saving")
print("=" * 60)

out_path = os.path.join(script_dir, OUTPUT_FILE)
dataset.to_csv(out_path, index=False)

print(f"\n  Saved: {out_path}")
print(f"  Shape: {dataset.shape}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("""
What you have now:
  modelling_dataset.csv — the final dataset ready for modelling

Columns:
  Identity  : material, month_str, year, month_num
  Target    : break_count, km_primary, failure_rate
  Network   : sin_month, cos_month, time_index
  Climate   : Tmean, Tmin, Tmax, ADD, FI, FD, FTC, FI_cum,
              TI, TD, TIG, TDG
  Lags      : Tmean_lag1/2/3, FI_lag1/2/3, FTC_lag1/2/3,
              FI_cum_lag1/2/3, FD_lag1/2/3, ADD_lag1/2/3
  Quality   : missing_days

Next step:
  Load modelling_dataset.csv and fit the negative binomial model
  separately for each material (CI, DI, PVC).
""")
