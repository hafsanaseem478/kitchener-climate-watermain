"""
build_climate.py
================
Merges all Kitchener/Waterloo climate station files into one clean daily
record (1996-2025), fills small gaps by linear interpolation, then
calculates monthly climate covariates needed for the water main failure model.

STATION PRIORITY (best data quality first):
  1. KITCHENER/WATERLOO          (6144239) -- primary, 2010-2025
  2. REGION OF WATERLOO INT'L    (6149388) -- 2002-2010
  3. WATERLOO WELLINGTON A       (6149387) -- 1996-2002
  4. WATERLOO WELLINGTON 2       (6149389) -- backup
  5. WATERLOO WPCP               (6149386) -- backup

OUTPUT FILES:
  climate_daily_merged.csv    -- one clean row per day 1996-2025
  climate_monthly.csv         -- monthly covariates, ready to merge with panel

MONTHLY COVARIATES PRODUCED:
  Tmean      -- mean of daily mean temperature (deg C)
  Tmin       -- mean of daily minimum temperature (deg C)
  Tmax       -- mean of daily maximum temperature (deg C)
  ADD        -- avg daily temperature range: mean(Tmax - Tmin)
  FI         -- freezing index: sum of min(Tmean, 0) [deg-C-days]
  FD         -- freezing days: count of days with Tmean < 0
  FTC        -- freeze-thaw cycles: days where Tmin < 0 AND Tmax > 0
  TI         -- thawing index: sum of max(Tmean, 0) [deg-C-days]
  TD         -- thawing days: count of days with Tmean > 0
  TIG        -- max rate of temperature increase over consecutive days
  TDG        -- max rate of temperature decrease over consecutive days
  FI_cum     -- cumulative freezing index since start of freeze season
  RI         -- total monthly rainfall (mm)
  RD         -- rainfall deficit vs 1981-2010 normal (mm)
  PREC       -- total monthly precipitation (mm)
  missing_days -- count of days gap-filled by interpolation that month

Run:
  python build_climate.py

Put all climate CSV files in the same folder as this script.
Requires: pandas, numpy
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#  0. SETTINGS
STUDY_START = "1996-01-01"
STUDY_END   = "2025-12-31"

# Max consecutive missing days to fill by interpolation
# Beyond this, the gap is too large to reliably fill
MAX_GAP_DAYS = 5

# 1981-2010 monthly normals from WATERLOO WELLINGTON A (climate-normals.csv)
# Used ONLY for rainfall deficit (RD) calculation
NORMAL_RAINFALL_MM = {
    1: 28.66, 2: 29.74, 3: 36.81,  4: 67.98,
    5: 81.80, 6: 82.39, 7: 98.55,  8: 83.92,
    9: 87.79, 10: 66.09, 11: 75.02, 12: 38.02
}

# Station priority: lower number = preferred
STATION_PRIORITY = {
    'KITCHENER/WATERLOO':                  1,
    "REGION OF WATERLOO INT'L AIRPORT":    2,
    'WATERLOO WELLINGTON A':               3,
    'WATERLOO WELLINGTON 2':               4,
    'WATERLOO WPCP':                       5,
}

OUTPUT_DAILY   = "climate_daily_merged.csv"
OUTPUT_MONTHLY = "climate_monthly.csv"


#  1. LOAD ALL CLIMATE FILES 

print("=" * 60)
print("STEP 1: Loading climate files")
print("=" * 60)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# Find all climate CSV files - user should put them all in same folder
climate_files = glob.glob(os.path.join(script_dir, "climate-daily*.csv")) + \
                glob.glob(os.path.join(script_dir, "WATERLOO*.csv"))     + \
                glob.glob(os.path.join(script_dir, "KITCHENER*.csv"))

if not climate_files:
    raise FileNotFoundError(
        "No climate files found in current folder.\n"
        "Make sure all climate-daily*.csv, WATERLOO*.csv, "
        "and KITCHENER*.csv files are in the same folder as this script."
    )

print(f"  Found {len(climate_files)} climate files:")
for f in sorted(climate_files):
    print(f"    {os.path.basename(f)}")

all_dfs = []
for f in climate_files:
    try:
        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
        all_dfs.append(df)
    except Exception as e:
        print(f"  WARNING: Could not read {os.path.basename(f)}: {e}")

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Total rows loaded: {len(combined):,}")


#  2. PARSE DATES AND FILTER TO STUDY WINDOW 

print("\n" + "=" * 60)
print("STEP 2: Parsing dates and filtering to study window")
print("=" * 60)

combined['date'] = pd.to_datetime(combined['LOCAL_DATE'], errors='coerce')
combined = combined[combined['date'].notna()].copy()
combined = combined[
    (combined['date'] >= STUDY_START) &
    (combined['date'] <= STUDY_END)
].copy()
combined['year']  = combined['date'].dt.year
combined['month'] = combined['date'].dt.month
combined['day']   = combined['date'].dt.day

print(f"  Rows in study window 1996-2025: {len(combined):,}")
print(f"  Stations present:")
for stn, grp in combined.groupby('STATION_NAME'):
    print(f"    {stn}: {grp['date'].dt.year.min():.0f}–"
          f"{grp['date'].dt.year.max():.0f} "
          f"({len(grp):,} rows)")


# ── 3. ASSIGN PRIORITY AND SELECT BEST STATION PER DAY ───────────────────────

print("\n" + "=" * 60)
print("STEP 3: Selecting best station per day")
print("=" * 60)

combined['priority'] = combined['STATION_NAME'].map(STATION_PRIORITY).fillna(99)

# Sort by date then priority so the best station comes first
combined_sorted = combined.sort_values(['date', 'priority'])

# For each day keep only the row from the highest-priority station
# that has actual temperature data
# First try: keep rows with MEAN_TEMPERATURE not null
has_temp = combined_sorted[combined_sorted['MEAN_TEMPERATURE'].notna()]
best_with_temp = has_temp.drop_duplicates('date', keep='first')

# For days with no temperature at all, keep the best-priority row anyway
# (we will gap-fill later)
all_days_best = combined_sorted.drop_duplicates('date', keep='first')

# Merge: use temp-row where available, fallback row otherwise
daily = all_days_best.copy()
temp_dates = set(best_with_temp['date'])
daily = daily[~daily['date'].isin(temp_dates)]
daily = pd.concat([best_with_temp, daily], ignore_index=True)
daily = daily.sort_values('date').reset_index(drop=True)

print(f"  Unique days after merge: {len(daily):,}")
print(f"  Days with MEAN_TEMPERATURE: {daily['MEAN_TEMPERATURE'].notna().sum():,}")
print(f"  Days with MIN_TEMPERATURE:  {daily['MIN_TEMPERATURE'].notna().sum():,}")
print(f"  Days with MAX_TEMPERATURE:  {daily['MAX_TEMPERATURE'].notna().sum():,}")
print(f"  Days with TOTAL_RAIN:       {daily['TOTAL_RAIN'].notna().sum():,}")


#  4. BUILD COMPLETE DATE SPINE 

print("\n" + "=" * 60)
print("STEP 4: Building complete date spine and filling gaps")
print("=" * 60)

# Create a row for every single day in the study window
spine = pd.DataFrame({
    'date': pd.date_range(start=STUDY_START, end=STUDY_END, freq='D')
})

# Merge our best-station data onto the spine
# Days where no station had any data will be NaN
daily_full = spine.merge(
    daily[['date', 'STATION_NAME', 'priority',
           'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
           'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'TOTAL_SNOW']],
    on='date', how='left'
)

daily_full['year']  = daily_full['date'].dt.year
daily_full['month'] = daily_full['date'].dt.month
daily_full['day']   = daily_full['date'].dt.day

total_days = len(daily_full)
print(f"  Total days in spine: {total_days:,}")

# Flag missing days BEFORE gap filling (so we can count them per month)
daily_full['was_missing'] = daily_full['MEAN_TEMPERATURE'].isna().astype(int)
missing_total = daily_full['was_missing'].sum()
print(f"  Days with missing MEAN_TEMPERATURE: {missing_total} ({100*missing_total/total_days:.1f}%)")

# Gap fill by linear interpolation, but ONLY for gaps <= MAX_GAP_DAYS
# Longer gaps are left as NaN and flagged
for col in ['MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
            'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'TOTAL_SNOW']:
    daily_full[col] = (
        daily_full[col]
        .interpolate(method='linear', limit=MAX_GAP_DAYS,
                     limit_direction='both')
    )

filled = daily_full['MEAN_TEMPERATURE'].notna().sum()
print(f"  Days with MEAN_TEMPERATURE after gap-fill (max {MAX_GAP_DAYS} days): "
      f"{filled:,} ({100*filled/total_days:.1f}%)")

still_missing = daily_full['MEAN_TEMPERATURE'].isna().sum()
if still_missing > 0:
    print(f"  WARNING: {still_missing} days still missing after interpolation.")
    print("  These are gaps longer than 5 days. Monthly values for those months")
    print("  will be based on fewer days than normal.")
    print("  Missing by year:")
    miss_yr = daily_full[daily_full['MEAN_TEMPERATURE'].isna()].groupby('year').size()
    print(miss_yr.to_string())


#  5. CALCULATE MONTHLY CLIMATE COVARIATES 

print("\n" + "=" * 60)
print("STEP 5: Calculating monthly covariates")
print("=" * 60)

def calc_gradients(group):
    """
    For a month's daily Tmean series, calculate:
    TIG = max rate of temperature increase over any pair of consecutive days
    TDG = max rate of temperature decrease over any pair of consecutive days
    Per Khashei et al. (2024): TIG = max{(T_k - T_j)/(k-j)} for j < k
    We use consecutive days only (k-j=1) which is the most common interpretation
    and avoids over-smoothing.
    """
    t = group['MEAN_TEMPERATURE'].dropna().values
    if len(t) < 2:
        return pd.Series({'TIG': np.nan, 'TDG': np.nan})
    diffs = np.diff(t)
    TIG = float(np.max(diffs))   if len(diffs) > 0 else np.nan
    TDG = float(np.max(-diffs))  if len(diffs) > 0 else np.nan
    return pd.Series({'TIG': TIG, 'TDG': TDG})

#  Helper: Freeze-thaw cycles 
def calc_ftc(group):
    """
    Freeze-thaw cycle: a day where Tmin < 0 AND Tmax > 0.
    This means the temperature crossed zero in both directions on the same day.
    """
    ftc = ((group['MIN_TEMPERATURE'] < 0) & (group['MAX_TEMPERATURE'] > 0)).sum()
    return int(ftc)

# ── Helper: Cumulative freezing index from start of freeze season ─────────────
def calc_fi_cumulative(df_year, year, month):
    """
    FI_cum = cumulative freezing index since the start of the freeze season.
    Freeze season starts in October of the previous year.
    So for Jan 2010, we sum FI from Oct 2009 through Jan 2010.
    """
    # Freeze season start: October of previous year if month < 10, else October of this year
    if month < 10:
        season_start = pd.Timestamp(year=year-1, month=10, day=1)
    else:
        season_start = pd.Timestamp(year=year, month=10, day=1)
    season_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    mask = (df_year['date'] >= season_start) & (df_year['date'] <= season_end)
    fi_cum = df_year.loc[mask, 'MEAN_TEMPERATURE'].apply(lambda x: min(x, 0) if pd.notna(x) else 0).sum()
    return round(fi_cum, 2)

# ── Main monthly calculation loop ─────────────────────────────────────────────

monthly_rows = []

for (yr, mo), grp in daily_full.groupby(['year', 'month']):
    grp = grp.copy()
    n_days       = len(grp)
    n_valid_temp = grp['MEAN_TEMPERATURE'].notna().sum()
    n_missing    = int(grp['was_missing'].sum())

    # Skip month if more than half the days are still missing
    if n_valid_temp < n_days / 2:
        print(f"  WARNING: {yr}-{mo:02d} has only {n_valid_temp}/{n_days} valid "
              f"temp days — covariates will be unreliable")

    # Temperature level covariates
    Tmean = grp['MEAN_TEMPERATURE'].mean()
    Tmin  = grp['MIN_TEMPERATURE'].mean()
    Tmax  = grp['MAX_TEMPERATURE'].mean()
    ADD   = (grp['MAX_TEMPERATURE'] - grp['MIN_TEMPERATURE']).mean()

    # Frost covariates
    FI  = grp['MEAN_TEMPERATURE'].apply(lambda x: min(x, 0) if pd.notna(x) else 0).sum()
    FD  = (grp['MEAN_TEMPERATURE'] < 0).sum()
    FTC = calc_ftc(grp)

    # Thaw covariates
    TI  = grp['MEAN_TEMPERATURE'].apply(lambda x: max(x, 0) if pd.notna(x) else 0).sum()
    TD  = (grp['MEAN_TEMPERATURE'] > 0).sum()

    # Temperature gradient covariates
    grads = calc_gradients(grp)
    TIG   = grads['TIG']
    TDG   = grads['TDG']

    # Cumulative freezing index since freeze season start
    FI_cum = calc_fi_cumulative(daily_full, yr, mo)

    # Precipitation covariates
    RI   = grp['TOTAL_RAIN'].sum()
    PREC = grp['TOTAL_PRECIPITATION'].sum()

    # Rainfall deficit vs 1981-2010 normal
    RD   = round(RI - NORMAL_RAINFALL_MM.get(mo, np.nan), 2)

    monthly_rows.append({
        'year':        yr,
        'month':       mo,
        'month_str':   f"{yr}-{mo:02d}",
        'n_days':      n_days,
        'n_valid_temp':n_valid_temp,
        'missing_days':n_missing,
        # Temperature
        'Tmean':   round(Tmean, 3) if pd.notna(Tmean) else np.nan,
        'Tmin':    round(Tmin,  3) if pd.notna(Tmin)  else np.nan,
        'Tmax':    round(Tmax,  3) if pd.notna(Tmax)  else np.nan,
        'ADD':     round(ADD,   3) if pd.notna(ADD)   else np.nan,
        # Frost
        'FI':      round(FI,    2),
        'FD':      int(FD),
        'FTC':     int(FTC),
        'FI_cum':  FI_cum,
        # Thaw
        'TI':      round(TI, 2),
        'TD':      int(TD),
        # Gradient
        'TIG':     round(TIG, 3) if pd.notna(TIG) else np.nan,
        'TDG':     round(TDG, 3) if pd.notna(TDG) else np.nan,
        # Precipitation
        'RI':      round(RI,   2),
        'RD':      RD,
        'PREC':    round(PREC, 2),
    })

monthly = pd.DataFrame(monthly_rows)
monthly = monthly.sort_values(['year','month']).reset_index(drop=True)

print(f"\n  Monthly covariate table shape: {monthly.shape}")
print(f"  Rows: {len(monthly)} (expect 360 for 1996-2025)")
print(f"\n  Sample — first 6 months:")
print(monthly.head(6)[['month_str','Tmean','FI','FD','FTC','FI_cum','RI','RD']].to_string(index=False))


#  6. SANITY CHECKS 

print("\n" + "=" * 60)
print("STEP 6: Sanity checks")
print("=" * 60)
# FI should be most negative (largest freezing magnitude) in Jan/Feb

print("\n  Mean FI by month (expect highest in Jan/Feb):")
print(monthly.groupby('month')['FI'].mean().round(1).to_string())

# Check 2: FTC should peak in March/April (transition season)
print("\n  Mean FTC by month (expect peak in Mar/Apr):")
print(monthly.groupby('month')['FTC'].mean().round(1).to_string())

# Check 3: Tmean should be negative Nov-Mar
print("\n  Mean Tmean by month (expect negative Nov-Mar):")
print(monthly.groupby('month')['Tmean'].mean().round(1).to_string())

# Check 4: RI should be highest in summer
print("\n  Mean RI by month (expect highest Jun-Sep):")
print(monthly.groupby('month')['RI'].mean().round(1).to_string())

# Check 5: Missing data flag
total_missing = monthly['missing_days'].sum()
print(f"\n  Total gap-filled days across all months: {total_missing}")
print(f"  Months with any gap-filled days: {(monthly['missing_days']>0).sum()}")
high_missing = monthly[monthly['missing_days'] > 5]
if len(high_missing) > 0:
    print(f"  Months with >5 gap-filled days (flag these in paper):")
    print(high_missing[['month_str','missing_days']].to_string(index=False))

# Check 6: No NaN in key covariates
print("\n  NaN counts in key covariates:")
key_cols = ['Tmean','Tmin','Tmax','FI','FD','FTC','TI','TD','TIG','TDG','FI_cum']
for col in key_cols:
    n = monthly[col].isna().sum()
    if n > 0:
        print(f"    {col}: {n} NaN values")
    else:
        print(f"    {col}: OK")


#  7. SAVE OUTPUTS 

print("\n" + "=" * 60)
print("STEP 7: Saving outputs")
print("=" * 60)

daily_out_path   = os.path.join(script_dir, OUTPUT_DAILY)
monthly_out_path = os.path.join(script_dir, OUTPUT_MONTHLY)

# Daily merged file
daily_full[['date','year','month','day','STATION_NAME',
            'MEAN_TEMPERATURE','MIN_TEMPERATURE','MAX_TEMPERATURE',
            'TOTAL_RAIN','TOTAL_PRECIPITATION','TOTAL_SNOW',
            'was_missing']].to_csv(daily_out_path, index=False)

# Monthly covariate file
monthly.to_csv(monthly_out_path, index=False)

print(f"\n  Daily merged file  : {daily_out_path}")
print(f"  Rows: {len(daily_full):,}  |  Columns: 12")
print(f"\n  Monthly covariate file: {monthly_out_path}")
print(f"  Rows: {len(monthly):,}  |  Columns: {len(monthly.columns)}")
print(f"\n  Columns in monthly file:")
for col in monthly.columns:
    print(f"    {col}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print("\nNext step:")
print("  Load climate_monthly.csv and merge with monthly_panel_primary.csv")
print("  on 'month_str' to get the final modelling dataset.")
