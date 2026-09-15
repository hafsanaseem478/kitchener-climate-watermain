
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BREAKS_FILE = "Water_Main_Breaks.csv"
MAINS_FILE  = "Water_Mains.csv"

OUTPUT_PRIMARY     = "monthly_panel_primary.csv"
OUTPUT_SENSITIVITY = "monthly_panel_sensitivity.csv"

STUDY_START    = "1997-01"
STUDY_END      = "2025-12"
CORE_MATERIALS = ["CI", "DI", "PVC"]

# Mean segment length per material from current inventory (metres)
# Used ONLY for the sensitivity analysis — not the primary model
MEAN_SEGMENT_M = {"CI": 75.1, "DI": 63.2, "PVC": 47.4}

print("Loading data...")

breaks = pd.read_csv(BREAKS_FILE, encoding="utf-8-sig", low_memory=False)
mains  = pd.read_csv(MAINS_FILE,  encoding="utf-8-sig", low_memory=False)

print(f"  Breaks loaded : {len(breaks):,} rows")
print(f"  Mains loaded  : {len(mains):,} rows")

print("\nCleaning break records...")

breaks["incident_dt"]  = pd.to_datetime(breaks["Incident date"], errors="coerce")
breaks["year"]         = breaks["incident_dt"].dt.year
breaks["month_period"] = breaks["incident_dt"].dt.to_period("M")

# MAIN breaks only (not service connections)
breaks = breaks[breaks["Type of Asset Broken"] == "MAIN"].copy()

# Complete study years only
breaks = breaks[(breaks["year"] >= 1997) & (breaks["year"] <= 2025)].copy()

# Three core materials only
breaks = breaks[breaks["Asset Material"].isin(CORE_MATERIALS)].copy()

# Must have a valid date
breaks = breaks[breaks["incident_dt"].notna()].copy()

print(f"  Breaks after cleaning : {len(breaks):,}")
print(f"  Material breakdown:")
print(breaks["Asset Material"].value_counts().to_string(header=False))

# Flag retired pipes — used for sensitivity only
n_retired = (breaks["Asset Exists"] == "N").sum()
print(f"\n  Breaks on retired pipes (Asset Exists=N): {n_retired:,}"
      f" ({100*n_retired/len(breaks):.1f}%)")
print("  These are excluded from the primary exposure but included"
      " in the sensitivity check.")

print("\nCleaning mains inventory...")

mains["install_dt"] = pd.to_datetime(mains["INSTALLATION_DATE"], errors="coerce")
mains["install_yr"] = mains["install_dt"].dt.year

mains = mains[mains["MATERIAL"].isin(CORE_MATERIALS)].copy()
mains = mains[mains["Shape__Length"] > 0].copy()
mains["length_km"] = mains["Shape__Length"] / 1000.0

print(f"  Mains after cleaning : {len(mains):,}")
print(f"  Total length         : {mains['length_km'].sum():.1f} km")
print(f"  Material breakdown (km):")
print(mains.groupby("MATERIAL")["length_km"].sum().round(1).to_string())
print(f"\n  Mean segment length by material (m) — used for sensitivity only:")
print(mains.groupby("MATERIAL")["Shape__Length"].mean().round(1).to_string())

monthly_index = pd.period_range(start=STUDY_START, end=STUDY_END, freq="M")
n_months = len(monthly_index)
print(f"\nStudy window: {STUDY_START} to {STUDY_END} ({n_months} months)")

print("\nBuilding monthly exposure...")

primary_rows = []
for mat in CORE_MATERIALS:
    mat_pipes = mains[mains["MATERIAL"] == mat]
    for month in monthly_index:
        yr = month.year
        km = mat_pipes.loc[mat_pipes["install_yr"] <= yr, "length_km"].sum()
        primary_rows.append({"material": mat, "month": month, "km_primary": km})

exposure_primary = pd.DataFrame(primary_rows)

# Get one record per retired pipe: material, install_yr, last_break_yr
retired_breaks = breaks[
    (breaks["Asset Exists"] == "N") &
    (breaks["Asset Material"].isin(CORE_MATERIALS))
].copy()

retired_breaks["install_yr_b"] = pd.to_numeric(
    retired_breaks["Year Asset Installed"], errors="coerce"
)

# Group by material + install_yr + road segment to identify unique retired pipes
retired_pipes = (
    retired_breaks[retired_breaks["install_yr_b"].notna()]
    .groupby(["Asset Material", "install_yr_b", "Road Segment ID"])
    .agg(last_break_yr=("year", "max"))
    .reset_index()
    .rename(columns={"Asset Material": "material",
                     "install_yr_b":   "install_yr"})
)

# Estimated retirement year = year after last known break
retired_pipes["retire_yr"] = retired_pipes["last_break_yr"] + 1

# Approximate length = mean segment length for that material
retired_pipes["length_km"] = (
    retired_pipes["material"].map(MEAN_SEGMENT_M) / 1000.0
)

# Sanity: clamp install years
retired_pipes = retired_pipes[
    (retired_pipes["install_yr"] >= 1880) &
    (retired_pipes["install_yr"] <= 2025)
].copy()

print(f"  Retired pipes reconstructed (for sensitivity): {len(retired_pipes):,}")
print(f"  Breakdown by material:")
print(retired_pipes["material"].value_counts().to_string(header=False))

# Build sensitivity exposure
sensitivity_rows = []
for mat in CORE_MATERIALS:
    mat_ret  = retired_pipes[retired_pipes["material"] == mat]
    # Primary km already computed above — look it up
    prim_mat = exposure_primary[exposure_primary["material"] == mat].set_index("month")

    for month in monthly_index:
        yr = month.year
        km_prim = prim_mat.loc[month, "km_primary"]

        # Add km from retired pipes active during this year
        km_ret = mat_ret.loc[
            (mat_ret["install_yr"] <= yr) & (mat_ret["retire_yr"] > yr),
            "length_km"
        ].sum()

        sensitivity_rows.append({
            "material":    mat,
            "month":       month,
            "km_primary":  km_prim,
            "km_retired":  km_ret,
            "km_sensitivity": km_prim + km_ret
        })

exposure_sensitivity = pd.DataFrame(sensitivity_rows)

print(f"\n  Average km in service by material:")
print(f"  {'Material':<10} {'Primary (snapshot)':>20} {'Sensitivity (+retired)':>24} {'Diff %':>8}")
for mat in CORE_MATERIALS:
    p = exposure_primary[exposure_primary["material"]==mat]["km_primary"].mean()
    s = exposure_sensitivity[exposure_sensitivity["material"]==mat]["km_sensitivity"].mean()
    print(f"  {mat:<10} {p:>20.1f} {s:>24.1f} {100*(s-p)/p:>7.2f}%")

print("\n  → If Diff% is small (<5%), the correction does not materially")
print("    affect results and can be safely set aside (report in paper).")

#COUNT BREAKS PER MATERIAL PER MONTH
print("\nCounting breaks per material per month...")

break_counts = (
    breaks
    .groupby(["Asset Material", "month_period"])
    .size()
    .reset_index(name="break_count")
    .rename(columns={"Asset Material": "material", "month_period": "month"})
)

# Full grid so zero months are explicit, not missing
full_grid = pd.MultiIndex.from_product(
    [CORE_MATERIALS, monthly_index], names=["material", "month"]
).to_frame(index=False)

break_counts = full_grid.merge(break_counts, on=["material", "month"], how="left")
break_counts["break_count"] = break_counts["break_count"].fillna(0).astype(int)

print(f"  Total breaks by material:")
print(break_counts.groupby("material")["break_count"].sum().to_string())
print(f"\n  Zero-count months by material:")
zero = break_counts[break_counts["break_count"] == 0].groupby("material").size()
print((zero.astype(str) + " / " + str(n_months)).to_string())

def build_panel(break_counts, exposure_df, km_col):
    panel = break_counts.merge(exposure_df, on=["material", "month"])
    panel["failure_rate"] = np.where(
        panel[km_col] > 0,
        panel["break_count"] / panel[km_col],
        np.nan
    )
    panel["year"]      = panel["month"].dt.year
    panel["month_num"] = panel["month"].dt.month
    panel["sin_month"] = np.sin(2 * np.pi * panel["month_num"] / 12)
    panel["cos_month"] = np.cos(2 * np.pi * panel["month_num"] / 12)
    panel["time_index"]= (panel["year"] - 1997) * 12 + panel["month_num"] - 1
    panel["month_str"] = panel["month"].astype(str)
    return panel.sort_values(["material", "month_str"]).reset_index(drop=True)


print("\nBuilding primary panel...")
panel_primary = build_panel(
    break_counts,
    exposure_primary[["material", "month", "km_primary"]],
    "km_primary"
)
panel_primary = panel_primary[[
    "material", "month_str", "year", "month_num",
    "break_count", "km_primary", "failure_rate",
    "sin_month", "cos_month", "time_index"
]]

print("Building sensitivity panel...")
panel_sensitivity = build_panel(
    break_counts,
    exposure_sensitivity[["material", "month", "km_primary",
                           "km_retired", "km_sensitivity"]],
    "km_sensitivity"
)
panel_sensitivity = panel_sensitivity[[
    "material", "month_str", "year", "month_num",
    "break_count", "km_primary", "km_retired", "km_sensitivity",
    "failure_rate", "sin_month", "cos_month", "time_index"
]]

script_dir = os.getcwd()

path_primary     = os.path.join(script_dir, OUTPUT_PRIMARY)
path_sensitivity = os.path.join(script_dir, OUTPUT_SENSITIVITY)

panel_primary.to_csv(path_primary, index=False)
panel_sensitivity.to_csv(path_sensitivity, index=False)

print(f"\n✓ Primary panel saved     : {path_primary}")
print(f"  Shape: {panel_primary.shape}")
print(f"\n✓ Sensitivity panel saved : {path_sensitivity}")
print(f"  Shape: {panel_sensitivity.shape}")

print(f"\n  Columns in primary panel:")
for col in panel_primary.columns:
    print(f"    {col}")