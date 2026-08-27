"""
make_figures.py
===============
Generates the 4 figures for the Kitchener climate-water main paper.

Figure 1: Projected CI failure rate to 2100 under 8 scenarios
Figure 2: Monthly seasonal pattern by material (CI vs DI vs PVC)
Figure 3: Actual vs predicted annual break rate on test period
Figure 4: DI rate multiplier by prior break count

Required data files (in same folder as script):
  - modelling_dataset.csv        (from build_final_dataset.py)
  - projections_annual.csv       (from project_breaks.py)
  - predictions_ci_splitB.csv    (from model_negbinom_v2.py)
  - predictions_di_splitB.csv    (from model_negbinom_v2.py)

Output:
  All 4 figures saved as PNG at 300 dpi in the same folder.

Run:
  pip install pandas numpy matplotlib
  python make_figures.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── PLOT STYLE (professional, journal-ready) ─────────────────────────────────

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUT_DIR = os.getcwd()


# ── LOAD DATA ────────────────────────────────────────────────────────────────

print("Loading data files...")

dataset  = pd.read_csv(os.path.join(OUT_DIR, 'modelling_dataset.csv'))
proj     = pd.read_csv(os.path.join(OUT_DIR, 'projections_annual.csv'))
pred_ci  = pd.read_csv(os.path.join(OUT_DIR, 'predictions_ci_splitB.csv'))
pred_di  = pd.read_csv(os.path.join(OUT_DIR, 'predictions_di_splitB.csv'))

print(f"  modelling_dataset:  {dataset.shape}")
print(f"  projections_annual: {proj.shape}")
print(f"  predictions_ci:     {pred_ci.shape}")
print(f"  predictions_di:     {pred_di.shape}")


# ── FIGURE 1: PROJECTED CI FAILURE RATE TO 2100 ──────────────────────────────

print("\nGenerating Figure 1 — CI projections to 2100...")

fig, ax = plt.subplots(figsize=(9, 5))

hist_ci = 0.3462  # historical baseline breaks/km/year

# Colour by SSP (climate ambition), linestyle by GCM
ssp_colors = {
    'ssp126': '#2c7bb6',   # blue
    'ssp245': '#abd9e9',   # light blue
    'ssp370': '#fdae61',   # orange
    'ssp585': '#d7191c',   # red
}
ssp_labels = {
    'ssp126': 'SSP1-2.6',
    'ssp245': 'SSP2-4.5',
    'ssp370': 'SSP3-7.0',
    'ssp585': 'SSP5-8.5',
}
gcm_styles = {'CanESM5': '-', 'MIROC6': '--'}

for (model, scen), grp in proj.groupby(['model', 'scenario']):
    grp = grp.sort_values('year')
    ax.plot(grp['year'], grp['rate_ci'],
            color=ssp_colors[scen], linestyle=gcm_styles[model],
            linewidth=1.4, alpha=0.85,
            label=f"{model} {ssp_labels[scen]}")

ax.axhline(hist_ci, color='black', linestyle=':', linewidth=1.8,
           label='Historical baseline (1997-2025)')
ax.set_xlabel('Year')
ax.set_ylabel('CI failure rate (breaks/km/year)')
ax.set_title('Projected Cast Iron Failure Rate to 2100 Under Eight Climate Scenarios',
             fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
ax.set_xlim(2026, 2100)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_1_CI_projections.png'))
plt.close()
print("  Saved: Figure_1_CI_projections.png")


# ── FIGURE 2: MONTHLY SEASONAL PATTERN BY MATERIAL ───────────────────────────

print("\nGenerating Figure 2 — Monthly seasonal pattern...")

fig, ax = plt.subplots(figsize=(9, 5))

months = list(range(1, 13))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

material_colors = {'CI': '#8B4513', 'DI': '#4682B4', 'PVC': '#228B22'}
material_labels = {'CI': 'Cast Iron (CI)',
                   'DI': 'Ductile Iron (DI)',
                   'PVC': 'PVC'}

# Compute % of breaks by month for each material
bar_width = 0.27
positions = np.arange(len(months))

for i, mat in enumerate(['CI', 'DI', 'PVC']):
    sub = dataset[dataset['material'] == mat]
    total_breaks = sub['break_count'].sum()
    monthly_share = sub.groupby('month_num')['break_count'].sum() / total_breaks * 100
    ax.bar(positions + i*bar_width, monthly_share.reindex(months).values,
           bar_width, color=material_colors[mat], label=material_labels[mat],
           alpha=0.85, edgecolor='black', linewidth=0.5)

# Highlight winter months
for m in [0, 1, 11]:  # Jan, Feb, Dec (0-indexed)
    ax.axvspan(m - 0.15, m + 3*bar_width - 0.15, alpha=0.08, color='blue', zorder=0)

ax.set_xticks(positions + bar_width)
ax.set_xticklabels(month_names)
ax.set_xlabel('Month')
ax.set_ylabel('Share of annual breaks (%)')
ax.set_title('Monthly Distribution of Water Main Breaks by Material (1997-2025)',
             fontweight='bold')
ax.set_ylim(0, 25)
ax.legend(loc='upper center', framealpha=0.9, bbox_to_anchor=(0.55, 0.98), ncol=3)

# Add annotation for winter concentration
ax.text(0.98, 0.98,
        'Winter (DJF) share:\nCI = 60.3%\nDI = 40.1%\nPVC = 55.0%',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                  edgecolor='gray', alpha=0.95))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_2_seasonal_pattern.png'))
plt.close()
print("  Saved: Figure_2_seasonal_pattern.png")


# ── FIGURE 3: ACTUAL VS PREDICTED ANNUAL FAILURE RATE ────────────────────────

print("\nGenerating Figure 3 — Actual vs predicted (test period)...")

# Add km_primary to predictions from the dataset
km_ci = dataset[(dataset['material']=='CI') &
                (dataset['year']>=2021)][['month_str','km_primary']]
km_di = dataset[(dataset['material']=='DI') &
                (dataset['year']>=2021)][['month_str','km_primary']]

pred_ci_full = pred_ci.merge(km_ci, on='month_str')
pred_di_full = pred_di.merge(km_di, on='month_str')

# Aggregate to annual
ci_annual = pred_ci_full.groupby('year').agg(
    actual=('break_count','sum'),
    predicted=('predicted','sum'),
    km=('km_primary','mean')).reset_index()
ci_annual['actual_rate']    = ci_annual['actual']    / ci_annual['km']
ci_annual['predicted_rate'] = ci_annual['predicted'] / ci_annual['km']

di_annual = pred_di_full.groupby('year').agg(
    actual=('break_count','sum'),
    predicted=('predicted','sum'),
    km=('km_primary','mean')).reset_index()
di_annual['actual_rate']    = di_annual['actual']    / di_annual['km']
di_annual['predicted_rate'] = di_annual['predicted'] / di_annual['km']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# CI
x = np.arange(len(ci_annual))
w = 0.35
ax1.bar(x - w/2, ci_annual['actual_rate'], w, label='Actual',
        color='#2C3E50', alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.bar(x + w/2, ci_annual['predicted_rate'], w, label='Predicted',
        color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(ci_annual['year'].astype(int))
ax1.set_xlabel('Year')
ax1.set_ylabel('Failure rate (breaks/km/year)')
ax1.set_title('(a) Cast Iron — Test period (2021-2025)', fontweight='bold')
ax1.legend(loc='upper right')
ax1.text(0.02, 0.95, f'MAE = 0.085\nMASE = 0.865',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='gray', alpha=0.9))

# DI
x = np.arange(len(di_annual))
ax2.bar(x - w/2, di_annual['actual_rate'], w, label='Actual',
        color='#2C3E50', alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.bar(x + w/2, di_annual['predicted_rate'], w, label='Predicted',
        color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(di_annual['year'].astype(int))
ax2.set_xlabel('Year')
ax2.set_ylabel('Failure rate (breaks/km/year)')
ax2.set_title('(b) Ductile Iron — Test period (2021-2025)', fontweight='bold')
ax2.legend(loc='upper right')
ax2.text(0.02, 0.95, f'MAE = 0.022\nMASE = 0.905',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='gray', alpha=0.9))

plt.suptitle('Model Validation: Actual vs Predicted Annual Failure Rate',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_3_validation.png'))
plt.close()
print("  Saved: Figure_3_validation.png")


# ── FIGURE 4: DI RATE MULTIPLIER BY PRIOR BREAK COUNT ────────────────────────

print("\nGenerating Figure 4 — DI prior break multiplier...")

fig, ax = plt.subplots(figsize=(8, 5))

prior_breaks = [0, 1, 2, 3, 4, 5]
multipliers  = [1.00, 1.76, 3.10, 5.46, 9.62, 16.95]

colors = plt.cm.Reds(np.linspace(0.35, 0.95, len(prior_breaks)))

bars = ax.bar(prior_breaks, multipliers, color=colors,
              edgecolor='black', linewidth=0.8, alpha=0.9)

# Value labels on bars
for bar, mult in zip(bars, multipliers):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{mult:.2f}×', ha='center', va='bottom', fontsize=10,
            fontweight='bold')

ax.set_xlabel('Number of prior breaks on same pipe')
ax.set_ylabel('Rate multiplier vs pipe with 0 prior breaks')
ax.set_title('DI Failure Rate Multiplier by Prior Break Count',
             fontweight='bold')
ax.set_xticks(prior_breaks)
ax.set_ylim(0, 20)

# Annotation
ax.text(0.98, 0.55,
        'Model coefficient:\nprior_breaks: +0.554\n(p < 0.001)\n\n'
        'Interpretation:\nEach additional prior\nbreak multiplies rate\nby 1.76×',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                  edgecolor='gray', alpha=0.95))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_4_prior_breaks.png'))
plt.close()
print("  Saved: Figure_4_prior_breaks.png")


# ── DONE ─────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DONE — 4 figures generated")
print("="*60)
print("""
Figures saved:
  Figure_1_CI_projections.png   — CI failure rate to 2100
  Figure_2_seasonal_pattern.png — Monthly breaks by material
  Figure_3_validation.png       — Actual vs predicted (test period)
  Figure_4_prior_breaks.png     — DI prior break multiplier

All at 300 dpi, ready for the report.
""")
