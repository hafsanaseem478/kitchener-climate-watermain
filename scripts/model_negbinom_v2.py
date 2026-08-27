"""
model_negbinom_v2.py
====================
Negative binomial count regression for Kitchener water main failures.
Runs TWO temporal splits for robustness comparison.

SPLIT A (Primary):
  Train : 1997-2016 (240 months)
  Val   : 2017-2021 (60 months)   -- model selection
  Test  : 2022-2025 (48 months)   -- final evaluation

SPLIT B (Secondary):
  Train : 1997-2020 (288 months)
  Test  : 2021-2025 (60 months)   -- final evaluation
  Selection via rolling-origin cross-validation on training block
  (no separate val set needed)

MODELS:
  M0 -- Seasonal naive baseline (minimum benchmark)
  M1 -- Negative binomial, full features
  M2 -- Negative binomial, backward stepwise AIC selection

MATERIALS:
  CI  -- Full climate model
  DI  -- Full climate model
  PVC -- Seasonal baseline only (9 test events, insufficient for climate model)

OUTPUT:
  model_results_both_splits.csv   -- all metrics side by side
  coefficients_ci_splitA_m2.csv  -- CI best model, Split A
  coefficients_ci_splitB_m2.csv  -- CI best model, Split B
  coefficients_di_splitA_m2.csv  -- DI best model, Split A
  coefficients_di_splitB_m2.csv  -- DI best model, Split B
  predictions_ci_splitA.csv
  predictions_ci_splitB.csv
  predictions_di_splitA.csv
  predictions_di_splitB.csv

Run:
  python model_negbinom_v2.py

Requires: pandas, numpy, statsmodels, scipy
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import scipy.stats as stats
warnings.filterwarnings('ignore')

# ── 0. SETTINGS ───────────────────────────────────────────────────────────────

DATA_FILE = "modelling_dataset.csv"

# Split A
A_TRAIN_END  = 2016
A_VAL_START  = 2017
A_VAL_END    = 2021
A_TEST_START = 2022

# Split B
B_TRAIN_END  = 2020
B_TEST_START = 2021

# Rolling CV folds for Split B model selection
# Fit on 1997-FOLD_START, predict FOLD_START+1 to FOLD_START+1
B_CV_FOLD_STARTS = list(range(2010, 2020))  # 10 folds

# Climate covariates (TD, TI dropped — collinear)
CLIMATE_CURRENT = ['Tmean', 'FI', 'FTC', 'FI_cum', 'FD', 'ADD', 'TIG', 'TDG']
LAG_COLS        = ['Tmean', 'FI', 'FTC', 'FI_cum', 'FD', 'ADD']
LAG_MONTHS      = [1, 2, 3]
CLIMATE_LAGS    = [f'{c}_lag{l}' for c in LAG_COLS for l in LAG_MONTHS]
NETWORK_COLS    = ['sin_month', 'cos_month', 'time_index']
ALL_FEATURES    = CLIMATE_CURRENT + CLIMATE_LAGS + NETWORK_COLS

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading data")
print("=" * 60)
df = pd.read_csv(os.path.join(script_dir, DATA_FILE))
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['month_str'].min()} to {df['month_str'].max()}")

# Print split sizes
print("\n  Split sizes:")
for mat in ['CI', 'DI']:
    m = df[df['material'] == mat]
    trA = m[m['year'] <= A_TRAIN_END]
    vaA = m[(m['year'] >= A_VAL_START) & (m['year'] <= A_VAL_END)]
    teA = m[m['year'] >= A_TEST_START]
    trB = m[m['year'] <= B_TRAIN_END]
    teB = m[m['year'] >= B_TEST_START]
    print(f"  {mat} Split A: train={len(trA)}({trA.break_count.sum()}br) "
          f"val={len(vaA)}({vaA.break_count.sum()}br) "
          f"test={len(teA)}({teA.break_count.sum()}br)")
    print(f"  {mat} Split B: train={len(trB)}({trB.break_count.sum()}br) "
          f"test={len(teB)}({teB.break_count.sum()}br)")


# ── 2. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def fit_negbinom(train_df, features, offset_col='km_primary'):
    X      = sm.add_constant(train_df[features])
    y      = train_df['break_count']
    offset = np.log(train_df[offset_col].clip(lower=0.001))
    model  = GLM(y, X, family=families.NegativeBinomial(alpha=1.0), offset=offset)
    return model.fit(maxiter=200, disp=False)


def predict_negbinom(result, pred_df, features,
                     offset_col='km_primary', alpha=0.10):
    X      = sm.add_constant(pred_df[features], has_constant='add')
    offset = np.log(pred_df[offset_col].clip(lower=0.001))
    mu     = result.predict(X, offset=offset)
    nb_a   = result.scale
    lower  = stats.nbinom.ppf(alpha/2,   n=1/nb_a, p=1/(1 + nb_a*mu))
    upper  = stats.nbinom.ppf(1-alpha/2, n=1/nb_a, p=1/(1 + nb_a*mu))
    return mu.values, lower, upper


def monthly_metrics(actual, predicted, lower, upper, baseline_mae):
    mae     = float(np.mean(np.abs(actual - predicted)))
    mase    = mae / baseline_mae if baseline_mae > 0 else np.nan
    cov90   = float(np.mean((actual >= lower) & (actual <= upper))) * 100
    bias    = float(np.mean(predicted - actual))
    return {'MAE_monthly': round(mae,3), 'MASE': round(mase,3),
            'Coverage_90': round(cov90,1), 'Bias': round(bias,3)}


def annual_failure_rate_mae(pred_df):
    """
    Convert monthly predictions to annual failure rate and compute MAE.
    failure_rate = annual_breaks / km_in_service
    This is comparable to Khashei et al. (2024).
    """
    ann = pred_df.groupby('year').agg(
        actual_breaks    = ('break_count', 'sum'),
        predicted_breaks = ('predicted',   'sum'),
        km               = ('km_primary',  'mean')
    ).reset_index()
    ann['actual_rate']    = ann['actual_breaks']    / ann['km']
    ann['predicted_rate'] = ann['predicted_breaks'] / ann['km']
    ann['abs_error_rate'] = np.abs(ann['actual_rate'] - ann['predicted_rate'])
    mae_rate = float(ann['abs_error_rate'].mean())
    return round(mae_rate, 4), ann


def annual_count_mae(actual, predicted, years):
    """Sum monthly to annual and compute MAE on counts."""
    temp = pd.DataFrame({'a': actual, 'p': predicted, 'y': years})
    ann  = temp.groupby('y').sum()
    return round(float(np.mean(np.abs(ann['a'] - ann['p']))), 2)


def stepwise_aic(train_df, features, offset_col='km_primary', verbose=False):
    """Backward stepwise AIC selection."""
    current = features.copy()
    try:
        current_aic = fit_negbinom(train_df, current, offset_col).aic
    except Exception:
        return features
    improved = True
    while improved and len(current) > 1:
        improved = False
        best_aic, best_drop = current_aic, None
        for feat in current:
            trial = [f for f in current if f != feat]
            try:
                aic = fit_negbinom(train_df, trial, offset_col).aic
                if aic < best_aic:
                    best_aic, best_drop = aic, feat
            except Exception:
                pass
        if best_drop:
            current.remove(best_drop)
            current_aic = best_aic
            improved = True
            if verbose:
                print(f"      drop '{best_drop}' → AIC={best_aic:.1f}")
    return current


def rolling_cv_select(mat_df, train_end, fold_starts,
                       offset_col='km_primary', verbose=False):
    """
    Rolling-origin CV to select features for Split B.
    For each fold year F: train on all data up to F, predict F+1.
    Returns the feature set with lowest mean CV MAE.
    """
    print(f"    Rolling-origin CV with {len(fold_starts)} folds "
          f"({fold_starts[0]}-{fold_starts[-1]})...")

    # Step 1: select features on full training block using AIC
    full_train = mat_df[mat_df['year'] <= train_end]
    selected   = stepwise_aic(full_train, ALL_FEATURES, verbose=False)
    print(f"    AIC-selected features ({len(selected)}): {selected}")

    # Step 2: estimate CV MAE for selected vs full feature set
    cv_mae_selected, cv_mae_full = [], []

    for fold_start in fold_starts:
        cv_train = mat_df[mat_df['year'] <= fold_start]
        cv_val   = mat_df[mat_df['year'] == fold_start + 1]
        if len(cv_val) == 0 or cv_val['break_count'].sum() == 0:
            continue
        try:
            # Selected features
            r_sel  = fit_negbinom(cv_train, selected, offset_col)
            p_sel, _, _ = predict_negbinom(r_sel, cv_val, selected, offset_col)
            cv_mae_selected.append(np.mean(np.abs(cv_val['break_count'].values - p_sel)))

            # Full features
            r_full = fit_negbinom(cv_train, ALL_FEATURES, offset_col)
            p_full, _, _ = predict_negbinom(r_full, cv_val, ALL_FEATURES, offset_col)
            cv_mae_full.append(np.mean(np.abs(cv_val['break_count'].values - p_full)))
        except Exception:
            pass

    mae_sel  = np.mean(cv_mae_selected)  if cv_mae_selected  else np.nan
    mae_full = np.mean(cv_mae_full) if cv_mae_full else np.nan
    print(f"    CV MAE — selected: {mae_sel:.3f}  full: {mae_full:.3f}")

    best = selected if mae_sel <= mae_full else ALL_FEATURES
    print(f"    → Using {'selected' if best is selected else 'full'} features for Split B")
    return best


# ── 3. M0 BASELINE ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("M0 — Seasonal naive baseline")
print("=" * 60)

m0 = {}
for mat in ['CI', 'DI', 'PVC']:
    mat_df = df[df['material'] == mat]

    # Split A baseline
    trA = mat_df[mat_df['year'] <= A_TRAIN_END]
    vaA = mat_df[(mat_df['year'] >= A_VAL_START) & (mat_df['year'] <= A_VAL_END)]
    teA = mat_df[mat_df['year'] >= A_TEST_START]
    means_A = trA.groupby('month_num')['break_count'].mean()

    # Split B baseline
    trB = mat_df[mat_df['year'] <= B_TRAIN_END]
    teB = mat_df[mat_df['year'] >= B_TEST_START]
    means_B = trB.groupby('month_num')['break_count'].mean()

    m0[mat] = {
        'means_A': means_A, 'means_B': means_B,
        'val_A_mae':  float(np.mean(np.abs(vaA['break_count'].values - vaA['month_num'].map(means_A).values))),
        'test_A_mae': float(np.mean(np.abs(teA['break_count'].values - teA['month_num'].map(means_A).values))),
        'test_B_mae': float(np.mean(np.abs(teB['break_count'].values - teB['month_num'].map(means_B).values))),
        'teA': teA, 'teB': teB, 'vaA': vaA,
    }
    print(f"  {mat}: val_A_MAE={m0[mat]['val_A_mae']:.3f}  "
          f"test_A_MAE={m0[mat]['test_A_mae']:.3f}  "
          f"test_B_MAE={m0[mat]['test_B_mae']:.3f}")


# ── 4. MAIN MODELLING LOOP ────────────────────────────────────────────────────

all_results  = []
all_preds    = {}
all_coefs    = {}

for mat in ['CI', 'DI']:
    mat_df = df[df['material'] == mat].copy()

    print(f"\n{'='*60}")
    print(f"Material: {mat}")
    print('='*60)

    # ── SPLIT A ───────────────────────────────────────────────────────────────
    print("\n  --- Split A (train 1997-2016, val 2017-2021, test 2022-2025) ---")

    trA = mat_df[mat_df['year'] <= A_TRAIN_END]
    vaA = mat_df[(mat_df['year'] >= A_VAL_START) & (mat_df['year'] <= A_VAL_END)]
    teA = mat_df[mat_df['year'] >= A_TEST_START]

    base_val_A  = m0[mat]['val_A_mae']
    base_test_A = m0[mat]['test_A_mae']

    # M1 Split A
    try:
        m1A = fit_negbinom(trA, ALL_FEATURES)
        p_v, lo_v, hi_v = predict_negbinom(m1A, vaA, ALL_FEATURES)
        met_m1A_val = monthly_metrics(vaA['break_count'].values, p_v, lo_v, hi_v, base_val_A)
        met_m1A_val['MAE_annual'] = annual_count_mae(vaA['break_count'].values, p_v, vaA['year'].values)
        print(f"  M1 val:  MAE_mo={met_m1A_val['MAE_monthly']}  "
              f"MAE_ann={met_m1A_val['MAE_annual']}  MASE={met_m1A_val['MASE']}")
    except Exception as e:
        print(f"  M1 Split A failed: {e}")
        m1A = None

    # M2 Split A (stepwise on training data)
    print(f"  Stepwise AIC selection (Split A)...")
    selA = stepwise_aic(trA, ALL_FEATURES, verbose=True)
    print(f"  Selected {len(selA)} features: {selA}")

    try:
        m2A = fit_negbinom(trA, selA)
        p_v, lo_v, hi_v = predict_negbinom(m2A, vaA, selA)
        met_m2A_val = monthly_metrics(vaA['break_count'].values, p_v, lo_v, hi_v, base_val_A)
        met_m2A_val['MAE_annual'] = annual_count_mae(vaA['break_count'].values, p_v, vaA['year'].values)
        print(f"  M2 val:  MAE_mo={met_m2A_val['MAE_monthly']}  "
              f"MAE_ann={met_m2A_val['MAE_annual']}  MASE={met_m2A_val['MASE']}")

        # Choose best Split A model
        best_A_features = selA if met_m2A_val['MASE'] <= met_m1A_val['MASE'] else ALL_FEATURES
        best_A_name     = 'M2' if met_m2A_val['MASE'] <= met_m1A_val['MASE'] else 'M1'

        # Refit on train+val, evaluate test
        trva_A = mat_df[mat_df['year'] <= A_VAL_END]
        best_A_refit = fit_negbinom(trva_A, best_A_features)
        p_t, lo_t, hi_t = predict_negbinom(best_A_refit, teA, best_A_features)

        met_A_test = monthly_metrics(teA['break_count'].values, p_t, lo_t, hi_t, base_test_A)
        met_A_test['MAE_annual'] = annual_count_mae(teA['break_count'].values, p_t, teA['year'].values)

        # Rate-based MAE
        pred_A_df = teA.copy()
        pred_A_df['predicted'] = p_t
        mae_rate_A, ann_A = annual_failure_rate_mae(pred_A_df)
        met_A_test['MAE_rate'] = mae_rate_A

        print(f"\n  Split A TEST ({best_A_name}): "
              f"MAE_mo={met_A_test['MAE_monthly']}  "
              f"MAE_ann={met_A_test['MAE_annual']}  "
              f"MAE_rate={mae_rate_A}  "
              f"MASE={met_A_test['MASE']}  "
              f"Cov90={met_A_test['Coverage_90']}%  "
              f"Bias={met_A_test['Bias']}")

        print(f"  Annual failure rates (Split A test):")
        print(ann_A[['year','actual_breaks','predicted_breaks',
                      'actual_rate','predicted_rate']].round(4).to_string(index=False))

        # Store
        pred_A_df['lower_90'] = lo_t
        pred_A_df['upper_90'] = hi_t
        pred_A_df['residual'] = teA['break_count'].values - p_t
        all_preds[f'{mat}_A'] = pred_A_df

        coef_A = pd.DataFrame({
            'feature': best_A_refit.params.index,
            'coefficient': best_A_refit.params.values,
            'std_err': best_A_refit.bse.values,
            'p_value': best_A_refit.pvalues.values,
            'significant': best_A_refit.pvalues.values < 0.05
        })
        all_coefs[f'{mat}_A'] = coef_A

        all_results.append({
            'material': mat, 'split': 'A', 'model': best_A_name,
            'train_years': '1997-2016', 'test_years': '2022-2025',
            **met_A_test
        })

    except Exception as e:
        print(f"  M2/test Split A failed: {e}")

    # ── SPLIT B ───────────────────────────────────────────────────────────────
    print(f"\n  --- Split B (train 1997-2020, test 2021-2025) ---")

    trB = mat_df[mat_df['year'] <= B_TRAIN_END]
    teB = mat_df[mat_df['year'] >= B_TEST_START]
    base_test_B = m0[mat]['test_B_mae']

    # Feature selection via rolling CV
    selB = rolling_cv_select(mat_df, B_TRAIN_END, B_CV_FOLD_STARTS)

    try:
        m2B = fit_negbinom(trB, selB)
        p_t, lo_t, hi_t = predict_negbinom(m2B, teB, selB)

        met_B_test = monthly_metrics(teB['break_count'].values, p_t, lo_t, hi_t, base_test_B)
        met_B_test['MAE_annual'] = annual_count_mae(teB['break_count'].values, p_t, teB['year'].values)

        # Rate-based MAE
        pred_B_df = teB.copy()
        pred_B_df['predicted'] = p_t
        mae_rate_B, ann_B = annual_failure_rate_mae(pred_B_df)
        met_B_test['MAE_rate'] = mae_rate_B

        print(f"\n  Split B TEST: "
              f"MAE_mo={met_B_test['MAE_monthly']}  "
              f"MAE_ann={met_B_test['MAE_annual']}  "
              f"MAE_rate={mae_rate_B}  "
              f"MASE={met_B_test['MASE']}  "
              f"Cov90={met_B_test['Coverage_90']}%  "
              f"Bias={met_B_test['Bias']}")

        print(f"  Annual failure rates (Split B test):")
        print(ann_B[['year','actual_breaks','predicted_breaks',
                      'actual_rate','predicted_rate']].round(4).to_string(index=False))

        # Store
        pred_B_df['lower_90'] = lo_t
        pred_B_df['upper_90'] = hi_t
        pred_B_df['residual'] = teB['break_count'].values - p_t
        all_preds[f'{mat}_B'] = pred_B_df

        coef_B = pd.DataFrame({
            'feature': m2B.params.index,
            'coefficient': m2B.params.values,
            'std_err': m2B.bse.values,
            'p_value': m2B.pvalues.values,
            'significant': m2B.pvalues.values < 0.05
        })
        all_coefs[f'{mat}_B'] = coef_B

        all_results.append({
            'material': mat, 'split': 'B', 'model': 'M2_rollingCV',
            'train_years': '1997-2020', 'test_years': '2021-2025',
            **met_B_test
        })

    except Exception as e:
        print(f"  Split B failed: {e}")


# ── 5. PVC BASELINE ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("PVC — Seasonal baseline only (insufficient test events)")
print('='*60)
pvc = df[df['material']=='PVC']
teA_pvc = pvc[pvc['year'] >= A_TEST_START]
teB_pvc = pvc[pvc['year'] >= B_TEST_START]
print(f"  Split A test breaks: {teA_pvc['break_count'].sum()} — too few for climate model")
print(f"  Split B test breaks: {teB_pvc['break_count'].sum()} — too few for climate model")


# ── 6. SUMMARY ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print('='*60)
results_df = pd.DataFrame(all_results)
print(results_df[['material','split','model','train_years','test_years',
                   'MAE_monthly','MAE_annual','MAE_rate','MASE',
                   'Coverage_90','Bias']].to_string(index=False))

print(f"\n  MAE_rate = breaks/km/year (comparable to Khashei et al. 2024: 0.040-0.192)")
print(f"  MASE < 1 = beats seasonal naive baseline")
print(f"  If Split B MAE < Split A MAE: more training data helped")
print(f"  If similar: model is stable regardless of split choice")


# ── 7. SAVE ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Saving outputs")
print('='*60)

results_df.to_csv(os.path.join(script_dir, 'model_results_both_splits.csv'), index=False)
print("  Saved: model_results_both_splits.csv")

for key, coef_df in all_coefs.items():
    mat, split = key.split('_')
    fname = f'coefficients_{mat.lower()}_split{split}.csv'
    coef_df.to_csv(os.path.join(script_dir, fname), index=False)
    print(f"  Saved: {fname}")

for key, pred_df in all_preds.items():
    mat, split = key.split('_')
    fname = f'predictions_{mat.lower()}_split{split}.csv'
    pred_df[['material','month_str','year','month_num','break_count',
             'km_primary','predicted','lower_90','upper_90','residual']
            ].to_csv(os.path.join(script_dir, fname), index=False)
    print(f"  Saved: {fname}")

print("\n" + "="*60)
print("DONE")
print("="*60)
