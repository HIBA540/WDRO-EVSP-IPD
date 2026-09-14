"""Tests statistiques : stationnarité, dimension intrinsèque, Wilcoxon."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.tsa.stattools import adfuller
from sklearn.neighbors import NearestNeighbors

from config.settings import CALIBRATION_METHODS


def run_stationarity_tests(price_mat, res_mat, verbose=True):
    N_days, T = price_mat.shape
    adf_p_price = adfuller(price_mat.flatten(), maxlag=24, autolag='AIC')[1]
    adf_p_res = adfuller(res_mat.flatten(), maxlag=24, autolag='AIC')[1]
    if verbose:
        print(f"\n  [Stationarity] ADF prix: p={adf_p_price:.4f}  ADF RES: p={adf_p_res:.4f}")
    return {'adf_price_p': float(adf_p_price), 'adf_res_p': float(adf_p_res)}


def estimate_intrinsic_dimension(Xi_hist, verbose=True):
    N, d = Xi_hist.shape
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean').fit(Xi_hist)
    dists, _ = nn.kneighbors(Xi_hist)
    r = dists[:, 2] / (dists[:, 1] + 1e-12)
    r = r[r > 1.0]
    if len(r) == 0:
        return min(d, 13.0)
    log_r = np.log(r)
    log_r = log_r[np.abs(log_r - np.mean(log_r)) < 3 * np.std(log_r)]
    if len(log_r) < 10:
        d_eff = d
    else:
        d_eff = -1.0 / (np.mean(log_r) + 1e-12)
    d_eff = float(np.clip(d_eff, 2, d))
    if verbose:
        print(f"\n  [TwoNN] d_eff = {d_eff:.1f} (d_nominal={d})")
    return d_eff


def run_wilcoxon_tests(all_res, calib_results, out_dir):
    methods = list(calib_results.keys())
    ipd = 'rho_fics'
    rows = []
    for mk in methods:
        if mk == ipd:
            continue
        e_ipd, e_mk = [], []
        c_ipd, c_mk = [], []
        for cname, cr in all_res.items():
            if ipd in cr and mk in cr:
                e_ipd.append(cr[ipd].get('E_OOS', np.nan))
                e_mk.append(cr[mk].get('E_OOS', np.nan))
                c_ipd.append(cr[ipd].get('CVaR90', np.nan))
                c_mk.append(cr[mk].get('CVaR90', np.nan))
        e_ipd = np.array(e_ipd); e_mk = np.array(e_mk)
        c_ipd = np.array(c_ipd); c_mk = np.array(c_mk)
        valid_e = np.isfinite(e_ipd) & np.isfinite(e_mk)
        valid_c = np.isfinite(c_ipd) & np.isfinite(c_mk)
        try:
            _, p_e = wilcoxon(e_ipd[valid_e], e_mk[valid_e]) if valid_e.sum() > 5 else (np.nan, np.nan)
        except Exception:
            p_e = np.nan
        try:
            _, p_c = wilcoxon(c_ipd[valid_c], c_mk[valid_c]) if valid_c.sum() > 5 else (np.nan, np.nan)
        except Exception:
            p_c = np.nan
        wins_e = float(np.mean(e_ipd[valid_e] < e_mk[valid_e])) if valid_e.sum() > 0 else np.nan
        wins_c = float(np.mean(c_ipd[valid_c] < c_mk[valid_c])) if valid_c.sum() > 0 else np.nan
        rows.append({
            'Comparison': f'IPD vs {CALIBRATION_METHODS.get(mk, {}).get("label", mk)}',
            'IPD wins E_OOS (%)': f"{100*wins_e:.0f}" if np.isfinite(wins_e) else "nan",
            'p-value E_OOS': f"{p_e:.4f}" if np.isfinite(p_e) else "nan",
            'IPD wins CVaR90 (%)': f"{100*wins_c:.0f}" if np.isfinite(wins_c) else "nan",
            'p-value CVaR90': f"{p_c:.4f}" if np.isfinite(p_c) else "nan",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'wilcoxon_tests.csv'), index=False)
    df.to_latex(os.path.join(out_dir, 'table_wilcoxon.tex'),
                index=False, escape=False,
                caption='Wilcoxon signed-rank tests.', label='tab:wilcoxon')
    print(f"  [Tests] Wilcoxon saved.")
    return df