"""Calibration IPD (Ours) — Integrated Prospective Distance."""
import numpy as np
from statsmodels.tsa.stattools import acf

from config.settings import cfg


def _w1_1d_empirical(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    all_vals = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, all_vals, side='right') / len(a)
    cdf_b = np.searchsorted(b, all_vals, side='right') / len(b)
    if len(all_vals) < 2:
        return 0.0
    diffs = np.diff(all_vals)
    w1 = float(np.sum(np.abs(cdf_a[:-1] - cdf_b[:-1]) * diffs))
    return w1


def _compute_retro_radius_block(Xi_IS_block, Xi_forecast_block):
    d_b = Xi_IS_block.shape[1]
    w1_per_dim = [_w1_1d_empirical(Xi_IS_block[:, k], Xi_forecast_block[:, k]) for k in range(d_b)]
    return float(np.mean(w1_per_dim)), float(np.max(w1_per_dim))


def _compute_forecast_error_process(forecast_engine, block_idx):
    start, end = block_idx
    lengths = [len(forecast_engine._residuals_oos.get(k, [])) for k in range(start, end)]
    if not lengths or min(lengths) == 0:
        return np.zeros(1)
    n_common = min(lengths)
    errors_per_dim = []
    for k_dim in range(start, end):
        res_k = forecast_engine._residuals_oos[k_dim][:n_common]
        errors_per_dim.append(np.abs(res_k))
    error_matrix = np.column_stack(errors_per_dim)
    col_stds = np.maximum(np.std(error_matrix, axis=0), 1e-10)
    error_matrix_norm = error_matrix / col_stds
    mean_std = float(np.mean(col_stds))
    return np.mean(error_matrix_norm, axis=1) * mean_std


def _compute_kappa_dep_from_error_process(error_process, block_name=''):
    e = np.asarray(error_process, dtype=float)
    e = e[np.isfinite(e)]
    N = len(e)
    if N < 10:
        return 1.0
    e_c = e - e.mean()
    sig_threshold = 1.96 / np.sqrt(N)
    K_max = min(30, N // 5)
    try:
        acf_vals = acf(e_c, nlags=K_max, fft=True)[1:]
    except Exception:
        return 1.0
    running_sum = 0.0
    for k, a in enumerate(acf_vals):
        if abs(float(a)) < sig_threshold and k >= 2:
            break
        running_sum += abs(float(a))
    return float(np.clip(np.sqrt(max(1.0, 1.0 + 2.0 * running_sum)), 1.0, 2.5))


def _compute_prosp_radius_block(error_process, kappa_dep, beta):
    e = np.asarray(error_process, dtype=float)
    e = e[np.isfinite(e) & (e >= 0)]
    N = len(e)
    if N == 0:
        return 0.0
    q_raw = float(np.quantile(e, 1.0 - beta))
    if N < 200:
        p = 1.0 - beta
        f_q = float(np.mean(np.abs(e - q_raw) < 0.1 * (q_raw + 1e-10)) / (0.2 * (q_raw + 1e-10) + 1e-10))
        f_q = max(f_q, 1e-6)
        std_quantile = float(np.sqrt(p * (1 - p) / (N * f_q ** 2 + 1e-10)))
        correction = 1.0 + 1.645 * std_quantile / (q_raw + 1e-10)
        correction = float(np.clip(correction, 1.0, 1.5))
        q_raw = q_raw * correction
    return float(q_raw * kappa_dep)


def _compute_metric_consistent_weights(Xi_IS, blocks):
    per_dim_var = np.var(Xi_IS, axis=0, ddof=1)
    per_dim_var = np.maximum(per_dim_var, 1e-12)
    total_var = float(np.sum(per_dim_var))
    block_weights = {}
    for bname, idx, *_ in blocks:
        block_var = float(np.sum(per_dim_var[idx[0]:idx[1]]))
        block_weights[bname] = float(np.sqrt(block_var / total_var))
    norm = float(np.sqrt(sum(w ** 2 for w in block_weights.values())))
    if norm > 1e-10:
        block_weights = {k: v / norm for k, v in block_weights.items()}
    return block_weights


def calibrate_rho_ipd(Xi_IS, Xi_forecast_norm, forecast_engine,
                       beta=0.05, verbose=True):
    N, D = Xi_IS.shape
    K = Xi_forecast_norm.shape[0]
    blocks = [('price', (0, 5), 'Price (5 dims)', 5),
              ('res', (5, 9), 'RES (4 dims)', 4),
              ('ev', (9, 13), 'EV (4 dims)', 4)]
    if verbose:
        print(f"\n  IPD Calibration — N={N}, K={K}, D={D}, beta={beta:.3f}")
    block_weights = _compute_metric_consistent_weights(Xi_IS, blocks)
    rhos = {}
    block_diagnostics = {}
    for bname, idx, desc, d_b in blocks:
        Xi_IS_b = Xi_IS[:, idx[0]:idx[1]]
        Xi_fc_b = Xi_forecast_norm[:, idx[0]:idx[1]]
        R_retro_mean, R_retro_max = _compute_retro_radius_block(Xi_IS_b, Xi_fc_b)
        error_proc = _compute_forecast_error_process(forecast_engine, idx)
        kappa_dep_b = _compute_kappa_dep_from_error_process(error_proc, block_name=bname)
        R_prosp_b = _compute_prosp_radius_block(error_proc, kappa_dep_b, beta)
        rho_b = R_retro_mean + R_prosp_b
        rhos[bname] = rho_b
        block_diagnostics[bname] = {
            'R_retro': R_retro_mean, 'R_retro_max': R_retro_max,
            'R_prosp': R_prosp_b, 'kappa_dep': kappa_dep_b, 'rho': rho_b,
            'weight': block_weights[bname],
        }
        if verbose:
            print(f"    [{desc}] R_retro={R_retro_mean:.5f}  kappa_dep={kappa_dep_b:.4f}  "
                  f"R_prosp={R_prosp_b:.5f}  rho_b={rho_b:.5f}")
    rho_ipd_sq = sum(block_weights[bname] ** 2 * rhos[bname] ** 2 for bname, _, _, _ in blocks)
    rho_ipd = float(np.sqrt(rho_ipd_sq))
    if verbose:
        print(f"    [Design Rule 1] rho_IPD = {rho_ipd:.5f}")
    return dict(rho=rho_ipd, block_diagnostics=block_diagnostics, block_weights=block_weights,
                rho_price=rhos['price'], rho_res=rhos['res'], rho_ev=rhos['ev'],
                beta=beta, N=N, K_forecast=K, label='IPD (Ours)', family='IPD-Principled')


def calibrate_rho_ipd_from_pipeline(Xi_IS, forecast_engine, beta=0.05, verbose=True,
                                     normalizer=None, Xi_forecast_norm=None):
    if Xi_forecast_norm is None:
        Xi_fc_raw, _ = forecast_engine.generate_forecast_scenarios(
            K=cfg.RHO_K_FORECAST, seed=cfg.SEED)
        Xi_forecast_norm = normalizer.transform(Xi_fc_raw)
    return calibrate_rho_ipd(Xi_IS=Xi_IS, Xi_forecast_norm=Xi_forecast_norm,
                              forecast_engine=forecast_engine, beta=beta, verbose=verbose)