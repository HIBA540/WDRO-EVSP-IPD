"""Calibration Gotoh (2021) — MV-Bootstrap Frontier."""
import time
import numpy as np

from config.settings import cfg
from src.optimization.wdro_model import build_wdro_direct_model, solve_wdro_for_rho, get_fleet_cache
from src.evaluation.kpis import oos_cost


def calibrate_rho_gotoh(Xi_hist_norm, vd, Pi_IS, ce_IS, w_IS, D_sp,
                         rho_grid=None, k_bootstrap=None, lambda_mv=None,
                         verbose=True) -> dict:
    if rho_grid is None:
        rho_grid = list(cfg.RHO_GOTOH_GRID)
    if k_bootstrap is None:
        k_bootstrap = cfg.GOTOH_K_BOOTSTRAP
    if lambda_mv is None:
        lambda_mv = cfg.GOTOH_LAMBDA_MV
    N, d = Xi_hist_norm.shape
    rng = np.random.RandomState(cfg.SEED)
    if verbose:
        print(f"\n  [GOTOH21] N={N} K_boot={k_bootstrap} grid={rho_grid} lambda_MV={lambda_mv}")
    k_actual = min(k_bootstrap, 50)
    bootstrap_indices = [rng.choice(N, N, replace=True) for _ in range(k_actual)]
    k_eval = min(k_actual, 10)

    mu_vals = {rho: [] for rho in rho_grid}
    var_vals = {rho: [] for rho in rho_grid}

    prev_max_scen = cfg.WDRO_MAX_RECOURSE_SCENARIOS
    cfg.WDRO_MAX_RECOURSE_SCENARIOS = min(prev_max_scen, 20)
    try:
        for j in range(k_eval):
            t_j0 = time.perf_counter()
            print(f"    [GOTOH] replica {j+1}/{k_eval}...", flush=True)
            idx_j = bootstrap_indices[j][:min(N, 100)]
            try:
                built_j = build_wdro_direct_model(
                    Pi_IS[:, idx_j], ce_IS[:, idx_j], cfg.PLAN_G_MAX,
                    D_sp[np.ix_(idx_j, idx_j)], np.ones(len(idx_j)) / len(idx_j),
                    day_indices=idx_j.tolist(), K=cfg.PLAN_N_EV,
                    fleet_cache=FLEET_CACHE, method_label=f'GOTOH_boot_{j}', verbose=False)
            except Exception as e:
                print(f"      *** WARN construction echouee replica {j}: {e}", flush=True)
                continue
            for rho in rho_grid:
                t_r0 = time.perf_counter()
                try:
                    r_j = solve_wdro_for_rho(
                        built_j, rho, time_limit=30, mip_gap=0.02, verbose=False)
                    co_orig = oos_cost(r_j['X_vals'], vd, Pi_IS, ce_IS, cfg.PLAN_G_MAX)
                    mu_vals[rho].append(float(np.mean(co_orig)))
                    var_vals[rho].append(float(np.var(co_orig)))
                    print(f"      rho={rho:.3f} -> {time.perf_counter()-t_r0:.1f}s", flush=True)
                except Exception:
                    print(f"      rho={rho:.3f} -> ECHEC ({time.perf_counter()-t_r0:.1f}s)", flush=True)
                    continue
            print(f"    [GOTOH] replica {j+1} termine en {time.perf_counter()-t_j0:.1f}s", flush=True)
    finally:
        cfg.WDRO_MAX_RECOURSE_SCENARIOS = prev_max_scen

    mv_results = {}
    for rho in rho_grid:
        if not mu_vals[rho]:
            mv_results[rho] = dict(mu=np.nan, var=np.nan, mv_score=np.nan)
            continue
        mu_mean = float(np.mean(mu_vals[rho]))
        var_mean = float(np.mean(var_vals[rho]))
        mv_results[rho] = dict(mu=mu_mean, var=var_mean,
                                mv_score=lambda_mv * var_mean + (1.0 - lambda_mv) * mu_mean)
        if verbose:
            print(f"      rho={rho:.3f}: mu={mu_mean:.2f}  var={var_mean:.2f}")

    valid = {r: v for r, v in mv_results.items() if np.isfinite(v.get('mv_score', np.nan))}
    rho_opt = min(valid, key=lambda r: valid[r]['mv_score']) if valid else float(np.median(rho_grid))
    if verbose:
        print(f"    -> rho*_Gotoh = {rho_opt:.5f}")
    return dict(rho=float(rho_opt), mv_results=mv_results, label='Gotoh (2021)',
                family='MV-Bootstrap Frontier', calib_time=0.0)