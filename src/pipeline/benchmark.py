"""Benchmark : calibration + évaluation OOS."""
import time
import numpy as np

from config.settings import (
    cfg, CALIBRATION_METHODS, ALL_CASES,
)
from src.calibration.ek import calibrate_rho_ek
from src.calibration.bai import calibrate_rho_bai
from src.calibration.gotoh import calibrate_rho_gotoh
from src.calibration.ipd import calibrate_rho_ipd_from_pipeline
from src.evaluation.kpis import (
    oos_cost, compute_e_oos, compute_cvar90, bootstrap_ci,
    compute_operational_kpis,
)


def run_calibration_benchmark(vd, Pi_IS, ce_IS, w_IS, Xi_IS, D_sp,
                               stationarity_results, Xi_forecast_norm, d_eff,
                               forecast_engine=None, proxy_costs=None,
                               price_mat=None, res_mat=None,
                               out_dir='.', normalizer=None):
    print("\n" + "=" * 70)
    print("  BENCHMARK: 4 MÉTHODES DE CALIBRATION (EK, Bai, Gotoh, IPD)")
    print("  ===========================================================")

    calib_results = {}

    t0 = time.perf_counter()
    calib_results['rho_ek'] = calibrate_rho_ek(Xi_IS, d_eff=d_eff, verbose=True)
    calib_results['rho_ek']['calib_time'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    calib_results['rho_bai'] = calibrate_rho_bai(Xi_IS, verbose=True)
    calib_results['rho_bai']['calib_time'] = time.perf_counter() - t0

    print("\n  [GOTOH21] Calibration MV-Bootstrap (grille complète, k_eval=10)...")
    t0 = time.perf_counter()
    try:
        calib_results['rho_gotoh'] = calibrate_rho_gotoh(
            Xi_IS, vd, Pi_IS, ce_IS, w_IS, D_sp, verbose=True)
    except Exception as e:
        print(f"  *** WARN Gotoh: {e} -> fallback rho=0.35")
        calib_results['rho_gotoh'] = {'rho': 0.35, 'label': 'Gotoh (2021)',
                                       'family': 'MV-Bootstrap Frontier', 'calib_time': 0.}
    calib_results['rho_gotoh']['calib_time'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if forecast_engine is not None and proxy_costs is not None:
        calib_results['rho_fics'] = calibrate_rho_ipd_from_pipeline(
            Xi_IS, forecast_engine, beta=cfg.IPD_BETA, verbose=True,
            normalizer=normalizer, Xi_forecast_norm=Xi_forecast_norm)
    else:
        print("  [WARN] Données insuffisantes pour IPD -> fallback rho=0.70")
        calib_results['rho_fics'] = {'rho': 0.70, 'label': 'IPD (fallback)',
                                      'family': 'IPD-Principled', 'calib_time': 0.}
    calib_results['rho_fics']['calib_time'] = time.perf_counter() - t0

    print("\n  ── Résumé des ρ calibrés ──")
    for mk, mv in calib_results.items():
        print(f"  {CALIBRATION_METHODS.get(mk, {}).get('label', mk):<25}: ρ={mv['rho']:.5f}")

    return calib_results


def evaluate_benchmark_oos(wdro_plans, data_loader, calib_results, out_dir, r_sp=None):
    print("\n" + "=" * 70)
    print("  OOS EVALUATION: 4 CALIBRATIONS × TOUS LES CAS")
    print("=" * 70)

    all_res = {}
    all_op_kpis = {}

    for cname, cc in ALL_CASES.items():
        print(f"\n  ── {cname}: {cc.get('desc', '')} ──")
        Pi_oos, ce_oos, ev_oos = data_loader.generate_OOS_for_case(
            cc, cfg.S_OOS, seed=cfg.SEED + abs(hash(cname)) % 10000)
        case_res = {}
        case_op = {}

        raw_costs = {}
        for mk, mv in wdro_plans.items():
            if mv is None:
                raw_costs[mk] = None
                continue
            co = oos_cost(mv['X_vals'], ev_oos, Pi_oos, ce_oos, cc['g_max_oos'],
                          u_peak_star=mv.get('u_peak', 0.0))
            raw_costs[mk] = co

        all_costs_concat = np.concatenate([c[np.isfinite(c)] for c in raw_costs.values()
                                            if c is not None and np.isfinite(c).any()])
        threshold_sp_case = (float(np.percentile(all_costs_concat, cfg.REL_ALPHA * 100))
                             if len(all_costs_concat) > 0 else np.nan)

        best_J = np.inf
        for mk, co in raw_costs.items():
            if co is not None:
                m = compute_e_oos(co)
                if np.isfinite(m) and m < best_J:
                    best_J = m
        if not np.isfinite(best_J):
            best_J = np.nan

        for mk, mv in wdro_plans.items():
            if mv is None or raw_costs[mk] is None:
                case_res[mk] = {k: np.nan for k in
                                ['E_OOS', 'CVaR90', 'reliability', 'J_IS',
                                 'composite_score', 'delta_J_pct']}
                case_res[mk].update({
                    'rho': calib_results[mk]['rho'],
                    'label': CALIBRATION_METHODS.get(mk, {}).get('label', mk),
                    'costs': None,
                })
                case_op[mk] = {}
                continue

            co = raw_costs[mk]
            J_m = mv.get('J_energy_cost', np.nan)
            ci_lo, ci_hi = bootstrap_ci(co, n_bootstrap=300)

            E_OOS = compute_e_oos(co)
            CVaR90 = compute_cvar90(co)

            if np.isfinite(threshold_sp_case):
                reliability = float(np.mean(co[np.isfinite(co)] <= threshold_sp_case))
            else:
                reliability = np.nan

            delta_J = (100.0 * (E_OOS - best_J) / best_J
                       if np.isfinite(best_J) and best_J > 0 else np.nan)

            score_parts = []
            if np.isfinite(reliability):
                score_parts.append(reliability)
            if np.isfinite(delta_J):
                score_parts.append(max(0.0, 1.0 - abs(delta_J) / 100.0))
            score = float(np.mean(score_parts)) if score_parts else np.nan

            case_res[mk] = {
                'E_OOS': E_OOS, 'CVaR90': CVaR90,
                'E_OOS_ci_lo': ci_lo, 'E_OOS_ci_hi': ci_hi,
                'reliability': reliability,
                'J_IS': J_m, 'rho': calib_results[mk]['rho'],
                'label': CALIBRATION_METHODS.get(mk, {}).get('label', mk),
                'family': CALIBRATION_METHODS.get(mk, {}).get('family', ''),
                'delta_J_pct': delta_J,
                'composite_score': score,
                'costs': co,
            }

            op_kpis = compute_operational_kpis(
                mv['X_vals'], ev_oos, Pi_oos, ce_oos, cc['g_max_oos'],
                u_peak_star=mv.get('u_peak', 0.0))
            case_op[mk] = op_kpis

        all_res[cname] = case_res
        all_op_kpis[cname] = case_op

        print(f"  {'Méthode':<25} {'ρ':>8} {'E_OOS':>8} {'CVaR90':>8} {'Rel':>7} {'ΔJ%':>7}")
        print('  ' + '-' * 70)
        for mk in wdro_plans:
            mr = case_res.get(mk, {})
            marker = ' ★' if mk == 'rho_fics' else ''
            print(f"  {mr.get('label', mk):<25} {mr.get('rho', 0.):>8.5f} "
                  f"{mr.get('E_OOS', np.nan):>8.1f} {mr.get('CVaR90', np.nan):>8.1f} "
                  f"{mr.get('reliability', np.nan):>7.3f} "
                  f"{mr.get('delta_j_pct', np.nan):>7.1f}{marker}")

    return all_res, all_op_kpis