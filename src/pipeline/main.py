"""Pipeline principal v10 complet."""
import os
import sys
import time
import datetime
import numpy as np

from config.settings import cfg, ALL_CASES, CALIBRATION_METHODS, D_EFF

from src.data.loader import RealDataLoader
from src.data.fleet_cache import ScenarioFleetCache
from src.forecast.engine import ForecastEngineV4
from src.optimization.distance import build_distance_matrix_exact
from src.optimization.wdro_model import (
    build_wdro_direct_model, solve_wdro_for_rho, set_fleet_cache,
)
from src.evaluation.statistics import (
    run_stationarity_tests, estimate_intrinsic_dimension, run_wilcoxon_tests,
)
from src.pipeline.benchmark import run_calibration_benchmark, evaluate_benchmark_oos
from src.visualization.figures import make_all_figures
from src.export.excel import export_operational_kpis_excel
from src.export.json_export import export_json
from src.export.csv_export import export_csv
from src.export.latex_export import export_latex_tables


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def make_output_tree(root):
    subdirs = ['excel', 'tables', 'figures', 'json', 'logs']
    for sd in subdirs:
        os.makedirs(os.path.join(root, sd), exist_ok=True)
    return {sd: os.path.join(root, sd) for sd in subdirs}


def calibrate_delta_star(fleet_cache, K, n_days_sample=60):
    rng = np.random.RandomState(cfg.SEED)
    days = rng.choice(len(fleet_cache.ev_dates),
                       min(n_days_sample, len(fleet_cache.ev_dates)), replace=False)
    deltas = []
    for j in days:
        vd_j = fleet_cache.get(int(j), K)
        deltas.append(vd_j['delta_star'])
    d_star = float(np.clip(np.mean(deltas) + np.std(deltas),
                            cfg.SOC_TOL_MIN, cfg.SOC_TOL_MAX))
    print(f"  [delta*] = {d_star:.4f}")
    return d_star


def main_v10_full(data_path_root='data', output_root='output_v10_full'):
    t_start = time.perf_counter()
    root = os.path.join(os.getcwd(), output_root)
    tree = make_output_tree(root)

    log_path = os.path.join(tree['logs'],
                             f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 80)
    print("  WDRO-EVSP v10 — PIPELINE COMPLET")
    print("=" * 80)
    print(f"  Output root : {root}")
    print(f"  Log file    : {log_path}")
    print("=" * 80)

    print("\n[0] Chargement des données...")
    DATA = RealDataLoader(cfg)

    print("\n[0b] Fleet cache + delta*...")
    FLEET_CACHE = ScenarioFleetCache(DATA.ev_df, DATA.ev_dates, seed=cfg.SEED)
    set_fleet_cache(FLEET_CACHE)
    delta_star = calibrate_delta_star(FLEET_CACHE, K=cfg.PLAN_N_EV)

    print("\n[1] Données IS...")
    Pi_IS, ce_IS, ev_IS, w_IS, Xi_IS = DATA.get_IS_planning(S=cfg.S_IS, seed=cfg.SEED)

    print("\n[2] Distance matrix...")
    D_sp = build_distance_matrix_exact(Xi_IS)

    print("\n[3] Dimension intrinsèque...")
    d_eff = estimate_intrinsic_dimension(Xi_IS, verbose=True)

    print("\n[4] ForecastEngineV4...")
    forecast_engine = ForecastEngineV4(DATA.Xi_hist_raw, DATA._proxy_hist,
                                        DATA.ev_dates,
                                        min_train_window=cfg.IPD_MIN_TRAIN_WINDOW)
    forecast_quality = forecast_engine.evaluate_forecast_quality(verbose=True)
    Xi_forecast_raw, _ = forecast_engine.generate_forecast_scenarios(
        K=cfg.RHO_K_FORECAST, seed=cfg.SEED)
    Xi_forecast_norm = DATA._normalizer.transform(Xi_forecast_raw)

    print("\n[5] Stationnarité...")
    stationarity_results = run_stationarity_tests(DATA.price_mat, DATA.res_mat, verbose=True)

    print("\n[6] Calibration benchmark...")
    calib_results = run_calibration_benchmark(
        ev_IS, Pi_IS, ce_IS, w_IS, Xi_IS, D_sp,
        stationarity_results, Xi_forecast_norm, d_eff,
        forecast_engine=forecast_engine,
        proxy_costs=DATA._proxy_hist,
        price_mat=DATA.price_mat, res_mat=DATA.res_mat,
        out_dir=tree['excel'],
        normalizer=DATA._normalizer)

    print("\n[7] Résolution WDRO...")
    built_final = build_wdro_direct_model(
        Pi_IS, ce_IS, cfg.PLAN_G_MAX, D_sp, w_IS,
        day_indices=list(range(cfg.S_IS)), K=cfg.PLAN_N_EV,
        fleet_cache=FLEET_CACHE, method_label='final', verbose=True)
    wdro_plans = {}
    for mk, mv in calib_results.items():
        try:
            r = solve_wdro_for_rho(built_final, mv['rho'], verbose=True)
            wdro_plans[mk] = r
            print(f"    {mk}: rho={mv['rho']:.4f}  J_IS={r['J_energy_cost']:.1f}EUR  "
                  f"t={r['exec_time']:.1f}s")
        except Exception as e:
            print(f"    ERREUR {mk}: {e}")
            wdro_plans[mk] = None

    print("\n[8] Évaluation OOS + KPIs...")
    all_res, all_op_kpis = evaluate_benchmark_oos(
        wdro_plans, DATA, calib_results, tree['excel'], r_sp=None)

    print("\n[9] Exports...")
    export_operational_kpis_excel(all_res, all_op_kpis, calib_results, tree['excel'])
    export_json(calib_results, all_res, all_op_kpis, forecast_quality,
                d_eff, stationarity_results,
                os.path.join(tree['json'], 'results_raw.json'))
    export_csv(all_res, calib_results, tree['tables'])
    export_latex_tables(all_res, calib_results, tree['tables'])
    run_wilcoxon_tests(all_res, calib_results, tree['tables'])
    make_all_figures(calib_results, all_res, all_op_kpis, Xi_IS,
                     forecast_quality, tree['figures'])

    print("\n" + "=" * 80)
    print("  RÉSUMÉ FINAL")
    print("=" * 80)
    print(f"\n  {'Méthode':<25} {'ρ':>8} {'E_OOS':>9} {'CVaR90':>10} {'Rel':>8} {'ΔJ%':>7}")
    print('  ' + '-' * 80)
    for mk in calib_results:
        lbl = CALIBRATION_METHODS.get(mk, {}).get('label', mk)
        rho_v = calib_results[mk]['rho']
        e_mean = np.nanmean([all_res[c][mk].get('E_OOS', np.nan) for c in ALL_CASES])
        c_mean = np.nanmean([all_res[c][mk].get('CVaR90', np.nan) for c in ALL_CASES])
        r_mean = np.nanmean([all_res[c][mk].get('reliability', np.nan) for c in ALL_CASES])
        dj_mean = np.nanmean([all_res[c][mk].get('delta_J_pct', np.nan) for c in ALL_CASES])
        marker = ' ★' if mk == 'rho_fics' else ''
        print(f"  {lbl:<25} {rho_v:>8.5f} {e_mean:>9.1f} {c_mean:>10.1f} "
              f"{r_mean:>8.3f} {dj_mean:>7.1f}{marker}")

    t_total = time.perf_counter() - t_start
    print(f"\n  Pipeline terminé en {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"  Résultats dans : {root}")

    sys.stdout = sys.__stdout__
    log_file.close()

    return dict(calib_results=calib_results, wdro_plans=wdro_plans,
                all_res=all_res, all_op_kpis=all_op_kpis,
                forecast_quality=forecast_quality,
                d_eff=d_eff, stationarity_results=stationarity_results,
                Xi_IS=Xi_IS, out_root=root)