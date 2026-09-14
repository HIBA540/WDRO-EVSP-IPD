"""KPIs OOS : E_OOS, CVaR90, Reliability, ΔJ%."""
import numpy as np

from config.settings import cfg
from src.optimization.oos_recourse import evaluate_oos_recourse


def compute_e_oos(costs: np.ndarray) -> float:
    valid = np.isfinite(costs)
    if not valid.any():
        return np.nan
    return float(np.mean(costs[valid]))


def compute_cvar90(costs: np.ndarray) -> float:
    valid = np.isfinite(costs)
    if not valid.any():
        return np.nan
    co_sorted = np.sort(costs[valid])
    n = len(co_sorted)
    n_tail = int(np.ceil(0.10 * n))
    if n_tail == 0:
        return float(co_sorted[-1])
    tail = co_sorted[-n_tail:]
    return float(np.mean(tail))


def compute_reliability(costs_method: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(costs_method)
    if not valid.any() or not np.isfinite(threshold):
        return np.nan
    return float(np.mean(costs_method[valid] <= threshold))


def oos_cost(X_vals_or_P_vals, vd_or_ev_oos, Pi_oos, ce_oos, g_max_oos, **kwargs):
    u_peak_star = kwargs.get('u_peak_star',
                              float(np.max(X_vals_or_P_vals.sum(axis=0)))
                              if hasattr(X_vals_or_P_vals, 'sum') else 0.0)
    result = evaluate_oos_recourse(
        X_star=X_vals_or_P_vals, u_peak_star=u_peak_star,
        Pi_oos=Pi_oos, ce_oos=ce_oos, ev_oos=vd_or_ev_oos,
        g_max_oos=g_max_oos,
        include_peak_in_cost=kwargs.get('include_peak_in_cost', True),
        verbose=False)
    return result['costs']


def compute_operational_kpis(X_vals, ev_oos, Pi_oos, ce_oos, g_max_oos,
                              u_peak_star, **kwargs):
    result = evaluate_oos_recourse(
        X_star=X_vals, u_peak_star=u_peak_star, Pi_oos=Pi_oos, ce_oos=ce_oos,
        ev_oos=ev_oos, g_max_oos=g_max_oos,
        include_peak_in_cost=kwargs.get('include_peak_in_cost', True), verbose=False)
    return dict(
        E_OOS=result['E_OOS'], CVaR90=result['CVaR90'], reliability=result['reliability'],
        peak_power_kw=result['peak_realized_mean'],
        res_selfconsumption_kwh=result['res_used_mean'],
        n_infeasible=result['n_infeasible'], n_scenarios=result['n_scenarios'],
    )


def bootstrap_ci(values, n_bootstrap=300, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    boot_means = np.array([np.mean(rng.choice(v, len(v), replace=True))
                            for _ in range(n_bootstrap)])
    return (float(np.percentile(boot_means, (alpha / 2) * 100)),
            float(np.percentile(boot_means, (1 - alpha / 2) * 100)))


def compute_calibration_metrics_v2(oos_costs, J_IS, rho_used, calib_time,
                                     solve_time, n_scenarios,
                                     target_reliability=0.90,
                                     J_SP_ref=None, threshold_sp_oos=None):
    co = np.asarray(oos_costs, float)
    co = co[np.isfinite(co)]
    if len(co) == 0:
        return {k: np.nan for k in ['E_OOS', 'CVaR90', 'reliability',
                                     'composite_score', 'delta_J_pct']}
    metrics = {}
    metrics['E_OOS'] = compute_e_oos(co)
    metrics['CVaR90'] = compute_cvar90(co)
    if threshold_sp_oos is not None and not np.isnan(threshold_sp_oos):
        metrics['reliability'] = compute_reliability(co, threshold_sp_oos)
    elif not np.isnan(J_IS):
        metrics['reliability'] = compute_reliability(co, J_IS)
    else:
        metrics['reliability'] = np.nan
    metrics['J_IS'] = J_IS
    metrics['rho_used'] = rho_used
    metrics['calib_time'] = calib_time
    metrics['solve_time'] = solve_time

    if J_SP_ref is not None and not np.isnan(J_SP_ref) and J_SP_ref > 0:
        metrics['delta_J_pct'] = 100.0 * (metrics['E_OOS'] - J_SP_ref) / J_SP_ref
    else:
        metrics['delta_J_pct'] = np.nan

    scores = []
    rel = metrics['reliability']
    if not np.isnan(rel):
        if rel >= target_reliability:
            rel_score = 1.0 - max(0.0, rel - 0.98) / 0.02
        else:
            rel_score = max(0.0, rel / target_reliability)
        scores.append(float(np.clip(rel_score, 0.0, 1.0)))
    dj = metrics.get('delta_J_pct', np.nan)
    if not np.isnan(dj):
        if dj < 0:
            cost_score = 1.0
        elif dj <= 5:
            cost_score = 1.0 - dj / 5.0 * 0.2
        elif dj <= 30:
            cost_score = 0.8
        elif dj <= 100:
            cost_score = 0.8 - (dj - 30) / 70 * 0.6
        else:
            cost_score = 0.0
        scores.append(float(np.clip(cost_score, 0.0, 1.0)))
    metrics['composite_score'] = float(np.mean(scores)) if scores else np.nan
    return metrics