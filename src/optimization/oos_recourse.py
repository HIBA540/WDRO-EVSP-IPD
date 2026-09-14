"""Recourse OOS : évaluation d'un plan X sur des scénarios out-of-sample."""
import time
import numpy as np
from multiprocessing import Pool
from functools import partial
import gurobipy as gp
from gurobipy import GRB

from config.settings import (
    cfg, CHARGER_POWER, N_CHARGERS, N_L1, N_L2,
    BAT_THRESH_DC, BAT_THRESH_L2,
)


def _ev_bootstrap_to_vd(ev: dict, s: int) -> dict:
    T = cfg.T
    n_ev = ev['tarr'].shape[0]
    ta = ev['tarr'][:, s].astype(int)
    td = ev['tdep'][:, s].astype(int)
    bat = ev['bat'][:, s].astype(float)
    s0 = ev['s0'][:, s].astype(float)
    sd = ev['sdep_raw'][:, s].astype(float)
    pm = ev['pmax'][:, s].astype(float)
    y = ev['y'][:, :, s].astype(int)

    chi = np.zeros((n_ev, N_CHARGERS), dtype=int)
    for v in range(n_ev):
        b = float(bat[v])
        chi[v, :N_L1] = 1
        if b >= BAT_THRESH_L2:
            chi[v, N_L1:N_L1 + N_L2] = 1
        if b >= BAT_THRESH_DC:
            chi[v, N_L1 + N_L2:] = 1

    if n_ev > 0:
        delta_star = float(np.clip(np.mean([
            max(0., 1. - (pm[v] * cfg.ETA * max(1, int(td[v]) - int(ta[v]))) /
                max(sd[v] - s0[v], 0.1))
            for v in range(n_ev)
        ]), cfg.SOC_TOL_MIN, cfg.SOC_TOL_MAX))
    else:
        delta_star = cfg.SOC_TOL_MIN

    return dict(N=n_ev, beta=bat, s0=s0, smin=0.10 * bat, smax=0.95 * bat,
                sdep=sd, tarr=ta, tdep=td, chi=chi, pmax_v=pm, y=y,
                delta_star=delta_star)


def solve_oos_recourse(X_star, vd_oos, price_oos, res_oos, g_max_oos,
                        time_limit=8.0, gap=0.02):
    T = cfg.T
    m = gp.Model("OOS_recourse")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = gap
    m.Params.Threads = 1

    X_fixed = {}
    for b in range(N_CHARGERS):
        for t in range(T):
            xv = float(np.clip(X_star[b, t], 0., float(CHARGER_POWER[b])))
            X_fixed[(b, t)] = m.addVar(lb=xv, ub=xv, name=f"Xfix_{b}_{t}")
    u_peak_diag = m.addVar(lb=0., ub=float(g_max_oos), name="u_peak_oos_diag")
    m.update()

    N = vd_oos['N']
    ds = vd_oos.get('delta_star', cfg.SOC_TOL_MIN)
    z = {(v, b): m.addVar(lb=0., ub=1., vtype=GRB.BINARY, name=f"z_OOS_{v}_{b}")
         for v in range(N) for b in range(N_CHARGERS) if vd_oos['chi'][v, b] == 1}
    Pn = {(v, t): m.addVar(lb=0., name=f"P_OOS_{v}_{t}") for v in range(N) for t in range(T)}
    Paux = {(v, b, t): m.addVar(lb=0., name=f"Pa_OOS_{v}_{b}_{t}")
            for (v, b) in z for t in range(T)}
    sn = {(v, t): m.addVar(lb=0., name=f"soc_OOS_{v}_{t}") for v in range(N) for t in range(T)}
    sd_v = {v: m.addVar(lb=0., name=f"sd_OOS_{v}") for v in range(N)}
    sigma = {v: m.addVar(lb=0., name=f"sigma_OOS_{v}") for v in range(N)}
    delta_soc = {v: m.addVar(lb=0., name=f"dsoc_OOS_{v}") for v in range(N)}
    G = {t: m.addVar(lb=0., ub=g_max_oos, name=f"G_OOS_{t}") for t in range(T)}
    m.update()

    for v in range(N):
        m.addConstr(gp.quicksum(z[v, b] for b in range(N_CHARGERS) if (v, b) in z) <= 1)
    for b in range(N_CHARGERS):
        for t in range(T):
            m.addConstr(gp.quicksum(z[v, b] * int(vd_oos['y'][v, t])
                                     for v in range(N) if (v, b) in z) <= 1)
    for v in range(N):
        m.addConstr(sn[v, 0] == vd_oos['s0'][v] + cfg.ETA * Pn[v, 0])
        for t in range(1, T):
            m.addConstr(sn[v, t] == sn[v, t - 1] + cfg.ETA * Pn[v, t])
        for t in range(T):
            m.addConstr(sn[v, t] >= vd_oos['smin'][v])
            m.addConstr(sn[v, t] <= vd_oos['smax'][v])
            if vd_oos['y'][v, t] == 0:
                m.addConstr(Pn[v, t] == 0)
            else:
                m.addConstr(Pn[v, t] == gp.quicksum(Paux[v, b, t] for b in range(N_CHARGERS) if (v, b) in z))
                m.addConstr(Pn[v, t] <= vd_oos['pmax_v'][v])
    for (v, b) in z:
        for t in range(T):
            if vd_oos['y'][v, t] == 1:
                m.addConstr(Paux[v, b, t] <= CHARGER_POWER[b] * z[v, b])
                m.addConstr(Paux[v, b, t] <= vd_oos['pmax_v'][v] * z[v, b])
            else:
                m.addConstr(Paux[v, b, t] == 0)

    for b in range(N_CHARGERS):
        for t in range(T):
            m.addConstr(
                gp.quicksum(Paux[v, b, t] for v in range(N) if (v, b) in z) <= X_fixed[(b, t)],
                name=f"link_OOS_{b}_{t}")

    for v in range(N):
        active = [t for t in range(T) if vd_oos['y'][v, t] == 1]
        if active:
            m.addConstr(sd_v[v] == vd_oos['s0'][v] + cfg.ETA * gp.quicksum(Pn[v, t] for t in active))
        else:
            m.addConstr(sd_v[v] == vd_oos['s0'][v])
        m.addConstr(sd_v[v] <= vd_oos['smax'][v])
        E_need = max(0., float(vd_oos['sdep'][v]) - float(vd_oos['s0'][v]))
        n_sl = max(1, len(active))
        E_max = float(vd_oos['pmax_v'][v]) * cfg.ETA * n_sl
        delta_v = max(max(0., 1. - E_max / max(E_need, 0.1)) if E_need > 0.01 else 0., ds)
        tgt_v = (1. - delta_v) * float(vd_oos['sdep'][v])
        m.addConstr(delta_soc[v] >= tgt_v - sd_v[v])
        m.addConstr(sigma[v] >= sd_v[v] - float(vd_oos['sdep'][v]))

    for t in range(T):
        lhs = gp.quicksum(Pn[v, t] * int(vd_oos['y'][v, t]) for v in range(N))
        m.addConstr(lhs <= float(res_oos[t]) + G[t])

    rl = cfg.DELTA_RAMP * g_max_oos
    for t in range(T - 1):
        m.addConstr(G[t + 1] - G[t] <= rl)
        m.addConstr(G[t] - G[t + 1] <= rl)

    for t in range(T):
        m.addConstr(u_peak_diag >= G[t])

    cost_expr = (gp.quicksum(float(price_oos[t]) * G[t] for t in range(T)) +
                 cfg.LAM_SOC * gp.quicksum(delta_soc[v] for v in range(N)) +
                 cfg.KAPPA_SLACK * gp.quicksum(sigma[v] for v in range(N)))

    m.setObjective(cost_expr, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return dict(status='infeasible', cost=np.nan, P_vals=None, G_vals=None,
                    slack_vals=None, delta_soc_vals=None, u_peak_realized=np.nan)

    def _sv(var):
        try:
            return var.X
        except Exception:
            return 0.0

    cost_val = float(cost_expr.getValue())
    G_vals = np.array([_sv(G[t]) for t in range(T)])
    slack_vals = np.array([_sv(sigma[v]) for v in range(N)])
    delta_soc_vals = np.array([_sv(delta_soc[v]) for v in range(N)])

    return dict(status='ok', cost=cost_val, P_vals=None, G_vals=G_vals,
                slack_vals=slack_vals, delta_soc_vals=delta_soc_vals,
                u_peak_realized=float(_sv(u_peak_diag)), vd=vd_oos)


def _solve_one_oos_scenario(s, X_star, ev_oos, ce_oos, Pi_oos, g_max_oos,
                             u_peak_star, include_peak_in_cost, feasibility_tol):
    vd_s = _ev_bootstrap_to_vd(ev_oos, s)
    r = solve_oos_recourse(X_star, vd_s, ce_oos[:, s], Pi_oos[:, s], g_max_oos,
                            time_limit=8, gap=0.02)
    if r['status'] != 'ok':
        return s, None
    peak_add = cfg.GAMMA_PEAK_EUR_PER_KW * float(u_peak_star) if include_peak_in_cost else 0.0
    cost = r['cost'] + peak_add
    feasible = bool(np.all(r['slack_vals'] < feasibility_tol) and
                     np.all(r['delta_soc_vals'] < feasibility_tol))
    return s, dict(cost=cost, feasible=feasible, peak=r['u_peak_realized'])


def evaluate_oos_recourse(X_star, u_peak_star, Pi_oos, ce_oos, ev_oos,
                           g_max_oos, S=None, include_peak_in_cost=True,
                           J_IS_reference=None, feasibility_tol=1e-6,
                           verbose=False, n_workers=6):
    S = S if S is not None else Pi_oos.shape[1]
    costs = np.full(S, np.nan)
    feasible_ok = np.zeros(S, dtype=bool)
    peak_realized = np.zeros(S)
    res_used = np.zeros(S)

    worker = partial(_solve_one_oos_scenario, X_star=X_star, ev_oos=ev_oos,
                      ce_oos=ce_oos, Pi_oos=Pi_oos, g_max_oos=g_max_oos,
                      u_peak_star=u_peak_star, include_peak_in_cost=include_peak_in_cost,
                      feasibility_tol=feasibility_tol)

    t0 = time.perf_counter()
    with Pool(n_workers) as pool:
        for i, (s, res) in enumerate(pool.imap_unordered(worker, range(S), chunksize=10)):
            if res is not None:
                costs[s] = res['cost']
                feasible_ok[s] = res['feasible']
                peak_realized[s] = res['peak']
            if (i + 1) % 50 == 0 or (i + 1) == S:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (S - i - 1)
                print(f"      OOS {i+1}/{S}  ({elapsed:.0f}s ecoulees, ETA {eta:.0f}s)", flush=True)

    n_infeasible = int(np.sum(np.isnan(costs)))
    valid = ~np.isnan(costs)

    # Import local pour éviter import circulaire
    from src.evaluation.kpis import compute_e_oos, compute_cvar90, compute_reliability

    E_OOS = compute_e_oos(costs)
    CVaR90 = compute_cvar90(costs)
    reliability = compute_reliability(costs, J_IS_reference) if J_IS_reference is not None else np.nan
    feasibility_rate = float(np.mean(feasible_ok[valid])) if valid.any() else np.nan

    return dict(
        costs=costs, E_OOS=E_OOS, CVaR90=CVaR90,
        reliability=reliability, feasibility_rate=feasibility_rate,
        peak_realized_mean=float(np.mean(peak_realized[valid])) if valid.any() else np.nan,
        res_used_mean=float(np.mean(res_used[valid])) if valid.any() else np.nan,
        n_infeasible=n_infeasible, n_scenarios=S,
    )