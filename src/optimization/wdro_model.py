"""MILP direct pour WDRO (Kantorovich dual)."""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config.settings import cfg, CHARGER_POWER, N_CHARGERS
from .distance import build_neighbor_map, _select_recourse_scenarios


_FLEET_CACHE = None


def set_fleet_cache(fc):
    global _FLEET_CACHE
    _FLEET_CACHE = fc


def get_fleet_cache():
    return _FLEET_CACHE


def _build_scenario_block_direct(m, tag, vd_j, price_j, res_j, g_max, X):
    T = cfg.T
    N = vd_j['N']
    ds = vd_j.get('delta_star', cfg.SOC_TOL_MIN)

    z = {(v, b): m.addVar(lb=0., ub=1., vtype=GRB.BINARY, name=f"z_{tag}_{v}_{b}")
         for v in range(N) for b in range(N_CHARGERS) if vd_j['chi'][v, b] == 1}
    Pn = {(v, t): m.addVar(lb=0., name=f"P_{tag}_{v}_{t}") for v in range(N) for t in range(T)}
    Paux = {(v, b, t): m.addVar(lb=0., name=f"Pa_{tag}_{v}_{b}_{t}")
            for (v, b) in z for t in range(T)}
    sn = {(v, t): m.addVar(lb=0., name=f"soc_{tag}_{v}_{t}") for v in range(N) for t in range(T)}
    sd_v = {v: m.addVar(lb=0., name=f"sd_{tag}_{v}") for v in range(N)}
    sigma = {v: m.addVar(lb=0., name=f"sigma_{tag}_{v}") for v in range(N)}
    delta_soc = {v: m.addVar(lb=0., name=f"dsoc_{tag}_{v}") for v in range(N)}
    G = {t: m.addVar(lb=0., ub=g_max, name=f"G_{tag}_{t}") for t in range(T)}
    m.update()

    for v in range(N):
        m.addConstr(gp.quicksum(z[v, b] for b in range(N_CHARGERS) if (v, b) in z) <= 1)
    for b in range(N_CHARGERS):
        for t in range(T):
            m.addConstr(gp.quicksum(z[v, b] * int(vd_j['y'][v, t])
                                     for v in range(N) if (v, b) in z) <= 1)
    for v in range(N):
        m.addConstr(sn[v, 0] == vd_j['s0'][v] + cfg.ETA * Pn[v, 0])
        for t in range(1, T):
            m.addConstr(sn[v, t] == sn[v, t - 1] + cfg.ETA * Pn[v, t])
        for t in range(T):
            m.addConstr(sn[v, t] >= vd_j['smin'][v])
            m.addConstr(sn[v, t] <= vd_j['smax'][v])
            if vd_j['y'][v, t] == 0:
                m.addConstr(Pn[v, t] == 0)
            else:
                m.addConstr(Pn[v, t] == gp.quicksum(Paux[v, b, t] for b in range(N_CHARGERS) if (v, b) in z))
                m.addConstr(Pn[v, t] <= vd_j['pmax_v'][v])
    for (v, b) in z:
        for t in range(T):
            if vd_j['y'][v, t] == 1:
                m.addConstr(Paux[v, b, t] <= CHARGER_POWER[b] * z[v, b])
                m.addConstr(Paux[v, b, t] <= vd_j['pmax_v'][v] * z[v, b])
            else:
                m.addConstr(Paux[v, b, t] == 0)

    for b in range(N_CHARGERS):
        for t in range(T):
            m.addConstr(
                gp.quicksum(Paux[v, b, t] for v in range(N) if (v, b) in z) <= X[b, t],
                name=f"link_{tag}_{b}_{t}")

    for v in range(N):
        active = [t for t in range(T) if vd_j['y'][v, t] == 1]
        if active:
            m.addConstr(sd_v[v] == vd_j['s0'][v] + cfg.ETA * gp.quicksum(Pn[v, t] for t in active))
        else:
            m.addConstr(sd_v[v] == vd_j['s0'][v])
        m.addConstr(sd_v[v] <= vd_j['smax'][v])
        E_need = max(0., float(vd_j['sdep'][v]) - float(vd_j['s0'][v]))
        n_sl = max(1, len(active))
        E_max = float(vd_j['pmax_v'][v]) * cfg.ETA * n_sl
        delta_v = max(max(0., 1. - E_max / max(E_need, 0.1)) if E_need > 0.01 else 0., ds)
        tgt_v = (1. - delta_v) * float(vd_j['sdep'][v])
        m.addConstr(delta_soc[v] >= tgt_v - sd_v[v])
        m.addConstr(sigma[v] >= sd_v[v] - float(vd_j['sdep'][v]))

    for t in range(T):
        lhs = gp.quicksum(Pn[v, t] * int(vd_j['y'][v, t]) for v in range(N))
        m.addConstr(lhs <= float(res_j[t]) + G[t])

    rl = cfg.DELTA_RAMP * g_max
    for t in range(T - 1):
        m.addConstr(G[t + 1] - G[t] <= rl)
        m.addConstr(G[t] - G[t + 1] <= rl)

    cost_expr = (gp.quicksum(float(price_j[t]) * G[t] for t in range(T)) +
                 cfg.LAM_SOC * gp.quicksum(delta_soc[v] for v in range(N)) +
                 cfg.KAPPA_SLACK * gp.quicksum(sigma[v] for v in range(N)))

    return cost_expr, G


def build_wdro_direct_model(Pi_IS, ce_IS, g_max, D_sp, w_IS,
                             day_indices=None, K=None, fleet_cache=None,
                             neighbor_map=None, method_label='WDRO', verbose=True):
    t_build0 = time.perf_counter()
    fleet_cache = fleet_cache or get_fleet_cache()
    K = K or cfg.PLAN_N_EV
    S = Pi_IS.shape[1]
    day_indices = day_indices if day_indices is not None else list(range(S))
    T = cfg.T

    if neighbor_map is None:
        neighbor_map = build_neighbor_map(D_sp, cfg.WDRO_K_NEIGHBORS)

    anchor_indices = list(range(S))
    needed_local, trunc_diag = _select_recourse_scenarios(
        anchor_indices, neighbor_map, cfg.WDRO_MAX_RECOURSE_SCENARIOS,
        seed=cfg.SEED, verbose=verbose)

    m = gp.Model(f"WDRO_direct_{method_label}")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.Threads = 4
    m.Params.Presolve = 2
    m.Params.MIPFocus = 1

    X = {(b, t): m.addVar(lb=0., ub=float(CHARGER_POWER[b]), name=f"X_{b}_{t}")
         for b in range(N_CHARGERS) for t in range(T)}
    u_peak = m.addVar(lb=0., ub=g_max, name="u_peak")
    lam = m.addVar(lb=0., ub=GRB.INFINITY, obj=0.0, name="lam")
    phi = {s: m.addVar(lb=0., obj=float(w_IS[s]), name=f"phi_{s}") for s in anchor_indices}
    u_peak.Obj = cfg.GAMMA_PEAK_EUR_PER_KW
    m.update()
    m.ModelSense = GRB.MINIMIZE

    cost_expr_by_j = {}
    G_by_j = {}
    for j_local in needed_local:
        j_real = day_indices[j_local]
        vd_j = fleet_cache.get(j_real, K)
        price_j = ce_IS[:, j_local]
        res_j = Pi_IS[:, j_local]
        cost_expr, G_j = _build_scenario_block_direct(
            m, f"j{j_local}", vd_j, price_j, res_j, g_max, X)
        cost_expr_by_j[j_local] = cost_expr
        G_by_j[j_local] = G_j

    for j_local, G_j in G_by_j.items():
        for t in range(T):
            m.addConstr(u_peak >= G_j[t], name=f"peak_{j_local}_{t}")

    n_constrs = 0
    for s_anchor in anchor_indices:
        neigh = [j for j in neighbor_map.get(s_anchor, []) if j in cost_expr_by_j]
        if not neigh:
            continue
        for j_local in neigh:
            d_sj = float(D_sp[s_anchor, j_local]) if s_anchor != j_local else 0.0
            m.addConstr(phi[s_anchor] >= cost_expr_by_j[j_local] - lam * d_sj,
                        name=f"dual_{s_anchor}_{j_local}")
            n_constrs += 1

    if verbose:
        print(f"    [Direct-MILP] {len(needed_local)} scenarios, {n_constrs} contraintes duales, "
              f"{m.NumBinVars} binaires (t_build={time.perf_counter()-t_build0:.1f}s)")

    return dict(model=m, X=X, u_peak=u_peak, lam=lam, phi=phi,
                anchor_indices=anchor_indices, needed_scenarios=needed_local,
                truncation_diagnostics=trunc_diag, T=T)


def solve_wdro_for_rho(built, rho, time_limit=None, mip_gap=None, verbose=True):
    m = built['model']; X = built['X']; u_peak = built['u_peak']; lam = built['lam']
    T = built['T']

    time_limit = time_limit or cfg.SOLVER_TIME
    mip_gap = mip_gap or cfg.SOLVER_GAP_WDRO
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 1 if verbose else 0

    lam.Obj = float(rho)
    m.update()

    t0 = time.perf_counter()
    m.optimize()
    exec_time = time.perf_counter() - t0

    if m.SolCount == 0:
        raise RuntimeError(f"[Direct-MILP] Aucune solution pour rho={rho} (statut={m.Status}).")

    X_vals = np.array([[float(X[b, t].X) for t in range(T)] for b in range(N_CHARGERS)])
    u_peak_val = float(u_peak.X)
    lam_val = float(lam.X)
    J_total = float(m.ObjVal)
    J_energy = float(J_total - cfg.GAMMA_PEAK_EUR_PER_KW * u_peak_val)

    return dict(
        X_vals=X_vals, u_peak=u_peak_val, J_energy_cost=J_energy, J_total=J_total,
        method='WDRO_direct', exec_time=exec_time, lam_star=lam_val, rho_used=rho,
        mip_gap_reached=float(m.MIPGap) if m.SolCount > 0 else np.nan,
        solver_status=m.Status, needed_scenarios=built['needed_scenarios'],
        truncation_diagnostics=built['truncation_diagnostics'],
        P_vals=None, vd_ref=None, G_vals=None, slack_vals=None, delta_soc_vals=None,
        energy_aggregate_kwh=0.0, soc_sat={'pct_satisfied': np.nan, 'soc_final_mean': np.nan},
        is_reference_note="MILP direct (Kantorovich dual, eq. 34-58, approx. voisinage §3.5).",
    )