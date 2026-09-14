"""Calibration EK (2018) — Mohajerin Esfahani & Kuhn."""
import numpy as np

from config.settings import cfg


def calibrate_rho_ek(Xi_hist_norm: np.ndarray, beta=None, d_eff=None,
                      verbose=True, c1=2.0, c2=0.5) -> dict:
    beta = beta or cfg.RHO_EK_BETA
    N, m_nominal = Xi_hist_norm.shape
    m = float(d_eff) if d_eff is not None else float(m_nominal)
    exponent_small = max(m, 2.0)
    a = exponent_small + 1.0
    log_term = np.log(c1 / beta)
    threshold_N = log_term / c2
    if N >= threshold_N:
        rho_ek = (log_term / (c2 * N)) ** (1.0 / exponent_small)
        regime = f"N>=N* -> exponent 1/{exponent_small:.1f}"
    else:
        rho_ek = (log_term / (c2 * N)) ** (1.0 / a)
        regime = f"N<N* -> exponent 1/{a:.1f}"
    if verbose:
        print(f"\n  [EK18] N={N} m={m:.1f} -> rho_EK = {rho_ek:.5f} ({regime})")
    return dict(rho=float(rho_ek), N=N, m=m, regime=regime,
                label='EK (2018)', family='Theoretical — A Priori', calib_time=0.0)