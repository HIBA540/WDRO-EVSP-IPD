"""Calibration Bai (2022) — KS-Statistical Confidence."""
import numpy as np
from scipy.stats import ks_2samp

from config.settings import cfg


def calibrate_rho_bai(Xi_hist_norm: np.ndarray, alpha=None, m_summaries=None,
                       verbose=True) -> dict:
    if alpha is None:
        alpha = cfg.RHO_BAI_ALPHA
    if m_summaries is None:
        m_summaries = cfg.BAI_M_SUMMARIES
    N, d = Xi_hist_norm.shape
    alpha_corrected = alpha / max(m_summaries, 1)
    level_corrected = 1.0 - alpha_corrected
    try:
        from scipy.stats import kstwobign
        q_ks = float(kstwobign.ppf(level_corrected))
    except Exception:
        ks_table = {0.80: 1.073, 0.85: 1.138, 0.90: 1.224,
                    0.925: 1.281, 0.95: 1.358, 0.975: 1.480, 0.99: 1.628}
        levels = sorted(ks_table.keys())
        q_ks = 1.358
        for i in range(len(levels) - 1):
            if levels[i] <= level_corrected <= levels[i + 1]:
                t = (level_corrected - levels[i]) / (levels[i + 1] - levels[i])
                q_ks = ks_table[levels[i]] * (1 - t) + ks_table[levels[i + 1]] * t
                break
    eta = q_ks / np.sqrt(N)

    rng = np.random.RandomState(cfg.SEED)
    n_half = N // 2
    ks_empirical = []
    for _ in range(50):
        idx_a = rng.choice(N, n_half, replace=False)
        idx_b = np.setdiff1d(np.arange(N), idx_a)[:n_half]
        ks_per_dim = []
        for k in range(min(d, 5)):
            stat_k, _ = ks_2samp(Xi_hist_norm[idx_a, k], Xi_hist_norm[idx_b, k])
            ks_per_dim.append(stat_k)
        ks_empirical.append(float(np.mean(ks_per_dim)))
    ks_emp_mean = float(np.mean(ks_empirical))
    ks_emp_p95 = float(np.percentile(ks_empirical, 95))
    rho_bai = float(eta)
    coverage_ok = (eta >= ks_emp_p95)

    if verbose:
        print(f"\n  [BAI22] N={N} d={d} alpha={alpha} m_summaries={m_summaries}")
        print(f"    alpha_corrige={alpha_corrected:.5f}  niveau={level_corrected:.5f}")
        print(f"    q_ks={q_ks:.4f}  eta=rho_bai={rho_bai:.5f}")
        print(f"    Validation empirique: KS_moyen={ks_emp_mean:.5f}  "
              f"KS_p95={ks_emp_p95:.5f}  couverture={'OK' if coverage_ok else 'INSUFFISANTE'}")

    return dict(rho=rho_bai, eta=rho_bai, q_ks=q_ks, alpha=alpha,
                alpha_corrected=alpha_corrected, m_summaries=m_summaries,
                ks_emp_mean=ks_emp_mean, ks_emp_p95=ks_emp_p95, coverage_ok=coverage_ok,
                label='Bai (2022)', family='KS-Statistical Confidence', calib_time=0.0)