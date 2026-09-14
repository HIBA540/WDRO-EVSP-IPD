"""
Configuration globale du projet WDRO-EVSP.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    T: int = 24
    S_IS: int = 365
    S_OOS: int = 150

    ETA: float = 0.95

    GAMMA_PEAK_EUR_PER_KW: float = 1.0
    KAPPA_SLACK: float = 50.0
    LAM_SOC: float = 2.0
    DELTA_RAMP: float = 0.20
    SOC_TOL_MIN: float = 0.02
    SOC_TOL_MAX: float = 0.10

    N_PRICE_FEATS: int = 5
    N_RES_FEATS: int = 4
    N_EV_FEATS: int = 4

    IDX_PRICE: tuple = (0, 5)
    IDX_RES: tuple = (5, 9)
    IDX_EV: tuple = (9, 13)

    WDRO_K_NEIGHBORS: int = 30
    WDRO_MAX_RECOURSE_SCENARIOS: int = 20

    RHO_EK_BETA: float = 0.05
    RHO_BAI_ALPHA: float = 0.05
    BAI_M_SUMMARIES: int = 1

    RHO_GOTOH_GRID: tuple = (0.1, 0.35, 0.7, 1.5)
    GOTOH_K_BOOTSTRAP: int = 50
    GOTOH_LAMBDA_MV: float = 0.5

    N_BOOTSTRAP: int = 2000
    BOOTSTRAP_CI_ALPHA: float = 0.05
    REL_ALPHA: float = 0.90

    RHO_K_FORECAST: int = 100
    IPD_BETA: float = 0.05
    IPD_MIN_TRAIN_WINDOW: int = 90

    PLAN_N_EV: int = 150
    PLAN_G_MAX: float = 300.
    PLAN_DEMAND_SCALE: float = 1.0

    OOS_VOL: float = 1.0
    OOS_TAIL_NU: float = 5.0
    OOS_HEAVY_FRAC: float = 0.15

    SOLVER_TIME: int = 300
    SOLVER_GAP_SP: float = 0.005
    SOLVER_GAP_WDRO: float = 0.003

    LBBD_MAX_ITERS: int = 50
    LBBD_GAP_TOL: float = 5e-3
    LBBD_SUB_TIME_LIMIT: float = 30.0
    LBBD_SUB_GAP: float = 0.01

    EV_DATA_PATH: str = "data/ACN-DATA.xlsx"
    PRIX_DATA_PATH: str = "data/PRIIX.xlsx"
    RES_DATA_PATH: str = "data/RES.xlsx"
    PRIX_ZONE: str = 'SCE'
    SEED: int = 42

    SCALE_N_EV_LIST: tuple = (20, 50, 100, 200, 300)
    SCALE_S_IS_LIST: tuple = (50, 100, 200, 365)
    SCALE_SEEDS: tuple = (42, 123, 777)


cfg = Config()
D_EFF = cfg.N_PRICE_FEATS + cfg.N_RES_FEATS + cfg.N_EV_FEATS
VERSION = "EVSP_WDRO_v10_FULL_CORRECTED 50EVS"

MASTER_SEED = 42

XI_FEATURE_NAMES = [
    'prix_mean', 'prix_std', 'prix_max', 'prix_skew', 'prix_opk_ratio',
    'res_mean', 'res_std', 'res_min_ratio', 'res_ramp_max',
    'arr_std_h', 'frac_fast', 'e_need_cv', 'soc_init_mean',
]

CALIBRATION_METHODS = {
    'rho_ek': {
        'label': 'EK (2018)', 'family': 'Theoretical — A Priori',
        'color': '#3498DB', 'ls': '-',
        'paper': 'Mohajerin Esfahani & Kuhn (2018), Theorem 3.4',
    },
    'rho_gotoh': {
        'label': 'Gotoh (2021)', 'family': 'MV-Bootstrap Frontier',
        'color': '#F39C12', 'ls': '--',
        'paper': 'Gotoh, Kim & Lim (2021), Algorithm 1',
    },
    'rho_bai': {
        'label': 'Bai (2022)', 'family': 'KS-Statistical Confidence',
        'color': '#95A5A6', 'ls': '-.',
        'paper': 'Bai, Huang & Lam (2022)',
    },
    'rho_fics': {
        'label': 'IPD ★ (Ours)', 'family': 'IPD-Principled',
        'color': '#C0392B', 'ls': '-',
        'paper': 'This work',
        'formula': 'rho_b = R^retro_b + R^prosp_b(beta); rho_IPD = (sum w_b^2 rho_b^2)^(1/2)',
    },
}
ALL_METHODS_ORDER = ['rho_ek', 'rho_gotoh', 'rho_bai', 'rho_fics']

N_CHARGERS = 30
N_L1, N_L2, N_DC = 8, 14, 8
CHARGER_POWER = np.array([3.7] * N_L1 + [11.0] * N_L2 + [50.0] * N_DC)
BAT_THRESH_DC = 30.0
BAT_THRESH_L2 = 15.0

CASES = {
    'C1-Baseline':   dict(prix_factor=0.9, res_factor=0.50, demand_scale=1.0, g_max_oos=300., regime='normal',   desc='Ordinary future'),
    'C2-HighPrice':  dict(prix_factor=3.5, res_factor=0.50, demand_scale=1.0, g_max_oos=300., regime='moderate', desc='Price shock x3.5'),
    'C3-REScrash':   dict(prix_factor=1.0, res_factor=0.05, demand_scale=1.0, g_max_oos=300., regime='moderate', desc='RES crash near-zero'),
    'C4-Congestion': dict(prix_factor=1.2, res_factor=0.35, demand_scale=1.0, g_max_oos=150., regime='moderate', desc='Network congestion'),
    'C5-HighDemand': dict(prix_factor=1.2, res_factor=0.35, demand_scale=2.0, g_max_oos=300., regime='moderate', desc='High EV demand x2'),
    'C6-PriceRES':   dict(prix_factor=3.0, res_factor=0.10, demand_scale=1.5, g_max_oos=250., regime='extreme',  desc='High price + low RES'),
    'C7-CongDem':    dict(prix_factor=1.5, res_factor=0.30, demand_scale=2.0, g_max_oos=120., regime='extreme',  desc='Congestion + high demand'),
    'C8-Storm':      dict(prix_factor=3.5, res_factor=0.05, demand_scale=2.0, g_max_oos=100., regime='extreme',  desc='Perfect storm'),
}

STRESS_CASES = {
    'S1-NearShift':   dict(prix_factor=1.1, res_factor=0.90, demand_scale=1.0, g_max_oos=300., regime='near',        shift_type='interpolation', desc='Mild shift'),
    'S2-FarShift':    dict(prix_factor=2.5, res_factor=0.30, demand_scale=1.3, g_max_oos=200., regime='far',         shift_type='extrapolation', desc='Moderate shift'),
    'S3-Regime':      dict(prix_factor=2.0, res_factor=0.20, demand_scale=1.5, g_max_oos=200., regime='transition', shift_type='bimodal',        desc='Regime transition'),
    'S4-HeavyTail':   dict(prix_factor=4.0, res_factor=0.10, demand_scale=2.0, g_max_oos=150., regime='extreme',    shift_type='pareto',          desc='Extreme heavy tail'),
    'S5-Adversarial': dict(prix_factor=5.0, res_factor=0.05, demand_scale=2.5, g_max_oos=100., regime='adversarial', shift_type='adversarial',    desc='Adversarial shift'),
}

ALL_CASES = {**CASES, **STRESS_CASES}
REGIME_COLORS = {'normal': '#4ECDC4', 'moderate': '#FFE66D', 'extreme': '#FF6B6B',
                 'near': '#4ECDC4', 'far': '#FFB347', 'transition': '#DDA0DD',
                 'adversarial': '#8B0000'}

CASES_LOCAL = {k: dict(regime=v['regime']) for k, v in CASES.items()}
STRESS_CASES_LOCAL = {k: dict(regime=v['regime']) for k, v in STRESS_CASES.items()}