"""Construction des features ξ et normalisation robuste."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import skew as scipy_skew
from sklearn.preprocessing import RobustScaler, StandardScaler

from config.settings import cfg, D_EFF


class RobustXiNormalizer:
    def __init__(self):
        self._robust = None
        self._scale = 1.0
        self._fitted = False

    def fit(self, Xi):
        Xi = np.nan_to_num(Xi, nan=0., posinf=5., neginf=-5.)
        self._robust = RobustScaler(quantile_range=(10., 90.))
        Xr = np.clip(self._robust.fit_transform(Xi), -5., 5.)
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.choice(len(Xr), min(200, len(Xr)), replace=False)
        D = cdist(Xr[idx], Xr[idx], 'euclidean')
        d_nz = D[D > 1e-12]
        self._scale = float(np.median(d_nz)) if len(d_nz) else 1.0
        self._fitted = True
        return self

    def transform(self, Xi):
        Xi = np.nan_to_num(Xi, nan=0., posinf=5., neginf=-5.)
        Xr = np.clip(self._robust.transform(Xi), -5., 5.)
        return Xr / (self._scale + 1e-12)

    def fit_transform(self, Xi):
        self.fit(Xi)
        return self.transform(Xi)


class XiConstructor:
    def __init__(self):
        self._price_sc = StandardScaler()
        self._res_sc = StandardScaler()
        self._ev_sc = StandardScaler()
        self._fitted = False

    def _pf(self, price_d):
        p = np.maximum(price_d, 1e-6)
        pm = float(np.mean(p))
        fst = float(np.std(p))
        fmx = float(np.max(p))
        try:
            fsk = float(np.clip(scipy_skew(p), -5., 5.))
        except Exception:
            fsk = 0.
        T = len(p)
        off_idx = list(range(0, 6)) + list(range(22, T))
        peak_idx = list(range(9, 21))
        off_m = float(np.mean(p[off_idx]))
        peak_m = float(np.mean(p[peak_idx]))
        fratio = np.clip(off_m / (peak_m + 1e-9), 0., 3.)
        return np.array([pm, fst, fmx, fsk, fratio])

    def _rf(self, res_d):
        r = np.maximum(res_d, 0.)
        rm = float(np.mean(r)) + 1e-9
        return np.array([rm, float(np.std(r)),
                         float(np.min(r)) / rm,
                         float(np.max(np.abs(np.diff(r)))) if len(r) > 1 else 0.])

    def _ef(self, day_df):
        if len(day_df) == 0:
            return np.zeros(4)
        if 't_arr_h' in day_df.columns:
            arr_std = float(day_df['t_arr_h'].std()) if len(day_df) > 1 else 0.
        else:
            arr_std = float(day_df['t_arr_slot'].std()) if len(day_df) > 1 else 0.
        arr_std_norm = np.clip(arr_std / 12., 0., 1.)
        if 'charger_type' in day_df.columns:
            frac_fast = float((day_df['charger_type'] == 'fast').mean())
        else:
            frac_fast = float((day_df.get('P_ev_max', pd.Series([7.2] * len(day_df))) > 15.).mean())
        if 'SOC_final' in day_df.columns and 'SOC_init' in day_df.columns:
            E_need = (day_df['SOC_final'] - day_df['SOC_init']) * day_df['battery_capacity']
        else:
            E_need = (day_df.get('socd_kwh', pd.Series([10.] * len(day_df))) -
                      day_df.get('soc0_kwh', pd.Series([5.] * len(day_df))))
        E_need = E_need.clip(lower=0.)
        e_mu = float(E_need.mean()) + 1e-9
        e_cv = np.clip(float(E_need.std()) / e_mu, 0., 2.)
        if 'SOC_init' in day_df.columns:
            soc_init = float(day_df['SOC_init'].mean())
        else:
            soc_init = float((day_df.get('soc0_kwh', pd.Series([12.] * len(day_df))) /
                              day_df.get('battery_capacity', pd.Series([40.] * len(day_df)))).mean())
        soc_init = np.clip(soc_init, 0., 1.)
        return np.array([arr_std_norm, frac_fast, e_cv, soc_init])

    def fit_scalers(self, pf, rf, ef):
        self._price_sc.fit(pf)
        self._res_sc.fit(rf)
        self._ev_sc.fit(ef)
        self._fitted = True

    def transform(self, price_d, res_d, day_df):
        pf = self._pf(price_d)
        rf = self._rf(res_d)
        ef = self._ef(day_df)
        if self._fitted:
            pf = self._price_sc.transform(pf.reshape(1, -1)).flatten()
            rf = self._res_sc.transform(rf.reshape(1, -1)).flatten()
            ef = self._ev_sc.transform(ef.reshape(1, -1)).flatten()
        xi = np.concatenate([pf, rf, ef])
        return np.nan_to_num(xi, nan=0., posinf=0., neginf=0.)

    def proxy_cost(self, price_d, res_d, day_df):
        T = cfg.T
        if len(day_df) == 0:
            return 0.
        if 'SOC_final' in day_df.columns:
            E_need = ((day_df['SOC_final'] - day_df['SOC_init']) * day_df['battery_capacity']).clip(lower=0.)
        else:
            E_need = (day_df.get('socd_kwh', pd.Series([10.] * len(day_df))) -
                      day_df.get('soc0_kwh', pd.Series([5.] * len(day_df)))).clip(lower=0.)
        E_total = float(E_need.sum())
        E_per_slot = E_total / T
        cost = 0.
        for t in range(T):
            deficit = min(max(0., E_per_slot - float(res_d[t])), cfg.PLAN_G_MAX)
            cost += float(price_d[t]) * deficit
        return cost


def _h2s(h):
    if pd.isna(h):
        return 12.0
    try:
        parts = str(h).split(':')
        v = int(parts[0]) % 24
        if len(parts) > 1:
            v += int(parts[1]) / 60.
        return v
    except Exception:
        return 12.0