"""Cache des flottes EV par jour historique (V_j^K)."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config.settings import (
    cfg, N_CHARGERS, N_L1, N_L2, BAT_THRESH_DC, BAT_THRESH_L2,
)


class ScenarioFleetCache:
    def __init__(self, ev_df: pd.DataFrame, ev_dates: list, seed: int = 42):
        self.ev_df = ev_df
        self.ev_dates = list(ev_dates)
        self.seed = seed
        self._cache: Dict[Tuple[int, int], dict] = {}

    def get(self, day_idx: int, K: int) -> dict:
        key = (day_idx, K)
        if key in self._cache:
            return self._cache[key]
        date = self.ev_dates[day_idx % len(self.ev_dates)]
        day_df = self.ev_df[self.ev_df['date'] == date]
        n_j = len(day_df)
        rng = np.random.RandomState((self.seed + 97 * day_idx + K) % (2**31 - 1))
        if n_j == 0:
            idx = rng.choice(len(self.ev_df), K, replace=True)
            sampled = self.ev_df.iloc[idx]
        elif n_j >= K:
            idx = rng.choice(n_j, K, replace=False)
            sampled = day_df.iloc[idx]
        else:
            idx = rng.choice(n_j, K, replace=True)
            sampled = day_df.iloc[idx]
        vd_j = self._df_to_vd(sampled)
        self._cache[key] = vd_j
        return vd_j

    def _df_to_vd(self, df: pd.DataFrame) -> dict:
        T = cfg.T
        n = len(df)
        ta = np.clip(df['t_arr_slot'].values.astype(int), 0, T - 2)
        td = np.clip(df['t_dep_slot'].values.astype(int), ta + 1, T)
        bat = df['battery_capacity'].values.astype(float)
        s0 = np.clip(df['soc0_kwh'].values.astype(float), 0.10 * bat, 0.80 * bat)
        pm = df['P_ev_max'].values.clip(0.5, 50.).astype(float)
        sd = np.clip(df['socd_kwh'].values.astype(float), s0 + 0.05 * bat, 0.95 * bat)
        for v in range(n):
            Em = pm[v] * cfg.ETA * max(1, int(td[v]) - int(ta[v]))
            if sd[v] - s0[v] > Em:
                sd[v] = s0[v] + Em * 0.90
        chi = np.zeros((n, N_CHARGERS), dtype=int)
        for v in range(n):
            b = float(bat[v])
            chi[v, :N_L1] = 1
            if b >= BAT_THRESH_L2:
                chi[v, N_L1:N_L1 + N_L2] = 1
            if b >= BAT_THRESH_DC:
                chi[v, N_L1 + N_L2:] = 1
        y = np.zeros((n, T), int)
        for v in range(n):
            y[v, ta[v]:min(td[v], T)] = 1
        delta_star = float(np.clip(
            np.mean([max(0., 1. - (pm[v] * cfg.ETA * max(1, int(td[v]) - int(ta[v]))) /
                         max(sd[v] - s0[v], 0.1)) for v in range(n)]),
            cfg.SOC_TOL_MIN, cfg.SOC_TOL_MAX)) if n > 0 else cfg.SOC_TOL_MIN
        return dict(N=n, beta=bat, s0=s0, smin=0.10 * bat, smax=0.95 * bat,
                    sdep=sd, tarr=ta, tdep=td, chi=chi, pmax_v=pm, y=y,
                    delta_star=delta_star)

    def as_dataframe(self, day_idx: int, K: int) -> pd.DataFrame:
        vd = self.get(day_idx, K)
        return pd.DataFrame({
            't_arr_slot': vd['tarr'], 't_dep_slot': vd['tdep'],
            'battery_capacity': vd['beta'], 'soc0_kwh': vd['s0'],
            'socd_kwh': vd['sdep'], 'P_ev_max': vd['pmax_v'],
            'SOC_init': vd['s0'] / (vd['beta'] + 1e-9),
            'SOC_final': vd['sdep'] / (vd['beta'] + 1e-9),
            't_arr_h': vd['tarr'].astype(float),
        })