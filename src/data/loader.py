"""Chargement et préparation des données réelles."""
import numpy as np
import pandas as pd
from scipy.stats import pareto

from config.settings import cfg, D_EFF
from .xi_constructor import XiConstructor, RobustXiNormalizer, _h2s


class RealDataLoader:
    def __init__(self, c=cfg):
        self.cfg = c
        self._load_price()
        self._load_res()
        self._load_ev()
        self.xi_ctor = XiConstructor()
        self._fit_scalers_and_normalizer()

    def _load_price(self):
        df = pd.read_excel(self.cfg.PRIX_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Zone'] == self.cfg.PRIX_ZONE].sort_values('Date').reset_index(drop=True)
        raw = np.clip(df['Price (cents/kWh)'].values / 100., 0.001, None)
        self.price_mat = raw.reshape(365, 24)
        self._price_lm = np.mean(np.log(self.price_mat + 1e-9), axis=0)
        self._price_cov = np.cov(np.log(self.price_mat + 1e-9).T) + 1e-8 * np.eye(24)
        self._price_chol = np.linalg.cholesky(self._price_cov)
        print(f"[Prix] mu={np.mean(self.price_mat)*100:.2f}c  sigma={np.std(self.price_mat)*100:.2f}c")

    def _load_res(self):
        df = pd.read_excel(self.cfg.RES_DATA_PATH)
        self.res_mat = np.maximum(df['P_total'].values, 0.).reshape(365, 24)
        self._res_mean = np.mean(self.res_mat, axis=0)
        self._res_cov = np.cov(self.res_mat.T) + 1e-6 * np.eye(24)
        self._res_chol = np.linalg.cholesky(self._res_cov)
        print(f"[RES] mu={np.mean(self.res_mat):.1f}kW  max={np.max(self.res_mat):.1f}kW")

    def _load_ev(self):
        df = pd.read_excel(self.cfg.EV_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['t_arr_h'] = df['arrival_hour'].apply(_h2s)
        df['t_dep_h'] = df['departure_hour'].apply(_h2s)
        mask = df['t_dep_h'] <= df['t_arr_h']
        df.loc[mask, 't_dep_h'] = 24.0
        df['t_arr_slot'] = np.floor(df['t_arr_h']).astype(int).clip(0, 22)
        df['t_dep_slot'] = np.ceil(df['t_dep_h']).astype(int).clip(1, 24)
        for col, default in [('battery_capacity', 40.), ('SOC_init', 0.3),
                             ('SOC_final', 0.85), ('P_ev_max', 7.2)]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        df['soc0_kwh'] = (df['SOC_init'].clip(0.05, 0.95) * df['battery_capacity']).clip(lower=0.5)
        df['socd_kwh'] = (df['SOC_final'].clip(0.10, 0.99) * df['battery_capacity']).clip(lower=1.)
        df['flex_slots'] = (df['t_dep_slot'] - df['t_arr_slot']).clip(lower=1)
        df['E_max'] = df['P_ev_max'] * cfg.ETA * df['flex_slots']
        exceed = df['socd_kwh'] - df['soc0_kwh'] > df['E_max']
        df.loc[exceed, 'socd_kwh'] = df.loc[exceed, 'soc0_kwh'] + df.loc[exceed, 'E_max'] * 0.90
        self.ev_df = df
        self.ev_dates = sorted(df['date'].unique())
        npd = df.groupby('date').size()
        print(f"[EV] {len(df)} sessions  n_ev/jour: mu={npd.mean():.1f}")

    def _fit_scalers_and_normalizer(self):
        N = 365
        pf_raw = np.zeros((N, cfg.N_PRICE_FEATS))
        rf_raw = np.zeros((N, cfg.N_RES_FEATS))
        ef_raw = np.zeros((N, cfg.N_EV_FEATS))
        proxy = np.zeros(N)
        for d, date in enumerate(self.ev_dates):
            day_df = self.ev_df[self.ev_df['date'] == date]
            pf_raw[d] = self.xi_ctor._pf(self.price_mat[d])
            rf_raw[d] = self.xi_ctor._rf(self.res_mat[d])
            ef_raw[d] = self.xi_ctor._ef(day_df)
            proxy[d] = self.xi_ctor.proxy_cost(self.price_mat[d], self.res_mat[d], day_df)
        self.xi_ctor.fit_scalers(pf_raw, rf_raw, ef_raw)
        Xi_raw = np.zeros((N, D_EFF))
        for d, date in enumerate(self.ev_dates):
            day_df = self.ev_df[self.ev_df['date'] == date]
            Xi_raw[d] = self.xi_ctor.transform(self.price_mat[d], self.res_mat[d], day_df)
        self._normalizer = RobustXiNormalizer()
        self.Xi_hist_norm = self._normalizer.fit_transform(Xi_raw)
        self.Xi_hist_raw = Xi_raw
        self._proxy_hist = proxy
        print(f"[DataLoader] Xi_hist={self.Xi_hist_norm.shape}  D_EFF={D_EFF}")

    def build_xi_scenario(self, price_s, res_s, ev_s_df):
        xi_raw = self.xi_ctor.transform(price_s, res_s, ev_s_df)
        return self._normalizer.transform(xi_raw.reshape(1, -1)).flatten()

    def get_IS_planning(self, S=None, seed=None):
        T = cfg.T
        S = S or cfg.S_IS
        seed = seed or cfg.SEED
        Pi = self.res_mat.T.copy()
        ce = self.price_mat.T.copy()
        rng = np.random.RandomState(seed)
        ev = self._bootstrap_ev(S, cfg.PLAN_N_EV, rng)
        w = np.ones(S) / S
        Xi = np.zeros((S, D_EFF))
        for s in range(S):
            day_df = self.ev_dict_to_df(ev, s)
            Xi[s] = self.build_xi_scenario(ce[:, s], Pi[:, s], day_df)
        return Pi, ce, ev, w, Xi

    def ev_dict_to_df(self, ev, s):
        df_s = pd.DataFrame({
            't_arr_slot': ev['tarr'][:, s],
            't_dep_slot': ev['tdep'][:, s],
            'battery_capacity': ev['bat'][:, s],
            'soc0_kwh': ev['s0'][:, s],
            'socd_kwh': ev['sdep_raw'][:, s],
            'P_ev_max': ev['pmax'][:, s],
            'SOC_init': ev['s0'][:, s] / (ev['bat'][:, s] + 1e-9),
            'SOC_final': ev['sdep_raw'][:, s] / (ev['bat'][:, s] + 1e-9),
            't_arr_h': ev['tarr'][:, s].astype(float),
        })
        df_s['charger_type'] = 'medium'
        df_s.loc[ev['bat'][:, s] >= 30.0, 'charger_type'] = 'fast'
        df_s.loc[ev['bat'][:, s] < 15.0, 'charger_type'] = 'small'
        return df_s

    def get_IS_xi(self):
        return self.Xi_hist_norm

    def generate_OOS_for_case(self, case_cfg, S, seed):
        rng = np.random.RandomState(seed)
        T = cfg.T
        pf = case_cfg['prix_factor']
        rs = case_cfg['res_factor'] * cfg.PLAN_N_EV * 0.5 / (np.mean(self._res_mean) + 1e-9)
        ds = case_cfg.get('demand_scale', 1.)
        nh = int(S * cfg.OOS_HEAVY_FRAC)
        ce = np.zeros((T, S))
        Pi = np.zeros((T, S))
        shift_type = case_cfg.get('shift_type', 'normal')
        for s in range(S):
            z = self._draw_shift(shift_type, T, rng, seed, s)
            if s < nh:
                chi2v = rng.chisquare(cfg.OOS_TAIL_NU)
                z = z / np.sqrt(chi2v / cfg.OOS_TAIL_NU)
            ce[:, s] = np.clip(np.exp(self._price_lm + cfg.OOS_VOL * (self._price_chol @ z)), 0.001, 5.) * pf
        for s in range(S):
            z = self._draw_shift(shift_type, T, rng, seed, s + 500000)
            Pi[:, s] = np.maximum(0., self._res_mean + cfg.OOS_VOL * (self._res_chol @ z)) * rs
        ev = self._bootstrap_ev(S, cfg.PLAN_N_EV, rng, demand_scale=ds)
        return Pi, ce, ev

    def _draw_shift(self, shift_type, T, rng, seed, salt):
        if shift_type == 'pareto':
            u = np.random.RandomState(seed + salt + 10000).uniform(0, 1, T)
            z = pareto.ppf(u, 1.5) - 1.5
            return z / np.std(z)
        if shift_type == 'bimodal':
            z = np.zeros(T)
            for t in range(T):
                z[t] = rng.normal(-1.5, 0.5) if rng.rand() < 0.5 else rng.normal(1.5, 0.5)
            return z
        if shift_type == 'adversarial':
            return -rng.standard_normal(T)
        return rng.standard_normal(T)

    def _bootstrap_ev(self, S, n_ev, rng, demand_scale=1.):
        T = cfg.T
        pool = self.ev_df
        np_ = len(pool)
        aa = pool['t_arr_slot'].values.astype(int)
        ad = pool['t_dep_slot'].values.astype(int)
        ab = pool['battery_capacity'].values.astype(float)
        as0 = pool['soc0_kwh'].values.astype(float)
        asd = pool['socd_kwh'].values.astype(float)
        apm = pool['P_ev_max'].values.clip(0.5, 50.).astype(float)
        ev = dict(tarr=np.zeros((n_ev, S), int), tdep=np.zeros((n_ev, S), int),
                  s0=np.zeros((n_ev, S)), sdep=np.zeros((n_ev, S)),
                  sdep_raw=np.zeros((n_ev, S)), bat=np.zeros((n_ev, S)),
                  pmax=np.zeros((n_ev, S)), y=np.zeros((n_ev, T, S), int))
        for s in range(S):
            idx = rng.choice(np_, n_ev, replace=True)
            ta = np.clip(aa[idx].copy(), 0, T - 2)
            td = np.clip(ad[idx].copy(), ta + 1, T)
            bat = ab[idx].copy()
            s0 = np.clip(as0[idx].copy(), 0.10 * bat, 0.80 * bat)
            pm = apm[idx].copy()
            sdr = np.clip(asd[idx].copy(), s0 + 0.05 * bat, 0.95 * bat)
            sds = np.clip(sdr * demand_scale, s0 + 0.05 * bat, 0.95 * bat)
            for v in range(n_ev):
                fl = max(1, int(td[v]) - int(ta[v]))
                Em = pm[v] * cfg.ETA * fl
                if sds[v] - s0[v] > Em:
                    sds[v] = s0[v] + Em * 0.90
                if sdr[v] - s0[v] > Em:
                    sdr[v] = s0[v] + Em * 0.90
            ev['tarr'][:, s] = ta
            ev['tdep'][:, s] = td
            ev['s0'][:, s] = s0
            ev['sdep'][:, s] = sds
            ev['sdep_raw'][:, s] = sdr
            ev['bat'][:, s] = bat
            ev['pmax'][:, s] = pm
            for v in range(n_ev):
                ev['y'][v, ta[v]:min(td[v], T), s] = 1
        return ev