"""Moteur de prévision ξ — GBM + Quantile Forest + rolling-origin residuals."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

from config.settings import cfg


class QuantileForest:
    def __init__(self, n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=2)
        self._leaf_ids_train = None
        self._y_train = None
        self._fitted = False

    def fit(self, X, y):
        self.rf.fit(X, y)
        self._leaf_ids_train = self.rf.apply(X)
        self._y_train = y.copy()
        self._fitted = True
        return self

    def predict_quantile_samples(self, x_new, n_samples=50, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        x_new = np.asarray(x_new).reshape(1, -1)
        leaf_new = self.rf.apply(x_new)[0]
        weights = np.zeros(len(self._y_train))
        for tree_idx in range(len(leaf_new)):
            match = (self._leaf_ids_train[:, tree_idx] == leaf_new[tree_idx])
            if match.sum() > 0:
                weights[match] += 1.0 / match.sum()
        weights /= (weights.sum() + 1e-12)
        idx = rng.choice(len(self._y_train), size=n_samples, replace=True, p=weights)
        return self._y_train[idx]


class ForecastEngineV4:
    def __init__(self, Xi_hist_raw, proxy_costs, ev_dates, min_train_window=90):
        self.Xi_hist_raw = np.asarray(Xi_hist_raw, dtype=float)
        self.ev_dates = list(ev_dates)
        self.N, self.d = self.Xi_hist_raw.shape
        self.L = min_train_window
        self.scaler = RobustScaler(quantile_range=(10., 90.))
        self.Xi_scaled = self.scaler.fit_transform(self.Xi_hist_raw)
        self._gbm_models = {}
        self._qrf_models = {}
        self._residuals_operational = {}
        self._residuals_oos = {}
        self._build_features_cache()
        self._fit_gbm_and_qrf_full_sample()
        self._fit_rolling_origin_residuals()
        print(f"  [ForecastV4] modele operationnel full-history + erreurs rolling-origin — d={self.d}  N={self.N}")

    def _build_features(self, Xi_scaled, dates):
        N = len(Xi_scaled)
        feats = np.zeros((N, self.d * 4 + 4))
        for j in range(N):
            lag1 = Xi_scaled[max(0, j - 1)]
            lag7 = Xi_scaled[max(0, j - 7)]
            win = Xi_scaled[max(0, j - 6):j + 1].mean(axis=0)
            date = dates[j]
            doy = date.timetuple().tm_yday / 365.0
            dow = date.weekday() / 6.0
            feats[j, :self.d] = Xi_scaled[j]
            feats[j, self.d:2 * self.d] = lag1
            feats[j, 2 * self.d:3 * self.d] = lag7
            feats[j, 3 * self.d:4 * self.d] = win
            feats[j, 4 * self.d] = doy
            feats[j, 4 * self.d + 1] = dow
            feats[j, 4 * self.d + 2] = np.sin(2 * np.pi * doy)
            feats[j, 4 * self.d + 3] = np.cos(2 * np.pi * doy)
        return feats

    def _build_features_cache(self):
        self._feats = self._build_features(self.Xi_scaled, self.ev_dates)

    def _fit_gbm_and_qrf_full_sample(self):
        X_tr = self._feats[:-1]
        for k in range(self.d):
            y_k = self.Xi_scaled[1:, k]
            gbm = GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=8, random_state=42)
            gbm.fit(X_tr, y_k)
            self._gbm_models[k] = gbm
            y_hat = gbm.predict(X_tr)
            self._residuals_operational[k] = y_k - y_hat
            qrf = QuantileForest(n_estimators=200, max_depth=5,
                                 min_samples_leaf=max(5, self.N // 50), random_state=42)
            qrf.fit(X_tr, self._residuals_operational[k])
            self._qrf_models[k] = qrf

    def _fit_rolling_origin_residuals(self):
        L = max(self.L, 10)
        idx_range = list(range(L, self.N - 1))
        oos = {k: [] for k in range(self.d)}
        for t in idx_range:
            origin_scaler = RobustScaler(quantile_range=(10., 90.))
            Xi_train_scaled = origin_scaler.fit_transform(self.Xi_hist_raw[:t + 1])
            feats_t = self._build_features(Xi_train_scaled, self.ev_dates[:t + 1])
            X_tr_t = feats_t[:t]
            for k in range(self.d):
                y_tr_t = Xi_train_scaled[1:t + 1, k]
                gbm_t = GradientBoostingRegressor(
                    n_estimators=80, max_depth=3, learning_rate=0.08,
                    subsample=0.8, min_samples_leaf=8, random_state=42)
                gbm_t.fit(X_tr_t, y_tr_t)
                x_next = feats_t[t].reshape(1, -1)
                y_hat_next = float(gbm_t.predict(x_next)[0])
                y_true_next = float(origin_scaler.transform(self.Xi_hist_raw[t + 1:t + 2])[0, k])
                oos[k].append(y_true_next - y_hat_next)
        for k in range(self.d):
            self._residuals_oos[k] = np.array(oos[k])
        print(f"    [IPD] Erreurs rolling-origin (scaler + GBM réentraînés): {len(idx_range)} points/dim")

    def predict_point(self, idx_today):
        row = self._feats[idx_today].reshape(1, -1)
        return np.array([self._gbm_models[k].predict(row)[0] for k in range(self.d)])

    def generate_forecast_scenarios(self, idx_today=None, K=100, seed=42):
        rng = np.random.RandomState(seed)
        if idx_today is None:
            idx_today = self.N - 1
        xi_fc_scaled = self.predict_point(idx_today)
        row = self._feats[idx_today].reshape(1, -1)
        K_raw = min(K * 3, 500)
        Xi_perturbed = np.zeros((K_raw, self.d))
        for k in range(self.d):
            residuals_k = self._qrf_models[k].predict_quantile_samples(row, n_samples=K_raw, rng=rng)
            Xi_perturbed[:, k] = xi_fc_scaled[k] + residuals_k
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(self.Xi_scaled)
        dists, idxs = nn.kneighbors(Xi_perturbed)
        dist_to_center = np.linalg.norm(Xi_perturbed - xi_fc_scaled, axis=1)
        order = np.argsort(dist_to_center)
        seen = set()
        selected_analog_idx = []
        for i in order:
            ai = int(idxs[i, 0])
            if ai not in seen:
                seen.add(ai)
                selected_analog_idx.append(ai)
            if len(selected_analog_idx) >= K:
                break
        while len(selected_analog_idx) < K:
            selected_analog_idx.append(rng.choice(self.N))
        analog_idxs = np.array(selected_analog_idx[:K])
        Xi_fc_scaled = np.zeros((K, self.d))
        for i, ai in enumerate(analog_idxs):
            noise = np.array([
                self._qrf_models[k].predict_quantile_samples(row, n_samples=1, rng=rng)[0] * 0.3
                for k in range(self.d)])
            Xi_fc_scaled[i] = self.Xi_scaled[ai] + noise
        Xi_fc_raw = self.scaler.inverse_transform(Xi_fc_scaled)
        return Xi_fc_raw, Xi_fc_scaled

    def evaluate_forecast_quality(self, verbose=True):
        X_tr = self._feats[:-1]
        rmse_per_dim = []
        coverage_90 = []
        for k in range(self.d):
            y_true = self.Xi_scaled[1:, k]
            y_hat = self._gbm_models[k].predict(X_tr)
            rmse_per_dim.append(float(np.sqrt(np.mean((y_true - y_hat) ** 2))))
            res = self._residuals_operational[k]
            p5, p95 = np.percentile(res, 5), np.percentile(res, 95)
            actual_res = y_true - y_hat
            coverage_90.append(float(np.mean((actual_res >= p5) & (actual_res <= p95))))
        mean_rmse = float(np.mean(rmse_per_dim))
        mean_coverage = float(np.mean(coverage_90))
        if verbose:
            print(f"  [ForecastV4] RMSE (scaled) = {mean_rmse:.4f}")
            print(f"  [ForecastV4] Coverage P5-P95 = {mean_coverage:.3f}")
        return dict(rmse=mean_rmse, coverage=mean_coverage,
                    rmse_per_dim=rmse_per_dim, coverage_per_dim=coverage_90)