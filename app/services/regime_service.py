

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker

from app.models import RegimeSnapshot
from app.schemas.regime import RegimeSnapshotResponse


logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("REGIME_MODEL_DIR", "/app/data"))
MODEL_PATH = MODEL_DIR / "regime_model.pkl"

FEATURE_NAMES = [
    "log_return",
    "realized_vol_5d",
    "realized_vol_21d",
    "momentum_60d",
    "drawdown_252d",
    "log_vix",
    "vix_change_5d",
]

LABEL_BULL = "bull"
LABEL_BEAR = "bear"
LABEL_SIDEWAYS = "sideways"

STRATEGY_MAP = {
    LABEL_BULL: "max_sharpe",
    LABEL_BEAR: "min_vol",
    LABEL_SIDEWAYS: "risk_parity",
}

REGIME_TICKERS = ["SPY", "^VIX"]

SMOOTH_WINDOW = 5
SMOOTH_CONF_THRESHOLD = 0.85


def _smooth_states(
    states: np.ndarray,
    window: int = SMOOTH_WINDOW,
    *,
    proba: np.ndarray | None = None,
    conf_threshold: float = SMOOTH_CONF_THRESHOLD,
) -> np.ndarray:
    states = np.asarray(states)
    if len(states) <= window:
        return states
    smoothed = states.copy()
    for i in range(window - 1, len(states)):
        if proba is not None and proba[i, states[i]] >= conf_threshold:
            smoothed[i] = int(states[i])
            continue
        win = states[i - window + 1 : i + 1]
        vals, counts = np.unique(win, return_counts=True)
        smoothed[i] = int(vals[np.argmax(counts)])
    return smoothed


class RegimeService:
    def __init__(self, data_service, session_factory: sessionmaker):
        self.data = data_service
        self.session_factory = session_factory

    async def train(self, lookback_days: int = 5040) -> None:
        series = await self._fetch_series(lookback_days)
        if series is None:
            logger.warning("regime.train: insufficient market data")
            return
        await asyncio.to_thread(self._train_sync, *series)

    async def predict_current(self) -> RegimeSnapshotResponse | None:
        series = await self._fetch_series(lookback_days=400)
        if series is None:
            return None

        if not MODEL_PATH.exists():
            await self.train()
            if not MODEL_PATH.exists():
                return None

        return await asyncio.to_thread(self._predict_sync, *series)

    def regime_to_strategy(self, regime: str) -> str:
        return STRATEGY_MAP.get(regime, "max_sharpe")

    async def _fetch_series(self, lookback_days: int):
        df = await self.data.get_prices(REGIME_TICKERS, lookback_days=lookback_days)
        if df.empty or "SPY" not in df.columns or "^VIX" not in df.columns:
            return None
        spy = df["SPY"].astype(float)
        vix = df["^VIX"].astype(float)
        return spy, vix

    def _train_sync(
        self,
        spy: pd.Series,
        vix: pd.Series,
        tnx: pd.Series | None = None,
        irx: pd.Series | None = None,
    ) -> None:
        features = self._build_features(spy, vix)
        if features.empty or len(features) < 320:
            logger.warning("regime._train_sync: not enough observations")
            return

        X_raw = features[FEATURE_NAMES].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        transmat_prior = np.eye(3) * 3.0 + 1.0

        log_return_idx = FEATURE_NAMES.index("log_return")
        vol5_idx = FEATURE_NAMES.index("realized_vol_5d")
        vol21_idx = FEATURE_NAMES.index("realized_vol_21d")
        dd_idx = FEATURE_NAMES.index("drawdown_252d")
        log_vix_idx = FEATURE_NAMES.index("log_vix")

        bear_q = np.quantile(X[:, log_return_idx], 0.10)
        side_q = np.quantile(X[:, log_return_idx], 0.50)
        bull_q = np.quantile(X[:, log_return_idx], 0.90)

        init_means = np.zeros((3, X.shape[1]))
        init_means[0, log_return_idx] = bear_q
        init_means[0, vol5_idx] = np.quantile(X[:, vol5_idx], 0.85)
        init_means[0, vol21_idx] = np.quantile(X[:, vol21_idx], 0.85)
        init_means[0, dd_idx] = np.quantile(X[:, dd_idx], 0.10)
        init_means[0, log_vix_idx] = np.quantile(X[:, log_vix_idx], 0.85)
        init_means[1, log_return_idx] = side_q
        init_means[2, log_return_idx] = bull_q
        init_means[2, vol5_idx] = np.quantile(X[:, vol5_idx], 0.20)
        init_means[2, vol21_idx] = np.quantile(X[:, vol21_idx], 0.20)
        init_means[2, dd_idx] = np.quantile(X[:, dd_idx], 0.85)
        init_means[2, log_vix_idx] = np.quantile(X[:, log_vix_idx], 0.20)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=500,
            tol=1e-4,
            random_state=42,
            transmat_prior=transmat_prior,
            init_params="stc",
            params="stmc",
        )
        model.means_ = init_means
        model.fit(X)

        means = model.means_[:, log_return_idx]
        order = np.argsort(means)
        label_map = {
            int(order[0]): LABEL_BEAR,
            int(order[1]): LABEL_SIDEWAYS,
            int(order[2]): LABEL_BULL,
        }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump((model, scaler, label_map, FEATURE_NAMES), MODEL_PATH)
        logger.info("regime model trained and saved to %s", MODEL_PATH)

    def _predict_sync(
        self,
        spy: pd.Series,
        vix: pd.Series,
        tnx: pd.Series | None = None,
        irx: pd.Series | None = None,
    ) -> RegimeSnapshotResponse | None:
        loaded = joblib.load(MODEL_PATH)
        if len(loaded) == 4:
            model, scaler, label_map, _ = loaded
        else:
            model, label_map, _ = loaded
            scaler = None

        features = self._build_features(spy, vix)
        if features.empty:
            return None

        X_raw = features[FEATURE_NAMES].values
        X = scaler.transform(X_raw) if scaler is not None else X_raw

        states = model.predict(X)
        proba = model.predict_proba(X)
        smoothed = _smooth_states(states, window=SMOOTH_WINDOW, proba=proba)

        last_state = int(smoothed[-1])
        regime = label_map[last_state]

        window = min(SMOOTH_WINDOW, len(proba))
        confidence = float(proba[-window:, last_state].mean())

        last_proba = proba[-1]
        probabilities = {
            label_map[i]: float(last_proba[i]) for i in range(model.n_components)
        }

        last_features = {
            name: float(features.iloc[-1][name]) for name in FEATURE_NAMES
        }
        ts = datetime.now(tz=timezone.utc)

        if self.session_factory is not None:
            with self.session_factory() as db:
                db.add(
                    RegimeSnapshot(
                        ts=ts,
                        regime=regime,
                        confidence=confidence,
                        features={**last_features, "_probabilities": probabilities},
                    )
                )
                db.commit()

        return RegimeSnapshotResponse(
            ts=ts,
            regime=regime,
            confidence=confidence,
            features=last_features,
            probabilities=probabilities,
        )

    @staticmethod
    def _build_features(
        spy: pd.Series,
        vix: pd.Series,
        tnx: pd.Series | None = None,
        irx: pd.Series | None = None,
    ) -> pd.DataFrame:
        spy = spy.astype(float).sort_index()
        vix = vix.astype(float).sort_index()
        df = pd.DataFrame(index=spy.index)

        df["log_return"] = np.log(spy / spy.shift(1))
        df["realized_vol_5d"] = df["log_return"].rolling(5).std()
        df["realized_vol_21d"] = df["log_return"].rolling(21).std()
        df["momentum_60d"] = spy / spy.shift(60) - 1.0
        df["drawdown_252d"] = spy / spy.rolling(252, min_periods=60).max() - 1.0

        log_vix = np.log(vix.reindex(df.index).ffill().clip(lower=1e-3))
        df["log_vix"] = log_vix
        df["vix_change_5d"] = log_vix - log_vix.shift(5)

        return df.dropna()
