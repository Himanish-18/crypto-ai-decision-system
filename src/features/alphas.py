import numpy as np
import pandas as pd


def rank(df):
    """Rank logic: rolling rank over window? Or cross-sectional?
    For single asset time series, 'rank' often implies normalizing over a window or just raw value if formula expects cross-sectional.
    Kakushadze alphas are cross-sectional. For single pair, we can approximate 'rank' as 'z-score' or 'percentile' over a lookback window.
    """
    return df.rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def ts_delay(df, d):
    return df.shift(d)


def ts_corr(x, y, d):
    return x.rolling(window=d).corr(y)


def ts_min(df, d):
    return df.rolling(window=d).min()


def ts_max(df, d):
    return df.rolling(window=d).max()


def ts_argmax(df, d):
    return df.rolling(window=d).apply(np.argmax) + 1


def ts_argmin(df, d):
    return df.rolling(window=d).apply(np.argmin) + 1


def ts_rank(df, d):
    return df.rolling(window=d).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])


def delta(df, d):
    return df.diff(d)


class FormulaicAlphas:
    """
    Implementation of 10-20 Kakushadze 101 Alphas suitable for crypto (Price/Volume only).
    Adapted for single-asset time-series application (using rolling windows instead of cross-section where needed).
    """

    def compute_all(self, df: pd.DataFrame, symbol: str = "btc") -> pd.DataFrame:
        open_ = df[f"{symbol}_open"]
        high = df[f"{symbol}_high"]
        low = df[f"{symbol}_low"]
        close = df[f"{symbol}_close"]
        volume = df[f"{symbol}_volume"]
        # VWAP approximation if not present
        vwap = (high + low + close) / 3

        # Alpha#6: (-1 * corr(open, volume, 10))
        df[f"{symbol}_alpha_006"] = -1 * ts_corr(open_, volume, 10)

        # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        delta_close = delta(close, 1)
        cond1 = ts_min(delta_close, 5) > 0
        cond2 = ts_max(delta_close, 5) < 0
        df[f"{symbol}_alpha_009"] = np.where(
            cond1, delta_close, np.where(cond2, delta_close, -delta_close)
        )

        # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        df[f"{symbol}_alpha_012"] = np.sign(delta(volume, 1)) * (-1 * delta(close, 1))

        # Alpha#18: (-1 * rank(((std(abs((close - open)), 5) + (close - open)) + corr(close, open, 10))))
        # Simplified for time-series: -1 * zscore(...)
        term = close.rolling(5).std() + (close - open_) + ts_corr(close, open_, 10)
        df[f"{symbol}_alpha_018"] = (
            -1 * (term - term.rolling(20).mean()) / (term.rolling(20).std() + 1e-9)
        )

        # Alpha#21: ((((sum(close, 7) / 7) - close) + ((corr(vwap, delay(close, 5), 230) < 0) ? close : 0))
        # Part 1: Mean(7) - Close -> Mean Reversion
        # Part 2: If Corr(VWAP, lag(Close,5)) < 0, add Close.
        cond = (
            ts_corr(vwap, ts_delay(close, 5), 20) < 0
        )  # Reduced window 230->20 for 1H crypto
        df[f"{symbol}_alpha_021"] = (close.rolling(7).mean() - close) + np.where(
            cond, close, 0
        )

        # Alpha#26: (-1 * ts_max(corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        df[f"{symbol}_alpha_026"] = -1 * ts_max(
            ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3
        )

        # Alpha#28: scale(((corr(adv20, low, 5) + ((high + low) / 2)) - close))
        # adv20 = Average Daily Volume 20
        adv20 = volume.rolling(20).mean()
        term = ts_corr(adv20, low, 5) + ((high + low) / 2) - close
        df[f"{symbol}_alpha_028"] = term  # Scale is just normalization

        # Alpha#34: (1 - rank((std(returns, 2) / std(returns, 5))))
        ret = close.pct_change()
        df[f"{symbol}_alpha_034"] = 1 - (
            ret.rolling(2).std() / (ret.rolling(5).std() + 1e-9)
        ).rolling(10).rank(pct=True)

        # Alpha#41: (((high * low)^0.5) - vwap)
        df[f"{symbol}_alpha_041"] = np.sqrt(high * low) - vwap

        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low + high - close)), 9))
        inner = ((close - low) - (high - close)) / (high - low + 1e-9)
        df[f"{symbol}_alpha_053"] = -1 * delta(inner, 9)

        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        # Simplified: (Low - Close) / (Low - High) -> Stoch-like
        # Ignoring Open^5/Close^5 scaling factor which explodes
        df[f"{symbol}_alpha_054"] = -1 * (low - close) / (low - high + 1e-9)

        # Alpha#101: ((close - open) / ((high - low) + .001))
        df[f"{symbol}_alpha_101"] = (close - open_) / ((high - low) + 1e-9)

        return df
