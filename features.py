import numpy as np
import pandas as pd


def make_features(lags, price, festival):
    """lags: rows of [last week, 2 weeks ago, 3 weeks ago, 4 weeks ago] in units/day.
    Returns (features, average). Used by BOTH train_models.py and the app, so they always match."""
    lags = np.asarray(lags, dtype=float)
    avg = lags.mean(axis=1)
    r = lags / np.where(avg == 0, 1, avg)[:, None]
    X = pd.DataFrame({
        "ratio_1": r[:, 0], "ratio_2": r[:, 1], "ratio_3": r[:, 2], "ratio_4": r[:, 3],
        "trend": r[:, 0] - r[:, 3],
        "volatility": r.std(axis=1),
        "log_avg": np.log1p(avg),
        "unit_price": np.asarray(price, dtype=float),
        "is_festival_season": np.asarray(festival, dtype=int),
    })
    return X, avg
