"""Train and compare Linear Regression, Decision Tree and Random Forest for demand forecasting.
Run:  python train_models.py      (needs demand_data.csv in the same folder)
Creates: models/<name>.pkl, models/metrics.json, models/model_comparison.png
"""
import json, os
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

LAGS = ["sales_last_week", "sales_2_weeks_ago", "sales_3_weeks_ago", "sales_4_weeks_ago"]
FEATURES = ["ratio_1", "ratio_2", "ratio_3", "ratio_4", "trend", "volatility", "log_avg", "unit_price", "is_festival_season"]


def make_features(lags, price, festival):
    """lags: array (n,4) newest first, units/day. Returns (feature DataFrame, avg array). Used by training AND the app."""
    lags = np.asarray(lags, dtype=float)
    avg = lags.mean(axis=1)
    safe = np.where(avg == 0, 1, avg)
    r = lags / safe[:, None]
    X = pd.DataFrame({
        "ratio_1": r[:, 0], "ratio_2": r[:, 1], "ratio_3": r[:, 2], "ratio_4": r[:, 3],
        "trend": r[:, 0] - r[:, 3],          # recent vs oldest week
        "volatility": r.std(axis=1),         # how jumpy sales are
        "log_avg": np.log1p(avg),
        "unit_price": np.asarray(price, dtype=float),
        "is_festival_season": np.asarray(festival, dtype=int),
    })
    return X, avg


def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("demand_data.csv")
    df = df[df[LAGS].mean(axis=1) > 0].reset_index(drop=True)
    X, avg = make_features(df[LAGS].values, df["unit_price"], df["is_festival_season"])
    y_units = df["next_week_sales"].values
    y_ratio = np.clip(y_units / avg, 0, 5)          # predict ratio to average, then scale back

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    Xtr, Xte = X.iloc[tr], X.iloc[te]

    candidates = {
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), None),
        "Decision Tree": (DecisionTreeRegressor(random_state=42),
                          {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [5, 20, 50]}),
        "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                          {"max_depth": [6, 8, 12], "min_samples_leaf": [5, 20]}),
    }

    results, preds, fitted = {}, {}, {}
    naive = Xte["ratio_1"].values * avg[te]
    results["Baseline (same as last week)"] = dict(
        MAE=mean_absolute_error(y_units[te], naive),
        RMSE=mean_squared_error(y_units[te], naive) ** 0.5,
        R2=r2_score(y_units[te], naive), CV_R2_ratio=None, best_params=None)
    preds["Baseline (same as last week)"] = naive

    kf = KFold(5, shuffle=True, random_state=42)
    for name, (est, grid) in candidates.items():
        best_params = None
        if grid:
            gs = GridSearchCV(est, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(Xtr, y_ratio[tr]); est, best_params = gs.best_estimator_, gs.best_params_
        else:
            est.fit(Xtr, y_ratio[tr])
        p = np.clip(est.predict(Xte), 0, None) * avg[te]
        cv = cross_val_score(est, Xtr, y_ratio[tr], cv=kf, scoring="r2").mean()
        results[name] = dict(MAE=mean_absolute_error(y_units[te], p),
                             RMSE=mean_squared_error(y_units[te], p) ** 0.5,
                             R2=r2_score(y_units[te], p), CV_R2_ratio=cv, best_params=best_params)
        preds[name], fitted[name] = p, est
        joblib.dump(est, f"models/{name.lower().replace(' ', '_')}.pkl")

    best = min(fitted, key=lambda n: results[n]["MAE"])
    results = {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in results.items()}
    json.dump({"results": results, "best_model": best, "features": FEATURES, "n_test": int(len(te))},
              open("models/metrics.json", "w"), indent=2)
    joblib.dump(fitted[best], "models/best_model.pkl")

    # ---- charts
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, m in zip(ax, ["MAE", "RMSE", "R2"]):
        a.bar(range(len(names)), [results[n][m] for n in names], color=["#999", "#4c78a8", "#f58518", "#54a24b"])
        a.set_xticks(range(len(names))); a.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=20, fontsize=8)
        a.set_title(m + (" (higher is better)" if m == "R2" else " (lower is better)"))
    plt.tight_layout(); plt.savefig("models/model_comparison.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cap = np.percentile(y_units[te], 99)
    for a, n in zip(ax, fitted):
        a.scatter(y_units[te], preds[n], s=6, alpha=.3); a.plot([0, cap], [0, cap], "r--")
        a.set_xlim(0, cap); a.set_ylim(0, cap); a.set_title(n); a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    plt.tight_layout(); plt.savefig("models/actual_vs_predicted.png", dpi=120); plt.close()

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.plot(kind="barh", title="Random Forest feature importance"); plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=120); plt.close()

    print(pd.DataFrame(results).T[["MAE", "RMSE", "R2", "CV_R2_ratio"]]); print("Best model:", best)


if __name__ == "__main__":
    main()
