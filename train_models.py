"""Compare Linear Regression, Decision Tree and Random Forest for next-week demand.
Run once:  python train_models.py     (needs demand_data.csv in the same folder)
Creates:   demand_best_model.pkl (used by the app), model_comparison.csv, model_comparison.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from features import make_features

LAGS = ["sales_last_week", "sales_2_weeks_ago", "sales_3_weeks_ago", "sales_4_weeks_ago"]

df = pd.read_csv("demand_data.csv")
df = df[df[LAGS].mean(axis=1) > 0].reset_index(drop=True)
X, avg = make_features(df[LAGS].values, df["unit_price"], df["is_festival_season"])
y_units = df["next_week_sales"].values
y_ratio = np.clip(y_units / avg, 0, 5)  # model predicts ratio to average sales; app scales it back

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
Xtr, Xte = X.iloc[idx_train], X.iloc[idx_test]
actual = y_units[idx_test]

models = {
    "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), None),
    "Decision Tree": (DecisionTreeRegressor(random_state=42),
                      {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [5, 20, 50]}),
    "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                      {"max_depth": [6, 8, 12], "min_samples_leaf": [5, 20]}),
}

rows, fitted = [], {}
naive = Xte["ratio_1"].values * avg[idx_test]
rows.append(["Simple guess (same as last week)", mean_absolute_error(actual, naive),
             mean_squared_error(actual, naive) ** 0.5, r2_score(actual, naive)])

for name, (model, grid) in models.items():
    if grid:
        search = GridSearchCV(model, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(Xtr, y_ratio[idx_train])
        model = search.best_estimator_
    else:
        model.fit(Xtr, y_ratio[idx_train])
    pred = np.clip(model.predict(Xte), 0, None) * avg[idx_test]
    rows.append([name, mean_absolute_error(actual, pred), mean_squared_error(actual, pred) ** 0.5, r2_score(actual, pred)])
    fitted[name] = model

table = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).round(3)
table.to_csv("model_comparison.csv", index=False)
print(table.to_string(index=False))

best = table[table["Model"].isin(fitted)].sort_values("MAE").iloc[0]["Model"]
joblib.dump(fitted[best], "demand_best_model.pkl")
print("\nBest model:", best, "-> saved as demand_best_model.pkl")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(table["Model"], table["MAE"], color="#4c78a8"); ax[0].set_title("MAE (lower is better)")
ax[1].bar(table["Model"], table["R2"], color="#54a24b"); ax[1].set_title("R2 (higher is better)")
for a in ax:
    a.tick_params(axis="x", rotation=25, labelsize=8)
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=120)
