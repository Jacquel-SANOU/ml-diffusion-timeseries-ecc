"""
Etape 7 : Interpretabilite avec SHAP

Comme vu dans le Cours 1 (Prof. Lamrani Alaoui) :
  - L'interpretabilite est essentielle pour comprendre les decisions du modele
  - SHAP (SHapley Additive exPlanations) decompose la prediction en contributions
    de chaque feature

On applique SHAP sur XGBoost et LightGBM pour identifier quelles series
et quels lags contribuent le plus aux previsions.

Produit :
  - SHAP summary plot (importance globale)
  - SHAP waterfall (explication d'une prediction)
  - Permutation importance pour comparaison
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import lightgbm as lgb

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
})

LOOKBACK = 20
HORIZON = 5

# =====================================================================
# 1. CHARGEMENT ET PREPARATION
# =====================================================================
print("=" * 60)
print("SHAP — Interpretabilite des modeles")
print("=" * 60)

train = pd.read_csv("data/processed/train.csv", index_col="Date", parse_dates=True)
val = pd.read_csv("data/processed/val.csv", index_col="Date", parse_dates=True)
test = pd.read_csv("data/processed/test.csv", index_col="Date", parse_dates=True)

series_names = list(train.columns)
n_features = len(series_names)


def create_windows(data, lookback, horizon):
    X, Y = [], []
    arr = data.values
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i + lookback])
        Y.append(arr[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(Y)


X_train, Y_train = create_windows(train, LOOKBACK, HORIZON)
X_val, Y_val = create_windows(val, LOOKBACK, HORIZON)
X_test, Y_test = create_windows(test, LOOKBACK, HORIZON)

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

feature_names = [f"{s}_t-{LOOKBACK - lag}" for lag in range(LOOKBACK) for s in series_names]

sp500_idx = series_names.index("S&P 500") if "S&P 500" in series_names else 7
target_h = 0

print(f"  Features : {len(feature_names)}")
print(f"  Cible : S&P 500, horizon h={target_h+1}")

# =====================================================================
# 2. ENTRAINER XGBOOST (pour SHAP)
# =====================================================================
print("\n--- XGBoost ---")

model_xgb = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    verbosity=0, random_state=42,
)
model_xgb.fit(
    X_train_flat, Y_train[:, target_h, sp500_idx],
    eval_set=[(X_val_flat, Y_val[:, target_h, sp500_idx])],
    verbose=False,
)
print("  XGBoost entraine")

# =====================================================================
# 3. SHAP SUR XGBOOST
# =====================================================================
print("\n--- Calcul SHAP ---")

explainer = shap.TreeExplainer(model_xgb)
X_explain = X_test_flat[:200]
shap_values = explainer.shap_values(X_explain)

print(f"  SHAP values shape : {shap_values.shape}")

# --- FIGURE 18 : SHAP Summary Plot (beeswarm) ---
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_values, X_explain,
    feature_names=feature_names,
    max_display=20, show=False,
)
plt.title("SHAP — Top 20 features (XGBoost, prevision S&P 500 h+1)", fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/18_shap_summary_xgboost.png", bbox_inches="tight")
plt.close("all")
print(">>> FIGURE 18 sauvegardee : results/figures/18_shap_summary_xgboost.png")

# --- FIGURE 19 : SHAP importance par serie (moyenne) ---
shap_abs = np.abs(shap_values)
importance_by_series = {}
for s_idx, s_name in enumerate(series_names):
    cols = [i * n_features + s_idx for i in range(LOOKBACK)]
    importance_by_series[s_name] = shap_abs[:, cols].mean()

imp_series = pd.Series(importance_by_series).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#3498db"] * len(imp_series)
colors_bar[-1] = "#e74c3c"
ax.barh(imp_series.index, imp_series.values, color=colors_bar)
ax.set_xlabel("Importance SHAP moyenne (|SHAP|)")
ax.set_title("Importance SHAP par serie financiere\n(XGBoost, prevision S&P 500 h+1)",
             fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("results/figures/19_shap_importance_par_serie.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 19 sauvegardee : results/figures/19_shap_importance_par_serie.png")

# --- FIGURE 20 : SHAP importance par lag (decroissance temporelle) ---
importance_by_lag = []
for lag in range(LOOKBACK):
    cols = [lag * n_features + s for s in range(n_features)]
    importance_by_lag.append(shap_abs[:, cols].mean())

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, LOOKBACK + 1), importance_by_lag, color="#3498db", alpha=0.8)
ax.set_xlabel("Lag (t-1 = plus recent)")
ax.set_ylabel("Importance SHAP moyenne")
ax.set_title("Decroissance temporelle de l'importance SHAP\n(XGBoost, prevision S&P 500 h+1)",
             fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/20_shap_importance_par_lag.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 20 sauvegardee : results/figures/20_shap_importance_par_lag.png")

# --- FIGURE 21 : SHAP Waterfall (explication d'une prediction) ---
fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_explain[0],
        feature_names=feature_names,
    ),
    max_display=15,
    show=False,
)
plt.title("SHAP Waterfall — Explication d'une prediction individuelle", fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/21_shap_waterfall.png", bbox_inches="tight")
plt.close("all")
print(">>> FIGURE 21 sauvegardee : results/figures/21_shap_waterfall.png")

# =====================================================================
# 4. LIGHTGBM + SHAP
# =====================================================================
print("\n--- LightGBM + SHAP ---")

model_lgb = lgb.LGBMRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    verbosity=-1, random_state=42,
)
model_lgb.fit(
    X_train_flat, Y_train[:, target_h, sp500_idx],
    eval_set=[(X_val_flat, Y_val[:, target_h, sp500_idx])],
)

explainer_lgb = shap.TreeExplainer(model_lgb)
shap_values_lgb = explainer_lgb.shap_values(X_explain)

# --- FIGURE 22 : Comparaison SHAP XGBoost vs LightGBM ---
imp_xgb = {}
imp_lgb = {}
for s_idx, s_name in enumerate(series_names):
    cols = [i * n_features + s_idx for i in range(LOOKBACK)]
    imp_xgb[s_name] = np.abs(shap_values).mean(axis=0)[cols].sum()
    imp_lgb[s_name] = np.abs(shap_values_lgb).mean(axis=0)[cols].sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (title, imp) in zip(axes, [("XGBoost", imp_xgb), ("LightGBM", imp_lgb)]):
    imp_s = pd.Series(imp).sort_values(ascending=True)
    ax.barh(imp_s.index, imp_s.values, color="#3498db")
    ax.set_xlabel("SHAP total par serie")
    ax.set_title(f"SHAP — {title}", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

plt.suptitle("Comparaison SHAP : XGBoost vs LightGBM", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/22_shap_xgb_vs_lgb.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 22 sauvegardee : results/figures/22_shap_xgb_vs_lgb.png")

print("\n" + "=" * 60)
print("INTERPRETABILITE SHAP TERMINEE — 5 figures (18-22)")
print("=" * 60)
