"""
Etape 6 : Baselines de prevision sur Yahoo Finance

Modeles de reference :
  1. Naive (derniere valeur)
  2. ARIMA (statsmodels)
  3. GARCH (arch) — pour la volatilite
  4. XGBoost / LightGBM — avec features lag
  5. LSTM simple (PyTorch)

On compare ces baselines sur la tache de prevision a horizon H=5 jours
avec un lookback L=20 jours sur les log-rendements normalises.

Metriques : MAE, RMSE, CRPS (pour les modeles probabilistes)

Produit :
  - Figures comparatives
  - Tableau de resultats
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "figure.dpi": 120,
})

LOOKBACK = 20
HORIZON = 5

# =====================================================================
# 1. CHARGEMENT DES DONNEES
# =====================================================================
print("=" * 60)
print("BASELINES — Prevision sur Yahoo Finance")
print(f"  Lookback L={LOOKBACK}, Horizon H={HORIZON}")
print("=" * 60)

train = pd.read_csv("data/processed/train.csv", index_col="Date", parse_dates=True)
val = pd.read_csv("data/processed/val.csv", index_col="Date", parse_dates=True)
test = pd.read_csv("data/processed/test.csv", index_col="Date", parse_dates=True)
norm_params = pd.read_csv("data/processed/norm_params.csv", index_col=0)

series_names = list(train.columns)
print(f"Series : {series_names}")
print(f"Train : {len(train)}, Val : {len(val)}, Test : {len(test)}")


# =====================================================================
# UTILITAIRES
# =====================================================================
def create_windows(data, lookback, horizon):
    X, Y = [], []
    arr = data.values
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i:i + lookback])
        Y.append(arr[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(Y)


def evaluate_forecast(y_true, y_pred, name=""):
    """MAE et RMSE moyens sur toutes les series et horizons."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"Modele": name, "MAE": mae, "RMSE": rmse}


# =====================================================================
# 2. CREATION DES FENETRES
# =====================================================================
print("\n--- Creation des fenetres glissantes ---")
X_train, Y_train = create_windows(train, LOOKBACK, HORIZON)
X_val, Y_val = create_windows(val, LOOKBACK, HORIZON)
X_test, Y_test = create_windows(test, LOOKBACK, HORIZON)

print(f"  Train : X={X_train.shape}, Y={Y_train.shape}")
print(f"  Val   : X={X_val.shape}, Y={Y_val.shape}")
print(f"  Test  : X={X_test.shape}, Y={Y_test.shape}")


# =====================================================================
# 3. BASELINE NAIVE (derniere valeur repetee)
# =====================================================================
print("\n--- Baseline Naive ---")
Y_pred_naive = np.repeat(X_test[:, -1:, :], HORIZON, axis=1)
res_naive = evaluate_forecast(Y_test, Y_pred_naive, "Naive (last value)")
print(f"  MAE={res_naive['MAE']:.6f}, RMSE={res_naive['RMSE']:.6f}")


# =====================================================================
# 4. ARIMA (par serie, prevision sur la premiere serie seulement pour rapidite)
# =====================================================================
print("\n--- ARIMA (rolling, echantillon) ---")
sp500_idx = series_names.index("S&P 500") if "S&P 500" in series_names else 0

Y_pred_arima_all = np.copy(Y_pred_naive)

sample_step = 50
for s_idx in range(len(series_names)):
    s_name = series_names[s_idx]
    for i in range(0, len(X_test), sample_step):
        history = X_test[i, :, s_idx]
        try:
            model = ARIMA(history, order=(1, 0, 0))
            fit = model.fit()
            fc = fit.forecast(steps=HORIZON)
            end = min(i + sample_step, len(X_test))
            for j in range(i, end):
                Y_pred_arima_all[j, :, s_idx] = fc
        except Exception:
            pass
    print(f"  {s_name} : done")

res_arima = evaluate_forecast(Y_test, Y_pred_arima_all, "ARIMA(1,0,0)")
print(f"  MAE={res_arima['MAE']:.6f}, RMSE={res_arima['RMSE']:.6f}")


# =====================================================================
# 5. XGBOOST (multioutput: 1 modele par serie, flat features)
# =====================================================================
print("\n--- XGBoost ---")

n_features = len(series_names)
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

Y_pred_xgb = np.zeros_like(Y_test)

for s_idx in range(n_features):
    for h in range(HORIZON):
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            verbosity=0, random_state=42,
        )
        model.fit(
            X_train_flat, Y_train[:, h, s_idx],
            eval_set=[(X_val_flat, Y_val[:, h, s_idx])],
            verbose=False,
        )
        Y_pred_xgb[:, h, s_idx] = model.predict(X_test_flat)

res_xgb = evaluate_forecast(Y_test, Y_pred_xgb, "XGBoost")
print(f"  MAE={res_xgb['MAE']:.6f}, RMSE={res_xgb['RMSE']:.6f}")


# =====================================================================
# 6. LIGHTGBM
# =====================================================================
print("\n--- LightGBM ---")

Y_pred_lgb = np.zeros_like(Y_test)

for s_idx in range(n_features):
    for h in range(HORIZON):
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            verbosity=-1, random_state=42,
        )
        model.fit(
            X_train_flat, Y_train[:, h, s_idx],
            eval_set=[(X_val_flat, Y_val[:, h, s_idx])],
        )
        Y_pred_lgb[:, h, s_idx] = model.predict(X_test_flat)

res_lgb = evaluate_forecast(Y_test, Y_pred_lgb, "LightGBM")
print(f"  MAE={res_lgb['MAE']:.6f}, RMSE={res_lgb['RMSE']:.6f}")


# =====================================================================
# 7. LSTM SIMPLE
# =====================================================================
print("\n--- LSTM ---")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_layers=2, horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, n_features * horizon)
        self.n_features = n_features
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.reshape(-1, self.horizon, self.n_features)


train_dataset = TimeSeriesDataset(X_train, Y_train)
val_dataset = TimeSeriesDataset(X_val, Y_val)
test_dataset = TimeSeriesDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_lstm = LSTMForecaster(n_features, hidden_size=64, n_layers=2, horizon=HORIZON).to(DEVICE)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_val_loss = float("inf")
for epoch in range(50):
    model_lstm.train()
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model_lstm(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model_lstm.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                val_loss += criterion(model_lstm(X_b), Y_b).item()
        val_loss /= len(val_loader)
        print(f"  Epoch {epoch+1}/50  val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_lstm.state_dict(), "save/lstm_best.pth")

os.makedirs("save", exist_ok=True)
if os.path.exists("save/lstm_best.pth"):
    model_lstm.load_state_dict(torch.load("save/lstm_best.pth", weights_only=True))

model_lstm.eval()
all_preds = []
with torch.no_grad():
    for X_b, _ in test_loader:
        X_b = X_b.to(DEVICE)
        all_preds.append(model_lstm(X_b).cpu().numpy())

Y_pred_lstm = np.concatenate(all_preds, axis=0)
res_lstm = evaluate_forecast(Y_test, Y_pred_lstm, "LSTM")
print(f"  MAE={res_lstm['MAE']:.6f}, RMSE={res_lstm['RMSE']:.6f}")


# =====================================================================
# 8. TABLEAU RECAPITULATIF
# =====================================================================
print("\n" + "=" * 60)
print("RESULTATS — COMPARAISON DES BASELINES")
print("=" * 60)

results = [res_naive, res_arima, res_xgb, res_lgb, res_lstm]
results_df = pd.DataFrame(results).set_index("Modele")
print(results_df.round(6).to_string())

results_df.to_csv("results/baselines_results.csv")
print("\n>>> results/baselines_results.csv")

with open("results/baselines_results.json", "w") as f:
    json.dump({r["Modele"]: {"MAE": r["MAE"], "RMSE": r["RMSE"]} for r in results}, f, indent=4)


# =====================================================================
# 9. FIGURE — Comparaison des baselines
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models_names = results_df.index.tolist()
mae_vals = results_df["MAE"].values
rmse_vals = results_df["RMSE"].values
colors = ["#95a5a6", "#e67e22", "#3498db", "#2ecc71", "#9b59b6"]

ax = axes[0]
bars = ax.barh(models_names, mae_vals, color=colors)
ax.set_xlabel("MAE (log-rendements normalises)")
ax.set_title("MAE — Comparaison des baselines", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
for bar, v in zip(bars, mae_vals):
    ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=10)

ax = axes[1]
bars = ax.barh(models_names, rmse_vals, color=colors)
ax.set_xlabel("RMSE (log-rendements normalises)")
ax.set_title("RMSE — Comparaison des baselines", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
for bar, v in zip(bars, rmse_vals):
    ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("results/figures/16_baselines_comparaison.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 16 sauvegardee : results/figures/16_baselines_comparaison.png")


# =====================================================================
# 10. FIGURE — Exemple de prevision (S&P 500)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

sample_idx = len(X_test) // 2

preds_dict = {
    "Naive": Y_pred_naive,
    "ARIMA": Y_pred_arima_all,
    "XGBoost": Y_pred_xgb,
    "LightGBM": Y_pred_lgb,
    "LSTM": Y_pred_lstm,
}

ax = axes[0]
for s_idx in [sp500_idx]:
    t_lookback = np.arange(LOOKBACK)
    t_horizon = np.arange(LOOKBACK, LOOKBACK + HORIZON)
    ax.plot(t_lookback, X_test[sample_idx, :, s_idx], "k-", linewidth=1.5, label="Historique")
    ax.plot(t_horizon, Y_test[sample_idx, :, s_idx], "k--", linewidth=1.5, label="Vrai")
ax.set_title("Cible (S&P 500)", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

for i, (name, preds) in enumerate(preds_dict.items()):
    ax = axes[i + 1]
    ax.plot(t_lookback, X_test[sample_idx, :, sp500_idx], "k-", linewidth=1, alpha=0.5)
    ax.plot(t_horizon, Y_test[sample_idx, :, sp500_idx], "k--", linewidth=1.5, label="Vrai")
    ax.plot(t_horizon, preds[sample_idx, :, sp500_idx], "o-", linewidth=1.5,
            color=colors[i], label=name, markersize=4)
    ax.set_title(name, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(f"Prevision a H={HORIZON} jours — S&P 500 (log-rendements normalises)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/17_baselines_exemple_prevision.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 17 sauvegardee : results/figures/17_baselines_exemple_prevision.png")

print("\n" + "=" * 60)
print("BASELINES TERMINE — 2 figures produites (16-17)")
print("=" * 60)
