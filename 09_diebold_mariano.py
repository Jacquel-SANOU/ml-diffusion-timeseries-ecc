"""
09 — Test de Diebold-Mariano
Compare les erreurs de prevision des baselines deux a deux.
H0 : les deux modeles ont la meme precision de prevision.
Produit figure 26.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os
import pickle
from sklearn.metrics import mean_squared_error

# ── Reload baseline predictions ─────────────────────────
# We need per-window errors. Re-run baselines quickly to get them.

df_lr = pd.read_csv("data/processed/yahoo_log_returns.csv", index_col=0, parse_dates=True)
norm_params = pd.read_csv("data/processed/norm_params.csv", index_col=0)
train_mean = norm_params["mean"].values
train_std = norm_params["std"].values
df_norm = (df_lr - train_mean) / train_std

# Train/Val/Test split
train = df_norm[df_lr.index < "2021-01-01"]
test = df_norm[df_lr.index >= "2023-01-01"]

L, H = 20, 5
columns = list(df_norm.columns)
n_features = len(columns)

# Sliding windows on test
test_vals = test.values
windows = []
for i in range(len(test_vals) - L - H + 1):
    X = test_vals[i:i + L]
    Y = test_vals[i + L:i + L + H]
    windows.append((X, Y))

print(f"Test windows: {len(windows)}")

# ── Naive predictions ───────────────────────────────────
naive_errors = []
for X, Y in windows:
    pred = np.tile(X[-1], (H, 1))
    naive_errors.append(((pred - Y)**2).mean())
naive_errors = np.array(naive_errors)

# ── Zero baseline ─────────────────────────────────────
zero_errors = []
for X, Y in windows:
    pred = np.zeros_like(Y)
    zero_errors.append(((pred - Y)**2).mean())
zero_errors = np.array(zero_errors)

# ── ARIMA (AR(1), refit par fenetre) ──────────────────
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings("ignore")

arima_errors = []
for wi, (X, Y) in enumerate(windows):
    preds = []
    for col in range(n_features):
        try:
            model = AutoReg(X[:, col], lags=1).fit()
            fc = model.predict(start=L, end=L + H - 1)
            preds.append(fc)
        except Exception:
            preds.append(np.full(H, X[-1, col]))
    preds = np.array(preds).T
    arima_errors.append(((preds - Y)**2).mean())
    if (wi + 1) % 200 == 0:
        print(f"  ARIMA : {wi+1}/{len(windows)}")
arima_errors = np.array(arima_errors)

# ── XGBoost ────────────────────────────────────────────
import xgboost as xgb

train_vals = train.values
X_train_list, Y_train_list = [], []
for i in range(len(train_vals) - L - H + 1):
    X_train_list.append(train_vals[i:i + L].flatten())
    Y_train_list.append(train_vals[i + L:i + L + H].flatten())

X_train_arr = np.array(X_train_list)
Y_train_arr = np.array(Y_train_list)

xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                               n_jobs=-1, random_state=42, verbosity=0,
                               multi_strategy="multi_output_tree")
xgb_model.fit(X_train_arr, Y_train_arr)

xgb_errors = []
for X, Y in windows:
    pred = xgb_model.predict(X.flatten().reshape(1, -1)).reshape(H, n_features)
    xgb_errors.append(((pred - Y)**2).mean())
xgb_errors = np.array(xgb_errors)

# ── LightGBM ──────────────────────────────────────────
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

lgb_base = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              n_jobs=-1, random_state=42, verbosity=-1)
lgb_model = MultiOutputRegressor(lgb_base, n_jobs=-1)
lgb_model.fit(X_train_arr, Y_train_arr)

lgb_errors = []
for X, Y in windows:
    pred = lgb_model.predict(X.flatten().reshape(1, -1)).reshape(H, n_features)
    lgb_errors.append(((pred - Y)**2).mean())
lgb_errors = np.array(lgb_errors)

# ── LSTM ───────────────────────────────────────────────
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model = LSTMModel(n_features, 64, H * n_features, 2).to(device)

X_train_t = torch.tensor(
    np.array([train_vals[i:i+L] for i in range(len(train_vals)-L-H+1)]),
    dtype=torch.float32
).to(device)
Y_train_t = torch.tensor(Y_train_arr, dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

lstm_model.train()
for epoch in range(50):
    for start in range(0, len(X_train_t), 64):
        end = min(start + 64, len(X_train_t))
        pred = lstm_model(X_train_t[start:end])
        loss = loss_fn(pred, Y_train_t[start:end])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

lstm_model.eval()
lstm_errors = []
with torch.no_grad():
    for X, Y in windows:
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        pred = lstm_model(X_t).cpu().numpy().reshape(H, n_features)
        lstm_errors.append(((pred - Y)**2).mean())
lstm_errors = np.array(lstm_errors)

# ── Diebold-Mariano Test ───────────────────────────────
def diebold_mariano(e1, e2, h=1):
    d = e1 - e2
    n = len(d)
    d_mean = d.mean()
    gamma_0 = np.var(d, ddof=0)
    # HAC variance for h-step ahead
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += gamma_k
    var_d = (gamma_0 + 2 * gamma_sum) / n
    if var_d <= 0:
        var_d = gamma_0 / n
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

model_names = ["Zero", "Naive", "ARIMA", "XGBoost", "LightGBM", "LSTM"]
model_errors = {
    "Zero": zero_errors,
    "Naive": naive_errors,
    "ARIMA": arima_errors,
    "XGBoost": xgb_errors,
    "LightGBM": lgb_errors,
    "LSTM": lstm_errors,
}

n_models = len(model_names)
dm_matrix = np.zeros((n_models, n_models))
pval_matrix = np.ones((n_models, n_models))

for i in range(n_models):
    for j in range(n_models):
        if i != j:
            dm_stat, p_val = diebold_mariano(
                model_errors[model_names[i]],
                model_errors[model_names[j]],
                h=H
            )
            dm_matrix[i, j] = dm_stat
            pval_matrix[i, j] = p_val

print("\n=== Test de Diebold-Mariano (H0: meme precision) ===\n")
print(f"{'':>12}", end="")
for name in model_names:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(model_names):
    print(f"{name:>12}", end="")
    for j in range(n_models):
        if i == j:
            print(f"{'---':>12}", end="")
        else:
            stars = ""
            if pval_matrix[i, j] < 0.01:
                stars = "***"
            elif pval_matrix[i, j] < 0.05:
                stars = "**"
            elif pval_matrix[i, j] < 0.1:
                stars = "*"
            print(f"{dm_matrix[i, j]:>8.2f}{stars:>4}", end="")
    print()

print("\n*** p<0.01, ** p<0.05, * p<0.1")
print("\nPositif = modele ligne a plus d'erreur que modele colonne")

# Save results
dm_results = {"models": model_names, "dm_stats": dm_matrix.tolist(),
              "p_values": pval_matrix.tolist()}
os.makedirs("results", exist_ok=True)
with open("results/diebold_mariano.json", "w") as f:
    json.dump(dm_results, f, indent=2)

# ── Figure 26 — DM heatmap ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.imshow(dm_matrix, cmap="RdBu_r", aspect="auto",
                  vmin=-max(abs(dm_matrix.min()), abs(dm_matrix.max())),
                  vmax=max(abs(dm_matrix.min()), abs(dm_matrix.max())))
ax1.set_xticks(range(n_models))
ax1.set_yticks(range(n_models))
ax1.set_xticklabels(model_names, rotation=45, ha="right")
ax1.set_yticklabels(model_names)
ax1.set_title("Statistique DM", fontweight="bold")
for i in range(n_models):
    for j in range(n_models):
        if i != j:
            ax1.text(j, i, f"{dm_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
        else:
            ax1.text(j, i, "—", ha="center", va="center", fontsize=9)
plt.colorbar(im1, ax=ax1)

sig_matrix = np.zeros_like(pval_matrix)
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            sig_matrix[i, j] = np.nan
        elif pval_matrix[i, j] < 0.01:
            sig_matrix[i, j] = 3
        elif pval_matrix[i, j] < 0.05:
            sig_matrix[i, j] = 2
        elif pval_matrix[i, j] < 0.1:
            sig_matrix[i, j] = 1
        else:
            sig_matrix[i, j] = 0

im2 = ax2.imshow(sig_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
ax2.set_xticks(range(n_models))
ax2.set_yticks(range(n_models))
ax2.set_xticklabels(model_names, rotation=45, ha="right")
ax2.set_yticklabels(model_names)
ax2.set_title("Significativite (p-value)", fontweight="bold")
labels_sig = {0: "NS", 1: "p<0.1", 2: "p<0.05", 3: "p<0.01"}
for i in range(n_models):
    for j in range(n_models):
        if i != j:
            ax2.text(j, i, f"p={pval_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)
        else:
            ax2.text(j, i, "—", ha="center", va="center", fontsize=9)
plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3],
             format=plt.FuncFormatter(lambda x, _: labels_sig.get(int(x), "")))

fig.suptitle("Test de Diebold-Mariano — Comparaison des baselines",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/26_diebold_mariano.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 26 sauvegardee")
