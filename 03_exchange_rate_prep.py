"""
Etape 3 : Preparation du dataset Exchange Rate pour validation CSDI

Ce dataset (8 taux de change journaliers, 1990-2010) est utilise
dans le papier CSDI original. On le prepare pour :
  1. Valider notre implementation CSDI contre les resultats publies
  2. Tester l'imputation avec masques artificiels

Produit :
  - data/processed/exchange_rate_clean.csv
  - data/processed/exchange_rate_train.csv, val.csv, test.csv
  - data/processed/exchange_rate_norm_params.csv
  - Figures pour le rapport
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "figure.dpi": 120,
})

NOMS_SERIES = {
    "0": "Australie", "1": "Angleterre", "2": "Canada",
    "3": "Suisse", "4": "Chine", "5": "Japon",
    "6": "Nouvelle-Zelande", "OT": "Singapour",
}

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

# =====================================================================
# 1. CHARGEMENT
# =====================================================================
print("=" * 60)
print("ETAPE 1 : Chargement du dataset Exchange Rate")
print("=" * 60)

df = pd.read_csv("exchange_rate.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
df.columns = [NOMS_SERIES.get(c, c) for c in df.columns]

print(f"Dimensions : {df.shape}")
print(f"Periode : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Series : {list(df.columns)}")
print(f"\nValeurs manquantes : {df.isna().sum().sum()}")

# =====================================================================
# 2. EDA — VISUALISATION
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 2 : Analyse exploratoire")
print("=" * 60)

print("\nStatistiques descriptives :")
print(df.describe().round(6).to_string())

# --- FIGURE 10 : Evolution des 8 taux de change ---
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    ax = axes[i]
    ax.plot(df.index, df[col], color=colors[i], linewidth=0.6)
    ax.set_title(col, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Exchange Rate — 8 taux de change (1990-2010)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/10_exchange_rate_series.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 10 sauvegardee : results/figures/10_exchange_rate_series.png")

# --- FIGURE 11 : Correlation des taux de change ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5, ax=ax,
    vmin=-1, vmax=1,
)
ax.set_title("Correlation des taux de change", fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/11_exchange_rate_correlation.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 11 sauvegardee : results/figures/11_exchange_rate_correlation.png")

# --- FIGURE 12 : Taux normalises (base 100) ---
fig, ax = plt.subplots(figsize=(16, 6))
norm_base = df.div(df.iloc[0]) * 100

for i, col in enumerate(norm_base.columns):
    ax.plot(norm_base.index, norm_base[col], label=col, color=colors[i], linewidth=0.8)

ax.set_title("Taux de change normalises (base 100)", fontweight="bold")
ax.set_ylabel("Valeur normalisee")
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="best", ncol=4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/12_exchange_rate_normalise.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 12 sauvegardee : results/figures/12_exchange_rate_normalise.png")

# =====================================================================
# 3. SPLIT TEMPOREL (comme dans le papier CSDI)
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 3 : Split temporel")
print("  Papier CSDI : ~80% train, ~10% val, ~10% test")
print("=" * 60)

n = len(df)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

train = df.iloc[:n_train]
val = df.iloc[n_train:n_train + n_val]
test = df.iloc[n_train + n_val:]

print(f"\nTrain : {train.index[0].date()} -> {train.index[-1].date()}  ({len(train)} jours)")
print(f"Val   : {val.index[0].date()} -> {val.index[-1].date()}  ({len(val)} jours)")
print(f"Test  : {test.index[0].date()} -> {test.index[-1].date()}  ({len(test)} jours)")
print(f"Total : {len(train) + len(val) + len(test)} jours")

# =====================================================================
# 4. NORMALISATION SUR LE TRAIN
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 4 : Normalisation (centrage-reduction sur le train)")
print("=" * 60)

train_mean = train.mean()
train_std = train.std()

norm_params = pd.DataFrame({"mean": train_mean, "std": train_std})
norm_params.to_csv("data/processed/exchange_rate_norm_params.csv")

train_norm = (train - train_mean) / train_std
val_norm = (val - train_mean) / train_std
test_norm = (test - train_mean) / train_std

print("\nParametres de normalisation :")
print(norm_params.round(6).to_string())

# =====================================================================
# 5. SAUVEGARDE
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 5 : Sauvegarde")
print("=" * 60)

df.to_csv("data/processed/exchange_rate_clean.csv")
train_norm.to_csv("data/processed/exchange_rate_train.csv")
val_norm.to_csv("data/processed/exchange_rate_val.csv")
test_norm.to_csv("data/processed/exchange_rate_test.csv")

print(">>> data/processed/exchange_rate_clean.csv")
print(">>> data/processed/exchange_rate_train.csv")
print(">>> data/processed/exchange_rate_val.csv")
print(">>> data/processed/exchange_rate_test.csv")
print(">>> data/processed/exchange_rate_norm_params.csv")

# --- FIGURE 13 : Split temporel visualise ---
fig, ax = plt.subplots(figsize=(16, 5))

ax.axvspan(train.index[0], train.index[-1], alpha=0.2, color="#2ecc71",
           label=f"Train ({len(train)} jours)")
ax.axvspan(val.index[0], val.index[-1], alpha=0.2, color="#f39c12",
           label=f"Validation ({len(val)} jours)")
ax.axvspan(test.index[0], test.index[-1], alpha=0.2, color="#e74c3c",
           label=f"Test ({len(test)} jours)")

ax.plot(df.index, df["Singapour"], color="#34495e", linewidth=0.8)
ax.set_title("Split temporel — Exchange Rate (exemple : Singapour)", fontweight="bold")
ax.set_ylabel("Taux de change")
ax.legend(loc="upper left", fontsize=12)
ax.grid(alpha=0.3)

ax.axvline(train.index[-1], color="black", linestyle="--", linewidth=1.5)
ax.axvline(val.index[-1], color="black", linestyle="--", linewidth=1.5)

plt.tight_layout()
plt.savefig("results/figures/13_exchange_rate_split.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 13 sauvegardee : results/figures/13_exchange_rate_split.png")

# =====================================================================
# 6. VERIFICATION
# =====================================================================
print("\n" + "=" * 60)
print("VERIFICATION — pas de fuite de donnees")
print("=" * 60)

print(f"\nMoyenne du train normalise (doit etre ~0) :")
print(train_norm.mean().round(6).to_string())
print(f"\nEcart-type du train normalise (doit etre ~1) :")
print(train_norm.std().round(6).to_string())

print("\n" + "=" * 60)
print("EXCHANGE RATE PREPARE — 4 figures produites (10 a 13)")
print("=" * 60)
