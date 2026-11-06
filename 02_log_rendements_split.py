"""
Etape 2 : Calcul des log-rendements + Split temporel Train/Val/Test

Comme vu dans le Cours 2 (Prof. Lamrani Alaoui) :
  - On travaille sur des log-rendements : r_t = log(prix_t / prix_{t-1})
  - Split temporel (Hold-Out temporel) : jamais de melange aleatoire
  - Normalisation (centrage-reduction) calculee UNIQUEMENT sur le train

Produit :
  - data/processed/yahoo_log_returns.csv
  - data/processed/train.csv, val.csv, test.csv  (log-rendements normalises)
  - data/processed/norm_params.csv  (moyenne et ecart-type du train)
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

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#d35400"]

# =====================================================================
# 1. CHARGEMENT DES PRIX NETTOYES
# =====================================================================
print("=" * 60)
print("ETAPE 1 : Chargement des prix nettoyes")
print("=" * 60)

prix = pd.read_csv("data/processed/yahoo_prix_clean.csv", index_col="Date", parse_dates=True)
print(f"Dimensions : {prix.shape}")
print(f"Periode : {prix.index[0].date()} -> {prix.index[-1].date()}")

# =====================================================================
# 2. CALCUL DES LOG-RENDEMENTS
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 2 : Calcul des log-rendements")
print("  r_t = log(prix_t / prix_{t-1})")
print("=" * 60)

log_ret = np.log(prix / prix.shift(1))
log_ret = log_ret.dropna()

print(f"Dimensions : {log_ret.shape}")
print(f"Periode : {log_ret.index[0].date()} -> {log_ret.index[-1].date()}")
print(f"\nApercu (5 premieres lignes) :")
print(log_ret.head().round(6).to_string())

log_ret.to_csv("data/processed/yahoo_log_returns.csv")
print("\n>>> Fichier sauvegarde : data/processed/yahoo_log_returns.csv")

# --- FIGURE 5 : Log-rendements des 9 series ---
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(log_ret.columns):
    ax = axes[i]
    ax.plot(log_ret.index, log_ret[col], color=colors[i], linewidth=0.3, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title(col, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Log-rendements journaliers — 9 series financieres", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/05_log_rendements_9series.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 5 sauvegardee : results/figures/05_log_rendements_9series.png")

# --- FIGURE 6 : Distribution des log-rendements (histogrammes + QQ) ---
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

print("\n" + "-" * 60)
print("Statistiques des log-rendements :")
print("-" * 60)
stats_table = []

for i, col in enumerate(log_ret.columns):
    ax = axes[i]
    data = log_ret[col].dropna()

    ax.hist(data, bins=80, density=True, alpha=0.7, color=colors[i], edgecolor="white", linewidth=0.3)

    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", linewidth=1.5, label="Gaussienne")

    skew_val = data.skew()
    kurt_val = data.kurtosis()
    ax.set_title(f"{col}\nskew={skew_val:.2f}, kurtosis={kurt_val:.2f}", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    stats_table.append({
        "Serie": col,
        "Moyenne": f"{mu:.6f}",
        "Ecart-type": f"{sigma:.4f}",
        "Skewness": f"{skew_val:.3f}",
        "Kurtosis": f"{kurt_val:.2f}",
        "Min": f"{data.min():.4f}",
        "Max": f"{data.max():.4f}",
    })

fig.suptitle("Distribution des log-rendements vs loi normale", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/06_distribution_log_rendements.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 6 sauvegardee : results/figures/06_distribution_log_rendements.png")

stats_df = pd.DataFrame(stats_table)
print("\n" + stats_df.to_string(index=False))
print("\nNote : kurtosis > 0 = queues lourdes (leptokurtique)")
print("       Typique en finance : les evenements extremes sont plus frequents qu'une gaussienne")

# --- FIGURE 7 : Matrice de correlation des LOG-RENDEMENTS ---
fig, ax = plt.subplots(figsize=(10, 8))
corr_ret = log_ret.corr()
mask = np.triu(np.ones_like(corr_ret, dtype=bool))
sns.heatmap(
    corr_ret, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5, ax=ax,
    vmin=-1, vmax=1,
)
ax.set_title("Matrice de correlation des log-rendements", fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/07_correlation_log_rendements.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 7 sauvegardee : results/figures/07_correlation_log_rendements.png")

# =====================================================================
# 3. SPLIT TEMPOREL TRAIN / VALIDATION / TEST
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 3 : Split temporel (Hold-Out temporel)")
print("  Comme vu en cours : respect de l'ordre chronologique")
print("  Train : 2010-2020 | Val : 2021-2022 | Test : 2023-2026")
print("=" * 60)

train = log_ret[log_ret.index < "2021-01-01"]
val = log_ret[(log_ret.index >= "2021-01-01") & (log_ret.index < "2023-01-01")]
test = log_ret[log_ret.index >= "2023-01-01"]

print(f"\nTrain : {train.index[0].date()} -> {train.index[-1].date()}  ({len(train)} jours)")
print(f"Val   : {val.index[0].date()} -> {val.index[-1].date()}  ({len(val)} jours)")
print(f"Test  : {test.index[0].date()} -> {test.index[-1].date()}  ({len(test)} jours)")
print(f"Total : {len(train) + len(val) + len(test)} jours")

# =====================================================================
# 4. NORMALISATION (centrage-reduction) SUR LE TRAIN UNIQUEMENT
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 4 : Normalisation (standardisation)")
print("  Comme vu en cours : x_norm = (x - moyenne_train) / sigma_train")
print("  IMPORTANT : moyenne et sigma calcules uniquement sur le train")
print("  => evite toute fuite de donnees (data leakage)")
print("=" * 60)

train_mean = train.mean()
train_std = train.std()

norm_params = pd.DataFrame({"mean": train_mean, "std": train_std})
norm_params.to_csv("data/processed/norm_params.csv")
print("\nParametres de normalisation (calcules sur le train) :")
print(norm_params.round(6).to_string())

train_norm = (train - train_mean) / train_std
val_norm = (val - train_mean) / train_std
test_norm = (test - train_mean) / train_std

train_norm.to_csv("data/processed/train.csv")
val_norm.to_csv("data/processed/val.csv")
test_norm.to_csv("data/processed/test.csv")

print("\n>>> Fichiers sauvegardes :")
print("    data/processed/train.csv")
print("    data/processed/val.csv")
print("    data/processed/test.csv")
print("    data/processed/norm_params.csv")

# --- FIGURE 8 : Visualisation du split temporel ---
fig, ax = plt.subplots(figsize=(16, 5))

ax.axvspan(train.index[0], train.index[-1], alpha=0.2, color="#2ecc71", label=f"Train ({len(train)} jours)")
ax.axvspan(val.index[0], val.index[-1], alpha=0.2, color="#f39c12", label=f"Validation ({len(val)} jours)")
ax.axvspan(test.index[0], test.index[-1], alpha=0.2, color="#e74c3c", label=f"Test ({len(test)} jours)")

ax.plot(prix.index, prix["S&P 500"], color="#34495e", linewidth=0.8)
ax.set_title("Split temporel — Train / Validation / Test (exemple : S&P 500)", fontweight="bold")
ax.set_ylabel("Prix S&P 500")
ax.legend(loc="upper left", fontsize=12)
ax.grid(alpha=0.3)

for boundary, label in [("2021-01-01", "2021"), ("2023-01-01", "2023")]:
    ax.axvline(pd.Timestamp(boundary), color="black", linestyle="--", linewidth=1.5)

plt.tight_layout()
plt.savefig("results/figures/08_split_temporel.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 8 sauvegardee : results/figures/08_split_temporel.png")

# --- FIGURE 9 : Log-rendements normalises (verification) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, data, color) in zip(axes, [
    ("Train (normalise)", train_norm, "#2ecc71"),
    ("Validation (normalise)", val_norm, "#f39c12"),
    ("Test (normalise)", test_norm, "#e74c3c"),
]):
    for col in data.columns:
        ax.plot(data.index, data[col], linewidth=0.3, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(name, fontweight="bold", color=color)
    ax.set_ylabel("Log-rendement normalise")
    ax.grid(alpha=0.3)

plt.suptitle("Log-rendements normalises par ensemble", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/09_log_rendements_normalises.png", bbox_inches="tight")
plt.close()
print(">>> FIGURE 9 sauvegardee : results/figures/09_log_rendements_normalises.png")

# =====================================================================
# 5. VERIFICATION : PAS DE FUITE DE DONNEES
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 5 : Verification — pas de fuite de donnees")
print("=" * 60)

print(f"\nMoyenne du train normalise (doit etre ~0) :")
print(train_norm.mean().round(6).to_string())
print(f"\nEcart-type du train normalise (doit etre ~1) :")
print(train_norm.std().round(6).to_string())
print(f"\nMoyenne du val normalise (peut devier de 0) :")
print(val_norm.mean().round(4).to_string())
print(f"\nMoyenne du test normalise (peut devier de 0) :")
print(test_norm.mean().round(4).to_string())

print("\n>>> Si val/test devient de 0, c'est NORMAL : la distribution")
print("    peut changer entre periodes (concept drift en finance)")

print("\n" + "=" * 60)
print("LOG-RENDEMENTS + SPLIT TERMINE — 5 figures produites")
print("=" * 60)
