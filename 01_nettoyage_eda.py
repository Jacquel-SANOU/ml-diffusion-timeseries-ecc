"""
Etape 1 : Nettoyage des donnees Yahoo Finance + Analyse Exploratoire (EDA)

Produit :
  - yahoo_prix_clean.csv : prix de cloture ajustes, NaN supprimes
  - Graphiques pour le rapport
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "figure.dpi": 120,
})

NOMS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "JPM": "JPMorgan",
    "XOM": "Exxon", "GLD": "Or (GLD)", "^GSPC": "S&P 500",
    "^FCHI": "CAC 40", "^VIX": "VIX", "EURUSD=X": "EUR/USD",
}

# =====================================================================
# 1. CHARGEMENT DES DONNEES BRUTES
# =====================================================================
print("=" * 60)
print("ETAPE 1 : Chargement des donnees brutes Yahoo Finance")
print("=" * 60)

df_brut = pd.read_csv("yahoo_brut.csv", header=[0, 1], index_col=0)
df_brut.index = pd.to_datetime(df_brut.index)

print(f"Dimensions brutes : {df_brut.shape}")
print(f"Periode : {df_brut.index.min().date()} -> {df_brut.index.max().date()}")
print(f"\nColonnes (niveaux) :")
print(f"  Niveau 1 (Price) : {df_brut.columns.get_level_values(0).unique().tolist()}")
print(f"  Niveau 2 (Ticker) : {df_brut.columns.get_level_values(1).unique().tolist()}")

# =====================================================================
# 2. EXTRACTION DES PRIX DE CLOTURE AJUSTES
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 2 : Extraction des prix de cloture ajustes (Adj Close)")
print("=" * 60)

prix = df_brut["Adj Close"].copy()
prix.columns = [NOMS.get(c, c) for c in prix.columns]
prix = prix.sort_index()

print(f"Dimensions apres extraction : {prix.shape}")
print(f"\nSeries extraites : {list(prix.columns)}")

# =====================================================================
# 3. ANALYSE DES VALEURS MANQUANTES (AVANT nettoyage)
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 3 : Analyse des valeurs manquantes")
print("=" * 60)

na_count = prix.isna().sum()
na_pct = (prix.isna().sum() / len(prix) * 100).round(2)
na_df = pd.DataFrame({"Manquantes": na_count, "% Manquant": na_pct})
print(na_df.to_string())

# --- FIGURE 1 : Heatmap des valeurs manquantes ---
fig, ax = plt.subplots(figsize=(16, 5))
# Downsample for readability
na_matrix = prix.isna().astype(int)
# Show 1 row per week for better visibility
weekly = na_matrix.resample("W").max()
sns.heatmap(
    weekly.T, cmap=["#2ecc71", "#e74c3c"], cbar=False,
    yticklabels=True, xticklabels=False, ax=ax,
)
ax.set_title("Valeurs manquantes par semaine (vert = present, rouge = manquant)")
ax.set_xlabel("Temps (semaines)")
ax.set_ylabel("Serie")
plt.tight_layout()
plt.savefig("results/figures/01_valeurs_manquantes_heatmap.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 1 sauvegardee : results/figures/01_valeurs_manquantes_heatmap.png")

# =====================================================================
# 4. SUPPRESSION DES JOURS OU TOUTES LES SERIES NE SONT PAS DISPONIBLES
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 4 : Nettoyage — suppression des jours incomplets")
print("=" * 60)

n_avant = len(prix)
prix_clean = prix.dropna()
n_apres = len(prix_clean)

print(f"Jours avant nettoyage : {n_avant}")
print(f"Jours apres nettoyage : {n_apres}")
print(f"Jours supprimes       : {n_avant - n_apres}")
print(f"Periode finale        : {prix_clean.index[0].date()} -> {prix_clean.index[-1].date()}")
print(f"\nVerification NaN restants : {prix_clean.isna().sum().sum()}")

# =====================================================================
# 5. STATISTIQUES DESCRIPTIVES
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 5 : Statistiques descriptives des prix")
print("=" * 60)

stats = prix_clean.describe().round(4)
print(stats.to_string())

# --- FIGURE 2 : Evolution des prix (toutes les series) ---
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#d35400"]

for i, col in enumerate(prix_clean.columns):
    ax = axes[i]
    ax.plot(prix_clean.index, prix_clean[col], color=colors[i], linewidth=0.8)
    ax.set_title(col, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Evolution des prix — 9 series financieres (2010-2026)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/figures/02_evolution_prix_9series.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 2 sauvegardee : results/figures/02_evolution_prix_9series.png")

# --- FIGURE 3 : Prix normalises (base 100) pour comparaison ---
fig, ax = plt.subplots(figsize=(16, 7))
prix_norm = prix_clean.div(prix_clean.iloc[0]) * 100

for i, col in enumerate(prix_norm.columns):
    ax.plot(prix_norm.index, prix_norm[col], label=col, color=colors[i], linewidth=1)

ax.set_title("Prix normalises (base 100) — Comparaison des 9 series", fontweight="bold")
ax.set_ylabel("Valeur normalisee (base 100)")
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="upper left", ncol=3)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("results/figures/03_prix_normalises_base100.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 3 sauvegardee : results/figures/03_prix_normalises_base100.png")

# --- FIGURE 4 : Matrice de correlation ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = prix_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5, ax=ax,
    vmin=-1, vmax=1,
)
ax.set_title("Matrice de correlation des prix", fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/04_correlation_prix.png", bbox_inches="tight")
plt.close()
print("\n>>> FIGURE 4 sauvegardee : results/figures/04_correlation_prix.png")

# =====================================================================
# 6. SAUVEGARDE DU FICHIER NETTOYE
# =====================================================================
print("\n" + "=" * 60)
print("ETAPE 6 : Sauvegarde du fichier nettoye")
print("=" * 60)

prix_clean.index.name = "Date"
prix_clean.to_csv("data/processed/yahoo_prix_clean.csv")
print(f"Fichier sauvegarde : data/processed/yahoo_prix_clean.csv")
print(f"Dimensions finales : {prix_clean.shape}")
print(f"\nApercu (5 premieres lignes) :")
print(prix_clean.head().to_string())

print("\n" + "=" * 60)
print("NETTOYAGE TERMINE — 4 figures produites pour le rapport")
print("=" * 60)
