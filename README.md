# Sujet 3 — Modèles de diffusion pour la prévision probabiliste et l'imputation de séries temporelles

**Cours :** Machine Learning — École Centrale Casablanca, S9  
**Deadline :** 30 novembre 2026

## Objectif

Comparer plusieurs approches (dont les modèles de diffusion CSDI) pour :
1. **Prévision probabiliste** : prévoir des séries financières avec une fourchette d'incertitude
2. **Imputation** : reconstruire des valeurs manquantes dans une série temporelle

## Données

| Dataset | Séries | Période | Usage |
|---|---|---|---|
| Yahoo Finance | 9 (Apple, MSFT, JPMorgan, Exxon, CAC40, SP500, GLD, VIX, EURUSD) | 2010–2026 | Travail principal |
| Exchange Rate | 8 taux de change | 1990–2010 | Validation contre article |

**Découpage temporel (sans fuite) :**
- Train : 2010–2020
- Validation : 2021–2022
- Test : 2023–2026

## Structure du projet

```
├── data/
│   ├── raw/           # Données brutes (non versionnées)
│   └── processed/     # Données nettoyées et normalisées
├── notebooks/
│   ├── 01_eda.ipynb           # Analyse exploratoire
│   ├── 02_baselines.ipynb     # Modèles de référence
│   └── 03_csdi.ipynb          # Modèle CSDI
├── src/
│   ├── data/          # Chargement et prétraitement
│   ├── models/        # Baselines + CSDI wrapper
│   ├── evaluation/    # Métriques (CRPS, MAE, RMSE, coverage)
│   └── utils/
├── CSDI/              # Dépôt officiel CSDI (submodule)
├── results/
│   ├── figures/
│   └── metrics/
├── app/               # Application Streamlit
├── requirements.txt
└── README.md
```

## Modèles comparés

| Niveau | Modèle | Tâche |
|---|---|---|
| Baseline simple | Naïve (dernière valeur), interpolation linéaire | Prévision + imputation |
| Classique | ARIMA, GARCH | Prévision |
| Deep Learning | LSTM / XGBoost / LightGBM | Prévision |
| **Principal** | **CSDI (modèle de diffusion)** | **Prévision + imputation** |

## Métriques

- **CRPS** : métrique principale pour l'incertitude (plus bas = meilleur)
- **MAE, RMSE** : prévision ponctuelle et imputation
- **Coverage** : % des vraies valeurs dans l'intervalle prédit
- **Largeur d'intervalle** : à coverage égale, plus étroit = meilleur

## Installation

```bash
# Cloner le repo
git clone https://github.com/Jacquel-SANOU/<repo-name>.git
cd <repo-name>

# Installer les dépendances
pip install -r requirements.txt

# Cloner CSDI
git submodule update --init --recursive
```

## Lancer l'application

```bash
streamlit run app/main.py
```
