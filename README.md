# Sujet 3 — Modeles de diffusion pour la prevision probabiliste et l'imputation de series temporelles

**Cours :** Machine Learning — Ecole Centrale Casablanca, S9  
**Professeur :** Y. Lamrani Alaoui  
**Deadline :** 30 novembre 2026

## Objectif

Comparer plusieurs approches (dont les modeles de diffusion CSDI) pour :
1. **Imputation** : reconstruire des valeurs manquantes dans des series temporelles financieres
2. **Prevision probabiliste** : prevoir des series financieres avec quantification de l'incertitude

## Donnees

| Dataset | Series | Periode | Usage |
|---|---|---|---|
| Yahoo Finance | 9 (Apple, MSFT, JPMorgan, Exxon, CAC40, S&P500, GLD, VIX, EUR/USD) | 2010-2026 | Travail principal |
| Exchange Rate | 8 taux de change (AUD, GBP, CAD, CHF, CNY, JPY, NZD, SGD) | 1990-2010 | Validation |

**Decoupage temporel (sans fuite) :**
- Yahoo : Train 2010-2020 / Val 2021-2022 / Test 2023-2026
- Exchange Rate : 80% / 10% / 10%

**Preprocessing :** log-rendements r_t = log(p_t / p_{t-1}), standardisation sur train uniquement.

## Structure du projet

```
├── 01_nettoyage_eda.py          # EDA Yahoo Finance (figures 1-9)
├── 02_log_rendements_split.py   # Log-rendements, split, normalisation
├── 03_exchange_rate_prep.py     # EDA Exchange Rate (figures 10-13)
├── 04_prepare_csdi_data.py      # Conversion en format CSDI (pickle)
├── 05_run_csdi_exchange_rate.py # CSDI sur Exchange Rate (figure 14)
├── 05c_csdi_figure.py           # Generation figure 15 standalone
├── 06_baselines_forecasting.py  # 5 baselines (figures 16-17)
├── 07_shap_interpretabilite.py  # SHAP XGBoost/LightGBM (figures 18-22)
├── 08_run_csdi_yahoo.py         # CSDI sur Yahoo Finance (figures 23-25)
├── 09_diebold_mariano.py        # Test statistique DM (figure 26)
├── app.py                       # Application Streamlit
├── telecharger_brut.py          # Telechargement donnees Yahoo
├── CSDI/                        # Code source CSDI (Tashiro et al.)
├── config/                      # Configs YAML pour CSDI
├── data/
│   ├── processed/               # CSV nettoyes et normalises
│   ├── csdi_exchange_rate/      # Pickle pour CSDI
│   └── csdi_yahoo/              # Pickle pour CSDI
├── save/                        # Modeles entraines (.pth)
├── results/
│   ├── figures/                 # 26 figures PNG
│   ├── baselines_results.json
│   └── diebold_mariano.json
└── requirements.txt
```

## Modeles compares

| Modele | Type | Tache |
|---|---|---|
| Naive (derniere valeur) | Baseline | Prevision |
| ARIMA(1,0,0) | Classique | Prevision |
| XGBoost (200 arbres) | ML | Prevision |
| LightGBM (200 arbres) | ML | Prevision |
| LSTM (64h, 2 couches) | Deep Learning | Prevision |
| **CSDI** | **Diffusion** | **Imputation + Prevision** |

## Metriques

- **CRPS** : metrique principale probabiliste (plus bas = meilleur)
- **MAE, RMSE** : erreur ponctuelle
- **Test de Diebold-Mariano** : significativite statistique des differences

## Resultats cles

### Baselines (prevision Yahoo, lookback=20, horizon=5)

| Modele | MAE | RMSE |
|--------|-----|------|
| LightGBM | 0.663 | 0.940 |
| XGBoost | 0.664 | 0.942 |
| LSTM | 0.667 | 0.946 |
| ARIMA | 0.674 | 0.955 |
| Naive | 0.944 | 1.320 |

### CSDI Imputation (Exchange Rate)

| Masquage | RMSE | MAE | CRPS |
|----------|------|-----|------|
| 10% aleatoire | 0.0040 | 0.0020 | 0.0022 |
| 50% bloc | 0.0206 | 0.0117 | 0.0129 |

Le masquage bloc est ~3-4x plus difficile que l'aleatoire. La performance reste solide meme a 50% de donnees manquantes.

### Interpretabilite SHAP
Exxon est la serie la plus influente pour predire le S&P 500, suivie d'Apple et JPMorgan. Les lags recents (t-1, t-2) dominent.

## Installation et execution

```bash
git clone https://github.com/Jacquel-SANOU/ml-diffusion-timeseries-ecc.git
cd ml-diffusion-timeseries-ecc
pip install -r requirements.txt

# Lancer l'application Streamlit
streamlit run app.py

# Reproduire les resultats
python 01_nettoyage_eda.py
python 02_log_rendements_split.py
python 03_exchange_rate_prep.py
python 04_prepare_csdi_data.py
python 05_run_csdi_exchange_rate.py
python 06_baselines_forecasting.py
python 07_shap_interpretabilite.py
python 08_run_csdi_yahoo.py
python 09_diebold_mariano.py
```

## References

- Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
