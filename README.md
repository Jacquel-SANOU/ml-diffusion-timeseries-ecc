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
├── 05_run_csdi_exchange_rate.py # CSDI sur Exchange Rate (figure 14-15)
├── 06_baselines_forecasting.py  # 6 baselines + zero (figures 16-17)
├── 07_baselines_imputation.py   # Baselines imputation (zero, interpolation, gaussienne cond.)
├── 07_shap_interpretabilite.py  # SHAP XGBoost/LightGBM (figures 18-22)
├── 08_run_csdi_yahoo.py         # CSDI sur Yahoo Finance (figures 23-25)
├── 09_diebold_mariano.py        # Test statistique DM avec correction HAC (figure 26)
├── app.py                       # Application Streamlit interactive
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
│   ├── baselines_imputation.json
│   └── diebold_mariano.json
└── requirements.txt
```

## Application Streamlit

L'application interactive comprend 9 pages :

| Page | Description |
|------|-------------|
| Accueil | Presentation du projet et de la methodologie |
| Donnees Yahoo Finance | EDA : prix, log-rendements, correlations, split temporel |
| Donnees Exchange Rate | EDA : series, correlations, normalisation |
| Baselines (Prevision) | Resultats des 6 modeles de reference (incluant le zero) |
| CSDI — Imputation | Resultats d'imputation avec courbes d'apprentissage |
| Interpretabilite SHAP | Analyse SHAP : importance globale, par serie, par lag |
| Comparaison globale | Synthese + test de Diebold-Mariano |
| **Prevision interactive** | Prevision en prix reels avec intervalles de confiance (IC 80%) et explication SHAP |
| **Imputation interactive** | Imputation CSDI en temps reel avec choix du masquage |

## Modeles compares

| Modele | Type | Tache | Probabiliste |
|---|---|---|---|
| Zero (moyenne) | Baseline | Prevision + Imputation | Non |
| Naive (derniere valeur) | Baseline | Prevision | Non |
| ARIMA(1,0,0) | Classique | Prevision | Non |
| Interpolation lineaire | Baseline | Imputation | Non |
| Gaussienne conditionnelle | Baseline | Imputation | Non |
| XGBoost (100 arbres) | ML | Prevision | Oui (regression quantile) |
| LightGBM (100 arbres) | ML | Prevision | Oui (regression quantile) |
| LSTM (64h, 2 couches) | Deep Learning | Prevision | Non |
| **CSDI** | **Diffusion** | **Imputation** | **Oui (echantillonnage natif)** |

## Metriques

- **CRPS** : metrique principale probabiliste (plus bas = meilleur)
- **MAE, RMSE** : erreur ponctuelle
- **Couverture IC 80%** : proportion des vraies valeurs dans [q10, q90] (cible = 80%)
- **Test de Diebold-Mariano** : significativite statistique (correction HAC Newey-West + Harvey et al. 1997)

## Resultats cles

### Baselines (prevision Yahoo, lookback=20, horizon=5)

| Modele | MAE | RMSE |
|--------|-----|------|
| **Zero** | **0.650** | **0.928** |
| LightGBM | 0.663 | 0.940 |
| XGBoost | 0.664 | 0.942 |
| LSTM | 0.667 | 0.947 |
| ARIMA | 0.678 | 0.964 |
| Naive | 0.944 | 1.320 |

Aucun modele appris ne bat le zero (+1.9% a +45%). Resultat coherent avec l'absence d'autocorrelation des log-rendements (Cont, 2001).

### CSDI Imputation (Exchange Rate)

| Masquage | RMSE | MAE | CRPS |
|----------|------|-----|------|
| 10% aleatoire | 0.0030 | 0.0017 | 0.0019 |
| 30% aleatoire | 0.0035 | 0.0019 | 0.0022 |
| 50% aleatoire | 0.0041 | 0.0023 | 0.0026 |
| 10% bloc | 0.0073 | 0.0045 | 0.0051 |
| 30% bloc | 0.0126 | 0.0078 | 0.0088 |
| 50% bloc | 0.0154 | 0.0096 | 0.0107 |

CSDI bat l'interpolation en masquage aleatoire ; l'interpolation reprend l'avantage en bloc.

### CSDI Imputation (Yahoo Finance)

| Masquage | RMSE | MAE | CRPS |
|----------|------|-----|------|
| 10% aleatoire | 0.0198 | 0.0110 | 0.0076 |
| 30% aleatoire | 0.0205 | 0.0113 | 0.0078 |
| 50% aleatoire | 0.0220 | 0.0118 | 0.0083 |
| 10% bloc | 0.0276 | 0.0140 | 0.0101 |
| 30% bloc | 0.0280 | 0.0142 | 0.0103 |
| 50% bloc | 0.0284 | 0.0142 | 0.0103 |

En masquage aleatoire, CSDI bat le zero (0.0198 vs 0.0268). En masquage bloc, le zero reprend l'avantage (0.0265 vs 0.0276).

### Couverture IC 80% (calibration)

Couverture empirique des intervalles a 80% (cible : 80%, N_SAMPLES=10) :
- Exchange Rate : 50–57% selon la configuration
- Yahoo Finance : 54–61% selon la configuration

Le modele est sous-dispersif : les intervalles predits sont trop etroits. Cela est en partie du au nombre limite d'echantillons (10).

### Interpretabilite SHAP

JPMorgan_t-14 est la feature la plus influente (XGBoost), suivie d'Exxon et Apple. Profil des lags non monotone avec pic a t-7. Les classements divergent entre XGBoost et LightGBM. Ces modeles ne battant pas le zero, SHAP explique principalement du bruit.

### Test de Diebold-Mariano

Test DM avec correction HAC (noyau de Bartlett / Newey-West) et correction petits echantillons (Harvey et al. 1997). Zero vs XGBoost : non significatif. Zero vs LightGBM : zero significativement meilleur.

## Installation et execution

```bash
pip install -r requirements.txt

# Lancer l'application Streamlit
python -m streamlit run app.py

# Reproduire les resultats (dans l'ordre)
python 01_nettoyage_eda.py
python 02_log_rendements_split.py
python 03_exchange_rate_prep.py
python 04_prepare_csdi_data.py
python 05_run_csdi_exchange_rate.py
python 06_baselines_forecasting.py
python 07_baselines_imputation.py
python 07_shap_interpretabilite.py
python 08_run_csdi_yahoo.py
python 09_diebold_mariano.py
```

## References

- Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Cont, "Empirical properties of asset returns: stylized facts and statistical issues", 2001
- Newey & West, "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix", Econometrica 1987
- Harvey, Leybourne & Newbold, "Testing the equality of prediction mean squared errors", IJOF 1997
