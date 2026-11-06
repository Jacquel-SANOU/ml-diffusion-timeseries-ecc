"""
Streamlit App — Modeles de diffusion pour la prevision et l'imputation
de series temporelles financieres
Sujet 3 — ML, ECC 2026-2027
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob as glob_mod
import pickle

st.set_page_config(
    page_title="CSDI — Series Temporelles Financieres",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "Accueil",
        "Donnees Yahoo Finance",
        "Donnees Exchange Rate",
        "Baselines (Prevision)",
        "CSDI — Imputation",
        "Interpretabilite SHAP",
        "Comparaison globale",
    ],
)

FIGURES_DIR = "results/figures"


def show_figure(filename, caption=""):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure non trouvee : {filename}")


# ── Accueil ─────────────────────────────────────────────
if page == "Accueil":
    st.title("Modeles de diffusion pour la prevision probabiliste et l'imputation de series temporelles")
    st.subheader("Application aux donnees financieres")
    st.markdown("""
    **Sujet 3** — Projet ML, Ecole Centrale Casablanca (2026-2027)

    ---

    ### Objectif
    Explorer les modeles de diffusion conditionnels (CSDI) pour :
    - **L'imputation** de valeurs manquantes dans des series temporelles multivariees
    - **La prevision probabiliste** avec intervalles de confiance

    ### Jeux de donnees
    | Dataset | Series | Periode | Frequence |
    |---------|--------|---------|-----------|
    | Yahoo Finance | 9 actifs (S&P 500, Apple, etc.) | 2010-2026 | Journaliere |
    | Exchange Rate | 8 taux de change | 1990-2010 | Journaliere |

    ### Methodologie
    1. **Preprocessing** : log-rendements, normalisation sur train uniquement
    2. **Baselines** : Naive, ARIMA, XGBoost, LightGBM, LSTM
    3. **CSDI** : Modele de diffusion conditionnel avec masquage
    4. **Evaluation** : CRPS (probabiliste), MAE, RMSE
    5. **Interpretabilite** : SHAP (TreeExplainer)

    ### Architecture CSDI
    - Encodeur Transformer (4 couches, 8 tetes)
    - 50 pas de diffusion (schedule quadratique)
    - Embedding temporel + embedding par feature
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Modeles compares", "6")
    col2.metric("Series analysees", "17")
    col3.metric("Configurations masquage", "6")

# ── Yahoo Finance ───────────────────────────────────────
elif page == "Donnees Yahoo Finance":
    st.title("Exploration — Yahoo Finance")

    tab1, tab2, tab3 = st.tabs(["Prix & Rendements", "Correlations", "Split temporel"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("02_evolution_prix_9series.png", "Evolution des prix (2010-2026)")
        with col2:
            show_figure("03_prix_normalises_base100.png", "Prix normalises base 100")

        col3, col4 = st.columns(2)
        with col3:
            show_figure("05_log_rendements_9series.png", "Log-rendements journaliers")
        with col4:
            show_figure("06_distribution_log_rendements.png", "Distribution des log-rendements")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("04_correlation_prix.png", "Correlation des prix")
        with col2:
            show_figure("07_correlation_log_rendements.png", "Correlation des log-rendements")

    with tab3:
        show_figure("08_split_temporel.png", "Split temporel : Train (2010-2020), Val (2021-2022), Test (2023-2026)")
        show_figure("09_log_rendements_normalises.png", "Log-rendements normalises (standardisation sur train)")

    show_figure("01_valeurs_manquantes_heatmap.png", "Valeurs manquantes")

# ── Exchange Rate ───────────────────────────────────────
elif page == "Donnees Exchange Rate":
    st.title("Exploration — Exchange Rate (8 devises)")

    col1, col2 = st.columns(2)
    with col1:
        show_figure("10_exchange_rate_series.png", "Evolution des 8 taux de change (1990-2010)")
    with col2:
        show_figure("11_exchange_rate_correlation.png", "Matrice de correlation")

    col3, col4 = st.columns(2)
    with col3:
        show_figure("12_exchange_rate_normalise.png", "Donnees normalisees")
    with col4:
        show_figure("13_exchange_rate_split.png", "Split 80/10/10")

# ── Baselines ───────────────────────────────────────────
elif page == "Baselines (Prevision)":
    st.title("Baselines — Prevision sur Yahoo Finance")

    st.markdown("""
    **Configuration** : Lookback L=20, Horizon H=5, sur log-rendements normalises.

    Cinq modeles de reference :
    - **Naive** : derniere valeur observee
    - **ARIMA(1,0,0)** : autoregression simple
    - **XGBoost** : gradient boosting (200 arbres)
    - **LightGBM** : gradient boosting (200 arbres)
    - **LSTM** : reseau recurrent (64 hidden, 2 couches, 50 epochs)
    """)

    results_path = "results/baselines_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        df_results = pd.DataFrame(results).T
        df_results.index.name = "Modele"
        df_results = df_results.round(4)
        st.dataframe(df_results, use_container_width=True)

        best_model = df_results["MAE"].idxmin()
        st.success(f"Meilleur modele (MAE) : **{best_model}** — MAE = {df_results.loc[best_model, 'MAE']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        show_figure("16_baselines_comparaison.png", "Comparaison des baselines")
    with col2:
        show_figure("17_baselines_exemple_prevision.png", "Exemple de prevision")

# ── CSDI Imputation ─────────────────────────────────────
elif page == "CSDI — Imputation":
    st.title("CSDI — Imputation par modele de diffusion")

    dataset_choice = st.selectbox("Dataset", ["Exchange Rate", "Yahoo Finance"])

    if dataset_choice == "Exchange Rate":
        st.subheader("Exchange Rate (8 devises)")
        show_figure("14_csdi_training_loss.png", "Courbe d'apprentissage")

        results_path = "save/csdi_exchange_rate_20261106_111634/results_imputation.json"
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            df_r = pd.DataFrame(results).T
            df_r.index.name = "Configuration"
            st.dataframe(df_r.round(6), use_container_width=True)

            st.markdown("""
            **Observations** :
            - Le masquage **bloc** est systematiquement plus difficile que le masquage **aleatoire**
            - La performance se degrade graduellement avec le taux de masquage
            - Meme a 50% de donnees manquantes, CSDI produit des imputations coherentes
            """)

        show_figure("15_csdi_imputation_exemple.png",
                     "Exemple d'imputation (30% masquage aleatoire)")

    else:
        st.subheader("Yahoo Finance (9 series)")
        show_figure("23_csdi_yahoo_training_loss.png", "Courbe d'apprentissage")

        yahoo_dirs = sorted(glob_mod.glob("save/csdi_yahoo_*/results_imputation.json"))
        if yahoo_dirs:
            with open(yahoo_dirs[-1]) as f:
                results = json.load(f)
            df_r = pd.DataFrame(results).T
            df_r.index.name = "Configuration"
            st.dataframe(df_r.round(6), use_container_width=True)

        show_figure("24_csdi_yahoo_imputation.png",
                     "Exemple d'imputation (30% masquage aleatoire)")
        show_figure("25_csdi_yahoo_masking_comparison.png",
                     "Comparaison des configurations de masquage")

# ── SHAP ────────────────────────────────────────────────
elif page == "Interpretabilite SHAP":
    st.title("Interpretabilite — SHAP")

    st.markdown("""
    Analyse SHAP (SHapley Additive exPlanations) sur les modeles XGBoost et LightGBM
    entraines pour predire le S&P 500 (h+1).

    **Methode** : TreeExplainer — exacte pour les modeles a base d'arbres.
    """)

    tab1, tab2, tab3 = st.tabs(["Importance globale", "Par serie / par lag", "XGBoost vs LightGBM"])

    with tab1:
        show_figure("18_shap_summary_xgboost.png",
                     "SHAP Summary — Top 20 features (XGBoost)")
        show_figure("21_shap_waterfall.png",
                     "SHAP Waterfall — Decomposition d'une prediction")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("19_shap_importance_par_serie.png",
                         "Importance SHAP par serie financiere")
        with col2:
            show_figure("20_shap_importance_par_lag.png",
                         "Importance SHAP par lag temporel")

        st.markdown("""
        **Resultats cles** :
        - **Exxon** est la serie la plus influente pour predire le S&P 500
        - Les lags recents (t-1, t-2) sont les plus importants
        - L'impact decroit exponentiellement avec le lag
        """)

    with tab3:
        show_figure("22_shap_xgb_vs_lgb.png",
                     "Comparaison XGBoost vs LightGBM — Importance SHAP")

# ── Comparaison globale ─────────────────────────────────
elif page == "Comparaison globale":
    st.title("Comparaison globale des modeles")

    st.markdown("### Baselines — Prevision (Yahoo Finance)")
    results_path = "results/baselines_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            bl = json.load(f)
        df_bl = pd.DataFrame(bl).T
        df_bl.index.name = "Modele"
        st.dataframe(df_bl.round(4), use_container_width=True)

    st.markdown("### CSDI — Imputation (Exchange Rate)")
    er_path = "save/csdi_exchange_rate_20261106_111634/results_imputation.json"
    if os.path.exists(er_path):
        with open(er_path) as f:
            er = json.load(f)
        df_er = pd.DataFrame(er).T
        df_er.index.name = "Masquage"
        st.dataframe(df_er.round(6), use_container_width=True)

    st.markdown("### CSDI — Imputation (Yahoo Finance)")
    yahoo_dirs = sorted(glob_mod.glob("save/csdi_yahoo_*/results_imputation.json"))
    if yahoo_dirs:
        with open(yahoo_dirs[-1]) as f:
            yh = json.load(f)
        df_yh = pd.DataFrame(yh).T
        df_yh.index.name = "Masquage"
        st.dataframe(df_yh.round(6), use_container_width=True)

    st.markdown("### Test de Diebold-Mariano")
    dm_path = "results/diebold_mariano.json"
    if os.path.exists(dm_path):
        with open(dm_path) as f:
            dm = json.load(f)
        dm_models = dm["models"]
        dm_pvals = np.array(dm["p_values"])
        df_dm = pd.DataFrame(dm_pvals, index=dm_models, columns=dm_models)
        st.dataframe(df_dm.round(4), use_container_width=True)
        st.caption("p-values du test DM. p < 0.05 = difference significative.")
        show_figure("26_diebold_mariano.png", "Heatmap du test de Diebold-Mariano")
    else:
        st.info("Test DM non encore execute (python 09_diebold_mariano.py)")

    st.markdown("""
    ### Synthese
    | Aspect | Observation |
    |--------|-------------|
    | **Meilleur baseline** | LightGBM (MAE=0.663, RMSE=0.940) |
    | **CSDI imputation** | Excellente performance meme a 50% masquage |
    | **Bloc vs Aleatoire** | Masquage bloc ~3-4x plus difficile |
    | **Interpretabilite** | Exxon et les lags recents dominent |
    | **Approche probabiliste** | CSDI fournit des intervalles de confiance (CRPS) |

    ### Avantages du modele CSDI
    1. **Flexibilite** : meme architecture pour imputation et prevision
    2. **Incertitude** : quantification native via echantillonnage
    3. **Multivarite** : capture les dependances entre series
    4. **Robustesse** : performance stable meme avec beaucoup de donnees manquantes
    """)
