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
import sys
import glob as glob_mod
import pickle
import warnings
warnings.filterwarnings("ignore")

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
        "🔮 Prevision interactive",
        "🧩 Imputation interactive",
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
    2. **Baselines** : Zero, Naive, ARIMA, XGBoost, LightGBM, LSTM
    3. **CSDI** : Modele de diffusion conditionnel avec masquage
    4. **Evaluation** : CRPS (probabiliste), MAE, RMSE + baselines d'imputation
    5. **Interpretabilite** : SHAP (TreeExplainer)

    ### Architecture CSDI
    - Attention bidirectionnelle (temporelle + inter-series, 4 couches, 8 tetes)
    - 50 pas de diffusion (schedule quadratique)
    - Embedding temporel + embedding par feature
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Modeles compares", "7")
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

    Six modeles de reference :
    - **Zero** : predit toujours 0 (la moyenne des log-rendements normalises)
    - **Naive** : derniere valeur observee
    - **ARIMA(1,0,0)** : autoregression simple (refit par fenetre)
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

        er_dirs = sorted(glob_mod.glob("save/csdi_exchange_rate_*/results_imputation.json"))
        if er_dirs:
            with open(er_dirs[-1]) as f:
                results = json.load(f)
            df_r = pd.DataFrame(results).T
            df_r.index.name = "Configuration"
            st.dataframe(df_r.round(6), use_container_width=True)

            imp_bl_path = "results/baselines_imputation.json"
            if os.path.exists(imp_bl_path):
                with open(imp_bl_path) as f:
                    imp_bl = json.load(f)
                if "exchange_rate" in imp_bl:
                    st.markdown("**Baselines d'imputation (comparaison) :**")
                    rows = []
                    for cfg, methods in imp_bl["exchange_rate"].items():
                        for method, metrics in methods.items():
                            rows.append({"Config": cfg, "Methode": method, **metrics})
                    df_bl_imp = pd.DataFrame(rows)
                    st.dataframe(df_bl_imp.round(6), use_container_width=True)

            st.markdown("""
            **Observations** :
            - Le masquage **bloc** est plus difficile que le masquage **aleatoire**
            - CSDI bat toutes les baselines en masquage aleatoire sur Exchange Rate
            - Sur Yahoo, la gaussienne conditionnelle bat CSDI en aleatoire (correlations inter-series)
            - En masquage bloc, le zero reprend l'avantage sur Yahoo, l'interpolation sur Exchange Rate
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

        imp_bl_path = "results/baselines_imputation.json"
        if os.path.exists(imp_bl_path):
            with open(imp_bl_path) as f:
                imp_bl = json.load(f)
            if "yahoo" in imp_bl:
                st.markdown("**Baselines d'imputation (comparaison) :**")
                rows = []
                for cfg, methods in imp_bl["yahoo"].items():
                    for method, metrics in methods.items():
                        rows.append({"Config": cfg, "Methode": method, **metrics})
                df_bl_imp = pd.DataFrame(rows)
                st.dataframe(df_bl_imp.round(6), use_container_width=True)

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
        - **Exxon** est en tete des deux modeles, mais les ecarts d'importance entre series sont faibles
        - L'importance est repartie sur l'ensemble des lags, sans decroissance nette
        - Ces modeles ne battant pas le predicteur zero, SHAP explique principalement du bruit
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
    er_result_dirs = sorted(glob_mod.glob("save/csdi_exchange_rate_*/results_imputation.json"))
    er_path = er_result_dirs[-1] if er_result_dirs else ""
    if er_path and os.path.exists(er_path):
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
    | **Baselines prevision** | Le predicteur zero (moyenne) bat tous les modeles appris en MAE |
    | **CSDI imputation (Yahoo)** | Bat zero et interpolation en aleatoire, mais pas la gaussienne conditionnelle ; en bloc le zero gagne |
    | **CSDI imputation (Exch. Rate)** | Bat toutes les baselines en aleatoire ; en bloc l'interpolation reprend l'avantage |
    | **Bloc vs Aleatoire** | Masquage bloc plus difficile (pas de voisins proches) |
    | **Interpretabilite** | Importance SHAP repartie, ecarts faibles entre series — explique surtout du bruit |
    | **Approche probabiliste** | CSDI quantifie l'incertitude via echantillonnage (couverture a mesurer) |

    ### Points forts du modele CSDI
    1. **Imputation multivariee** : capture les dependances entre series pour reconstruire les manquants
    2. **Incertitude** : quantification native via echantillonnage (N echantillons → intervalles)
    3. **Flexibilite** : meme architecture pour differents schemas de masquage

    ### Limites
    - Prevision par diffusion (CRPS, couverture) non realisee dans le temps imparti
    - Rendements financiers quasi bruit blanc : peu de signal exploitable en prevision
    - Gaussienne conditionnelle non testee comme baseline d'imputation
    """)


# ═══════════════════════════════════════════════════════════
# PREVISION INTERACTIVE
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Prevision interactive":
    st.title("🔮 Prevision interactive — Baselines")

    LOOKBACK = 20
    HORIZON = 5

    @st.cache_data
    def load_yahoo_data():
        train = pd.read_csv("data/processed/train.csv", index_col="Date", parse_dates=True)
        val = pd.read_csv("data/processed/val.csv", index_col="Date", parse_dates=True)
        test = pd.read_csv("data/processed/test.csv", index_col="Date", parse_dates=True)
        norm_params = pd.read_csv("data/processed/norm_params.csv", index_col=0)
        prix = pd.read_csv("data/processed/yahoo_prix_clean.csv", index_col="Date", parse_dates=True)
        return train, val, test, norm_params, prix

    try:
        train, val, test, norm_params, prix = load_yahoo_data()
    except FileNotFoundError:
        st.error("Donnees non trouvees. Lancez d'abord les scripts de preprocessing.")
        st.stop()

    series_names = list(train.columns)

    def create_windows(data, lookback, horizon):
        X, Y = [], []
        arr = data.values
        for i in range(len(arr) - lookback - horizon + 1):
            X.append(arr[i:i + lookback])
            Y.append(arr[i + lookback:i + lookback + horizon])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_windows(train, LOOKBACK, HORIZON)
    X_test, Y_test = create_windows(test, LOOKBACK, HORIZON)
    X_val, Y_val = create_windows(val, LOOKBACK, HORIZON)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    mode_prev = st.radio("Mode", ["Backtest (donnees passees)", "Prevision future (derniers jours connus)"],
                         horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Modele", ["XGBoost", "LightGBM", "LSTM", "Naive"])
    with col2:
        serie_choice = st.selectbox("Serie a predire", series_names,
                                    index=series_names.index("S&P 500") if "S&P 500" in series_names else 0)

    s_idx = series_names.index(serie_choice)

    if mode_prev == "Backtest (donnees passees)":
        window_idx = st.slider("Position dans le test", 0, len(X_test) - 1, len(X_test) // 2)
        st.caption(f"Fenetre : jours test #{window_idx} a #{window_idx + LOOKBACK - 1} → predit #{window_idx + LOOKBACK} a #{window_idx + LOOKBACK + HORIZON - 1}")

    if st.button("Lancer la prevision", type="primary"):
        import xgboost as xgb_lib
        import lightgbm as lgb_lib
        import torch
        import torch.nn as nn
        import matplotlib.pyplot as plt

        is_future = (mode_prev == "Prevision future (derniers jours connus)")

        if is_future:
            all_log_returns = pd.concat([train, val, test])
            last_window = all_log_returns.values[-LOOKBACK:]
            x_input_flat = last_window.reshape(1, -1)
            x_input_3d = last_window.reshape(1, LOOKBACK, -1)
            last_price = prix[serie_choice].iloc[-1]
            last_date = prix.index[-1]
        else:
            x_input_flat = X_test_flat[window_idx:window_idx + 1]
            x_input_3d = X_test[window_idx:window_idx + 1]

        with st.spinner("Entrainement et prevision en cours..."):
            importance_info = None

            has_quantiles = False

            if model_choice == "Naive":
                if is_future:
                    y_pred_serie = np.repeat(last_window[-1:, s_idx], HORIZON)
                else:
                    y_pred_serie = np.repeat(X_test[window_idx, -1, s_idx], HORIZON)

            elif model_choice in ("XGBoost", "LightGBM"):
                if model_choice == "XGBoost":
                    make_model = lambda obj="reg:squarederror": xgb_lib.XGBRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        objective=obj, verbosity=0, random_state=42)
                else:
                    make_model = lambda obj="regression": lgb_lib.LGBMRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        objective=obj, verbosity=-1, random_state=42)

                y_pred_serie = np.zeros(HORIZON)
                y_pred_q10 = np.zeros(HORIZON)
                y_pred_q90 = np.zeros(HORIZON)

                for h in range(HORIZON):
                    mdl = make_model()
                    if model_choice == "XGBoost":
                        mdl.fit(X_train_flat, Y_train[:, h, s_idx],
                                eval_set=[(X_val_flat, Y_val[:, h, s_idx])], verbose=False)
                    else:
                        mdl.fit(X_train_flat, Y_train[:, h, s_idx],
                                eval_set=[(X_val_flat, Y_val[:, h, s_idx])])
                    y_pred_serie[h] = mdl.predict(x_input_flat)[0]

                    if h == 0:
                        import shap
                        explainer = shap.TreeExplainer(mdl)
                        sv = explainer.shap_values(x_input_flat)
                        importance_info = {"shap_values": sv[0], "model_name": model_choice}

                    if model_choice == "XGBoost":
                        mdl_q10 = make_model("reg:quantileerror")
                        mdl_q10.set_params(quantile_alpha=0.1)
                        mdl_q10.fit(X_train_flat, Y_train[:, h, s_idx],
                                    eval_set=[(X_val_flat, Y_val[:, h, s_idx])], verbose=False)
                        y_pred_q10[h] = mdl_q10.predict(x_input_flat)[0]
                        mdl_q90 = make_model("reg:quantileerror")
                        mdl_q90.set_params(quantile_alpha=0.9)
                        mdl_q90.fit(X_train_flat, Y_train[:, h, s_idx],
                                    eval_set=[(X_val_flat, Y_val[:, h, s_idx])], verbose=False)
                        y_pred_q90[h] = mdl_q90.predict(x_input_flat)[0]
                    else:
                        mdl_q10 = make_model("quantile")
                        mdl_q10.set_params(alpha=0.1)
                        mdl_q10.fit(X_train_flat, Y_train[:, h, s_idx],
                                    eval_set=[(X_val_flat, Y_val[:, h, s_idx])])
                        y_pred_q10[h] = mdl_q10.predict(x_input_flat)[0]
                        mdl_q90 = make_model("quantile")
                        mdl_q90.set_params(alpha=0.9)
                        mdl_q90.fit(X_train_flat, Y_train[:, h, s_idx],
                                    eval_set=[(X_val_flat, Y_val[:, h, s_idx])])
                        y_pred_q90[h] = mdl_q90.predict(x_input_flat)[0]

                has_quantiles = True

            elif model_choice == "LSTM":
                DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                n_features = len(series_names)

                class LSTMForecaster(nn.Module):
                    def __init__(self, n_feat, hidden_size=64, n_layers=2, horizon=5):
                        super().__init__()
                        self.lstm = nn.LSTM(n_feat, hidden_size, n_layers, batch_first=True, dropout=0.1)
                        self.fc = nn.Linear(hidden_size, n_feat * horizon)
                        self.n_features = n_feat
                        self.horizon = horizon
                    def forward(self, x):
                        out, _ = self.lstm(x)
                        out = self.fc(out[:, -1, :])
                        return out.reshape(-1, self.horizon, self.n_features)

                lstm_model = LSTMForecaster(n_features, 64, 2, HORIZON).to(DEVICE)
                if os.path.exists("save/lstm_best.pth"):
                    lstm_model.load_state_dict(torch.load("save/lstm_best.pth",
                                                          weights_only=True, map_location=DEVICE))
                    lstm_model.eval()
                    x_tensor = torch.FloatTensor(x_input_3d).to(DEVICE)
                    with torch.no_grad():
                        pred = lstm_model(x_tensor).cpu().numpy()
                    y_pred_serie = pred[0, :, s_idx]
                else:
                    st.error("Modele LSTM non trouve (save/lstm_best.pth).")
                    st.stop()

        # ── Recuperer les parametres de normalisation ──
        mu = norm_params.loc[serie_choice, "mean"]
        sigma = norm_params.loc[serie_choice, "std"]

        # ═══════════════════════════════════════════════════
        # MODE PREVISION FUTURE — affichage en prix reels
        # ═══════════════════════════════════════════════════
        if is_future:
            st.subheader(f"Prevision future {model_choice} — {serie_choice}")

            pred_log_returns = y_pred_serie * sigma + mu

            pred_prices = np.zeros(HORIZON)
            current_price = last_price
            for h in range(HORIZON):
                current_price = current_price * np.exp(pred_log_returns[h])
                pred_prices[h] = current_price

            if has_quantiles:
                pred_lr_q10 = y_pred_q10 * sigma + mu
                pred_lr_q90 = y_pred_q90 * sigma + mu
                pred_prices_q10 = np.zeros(HORIZON)
                pred_prices_q90 = np.zeros(HORIZON)
                cp10, cp90 = last_price, last_price
                for h in range(HORIZON):
                    cp10 = cp10 * np.exp(pred_lr_q10[h])
                    cp90 = cp90 * np.exp(pred_lr_q90[h])
                    pred_prices_q10[h] = cp10
                    pred_prices_q90[h] = cp90
                pp_lo = np.minimum(pred_prices_q10, pred_prices_q90)
                pp_hi = np.maximum(pred_prices_q10, pred_prices_q90)

            n_hist_show = 30
            hist_prices = prix[serie_choice].iloc[-n_hist_show:].values
            hist_dates = prix.index[-n_hist_show:]

            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(hist_dates, hist_prices, "k-", linewidth=2, marker="o", markersize=3,
                    label=f"Historique ({serie_choice})")
            ax.plot(future_dates, pred_prices, "s-", linewidth=2.5, markersize=8,
                    color="#e74c3c", label=f"Prevision {model_choice} (mediane)")
            if has_quantiles:
                ax.fill_between(future_dates, pp_lo, pp_hi,
                                alpha=0.2, color="#e74c3c", label="Intervalle de confiance 80%")
            ax.axvline(x=last_date, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.annotate("Aujourd'hui", xy=(last_date, last_price),
                         xytext=(last_date, last_price * 1.02),
                         fontsize=10, ha="center", color="gray")

            for i, (d, p) in enumerate(zip(future_dates, pred_prices)):
                ax.annotate(f"{p:.2f}", xy=(d, p), xytext=(0, 12),
                            textcoords="offset points", fontsize=9, ha="center",
                            fontweight="bold", color="#e74c3c")

            ax.set_xlabel("Date")
            ax.set_ylabel(f"Prix ({serie_choice})")
            ax.set_title(f"Prevision probabiliste {model_choice} — {serie_choice} — 5 prochains jours ouvrables",
                         fontsize=14, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Previsions detaillees")
            prev_data = {
                "Date": future_dates.strftime("%Y-%m-%d"),
                "Prix predit": [f"{p:.2f}" for p in pred_prices],
                "Variation (%)": [f"{(np.exp(pred_log_returns[h]) - 1) * 100:+.2f}%" for h in range(HORIZON)],
            }
            if has_quantiles:
                prev_data["Borne basse (10%)"] = [f"{p:.2f}" for p in pp_lo]
                prev_data["Borne haute (90%)"] = [f"{p:.2f}" for p in pp_hi]
                prev_data["Largeur IC"] = [f"{hi - lo:.2f}" for lo, hi in zip(pp_lo, pp_hi)]
            df_prev = pd.DataFrame(prev_data)
            df_prev.index = [f"J+{i+1}" for i in range(HORIZON)]
            st.dataframe(df_prev, use_container_width=True)

            variation_totale = (pred_prices[-1] / last_price - 1) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Dernier prix connu", f"{last_price:.2f}", delta=None)
            col2.metric("Prix predit J+5", f"{pred_prices[-1]:.2f}",
                         delta=f"{variation_totale:+.2f}%")
            col3.metric("Tendance", "Hausse" if variation_totale > 0 else "Baisse")

            if has_quantiles:
                st.success(f"""
                **Prevision probabiliste** : le modele est confiant a 80% que le prix de
                {serie_choice} sera entre **{pp_lo[-1]:.2f}** et **{pp_hi[-1]:.2f}** dans 5 jours.
                Largeur de l'intervalle : {pp_hi[-1] - pp_lo[-1]:.2f}
                (plus c'est etroit, plus le modele est confiant).
                """)
                st.info(f"""
                **Comment ca marche** : on entraine 3 modeles {model_choice} :
                - Un pour le **quantile 10%** (borne basse)
                - Un pour la **mediane 50%** (prediction centrale)
                - Un pour le **quantile 90%** (borne haute)

                C'est la **regression quantile** — au lieu de predire la moyenne,
                chaque modele predit un quantile different de la distribution.
                L'intervalle [10%, 90%] = intervalle de confiance a 80%.
                """)
            else:
                st.info(f"""
                **Note** : {model_choice} est un modele deterministe — il donne une seule
                prediction sans intervalle de confiance. Pour des intervalles, utilisez
                XGBoost ou LightGBM (regression quantile) ou CSDI (echantillonnage par diffusion).
                """)

        # ═══════════════════════════════════════════════════
        # MODE BACKTEST — donnees passees
        # ═══════════════════════════════════════════════════
        else:
            st.subheader(f"Backtest {model_choice} — {serie_choice}")

            historique_norm = X_test[window_idx, :, s_idx]
            vrai_norm = Y_test[window_idx, :, s_idx]

            test_start = len(train) + len(val)
            prix_start_idx = test_start + window_idx
            prix_end_idx = prix_start_idx + LOOKBACK + HORIZON

            if prix_end_idx <= len(prix):
                hist_prices = prix[serie_choice].iloc[prix_start_idx:prix_start_idx + LOOKBACK].values
                true_prices = prix[serie_choice].iloc[prix_start_idx + LOOKBACK:prix_end_idx].values
                hist_dates = prix.index[prix_start_idx:prix_start_idx + LOOKBACK]
                pred_dates = prix.index[prix_start_idx + LOOKBACK:prix_end_idx]

                pred_log_returns = y_pred_serie * sigma + mu
                pred_prices = np.zeros(HORIZON)
                current_p = hist_prices[-1]
                for h in range(HORIZON):
                    current_p = current_p * np.exp(pred_log_returns[h])
                    pred_prices[h] = current_p

                if has_quantiles:
                    bt_lr_q10 = y_pred_q10 * sigma + mu
                    bt_lr_q90 = y_pred_q90 * sigma + mu
                    bt_pp_q10, bt_pp_q90 = np.zeros(HORIZON), np.zeros(HORIZON)
                    cp10, cp90 = hist_prices[-1], hist_prices[-1]
                    for h in range(HORIZON):
                        cp10 = cp10 * np.exp(bt_lr_q10[h])
                        cp90 = cp90 * np.exp(bt_lr_q90[h])
                        bt_pp_q10[h], bt_pp_q90[h] = cp10, cp90
                    bt_lo = np.minimum(bt_pp_q10, bt_pp_q90)
                    bt_hi = np.maximum(bt_pp_q10, bt_pp_q90)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

                ax1.plot(range(LOOKBACK), historique_norm, "k-", linewidth=2, label="Historique")
                ax1.plot(range(LOOKBACK, LOOKBACK + HORIZON), vrai_norm, "ko--", linewidth=2,
                         markersize=6, label="Valeurs reelles")
                ax1.plot(range(LOOKBACK, LOOKBACK + HORIZON), y_pred_serie, "s-", linewidth=2,
                         markersize=6, color="#e74c3c", label=f"Prevision {model_choice}")
                if has_quantiles:
                    ax1.fill_between(range(LOOKBACK, LOOKBACK + HORIZON),
                                     y_pred_q10, y_pred_q90, alpha=0.2, color="#e74c3c",
                                     label="IC 80%")
                ax1.axvline(x=LOOKBACK - 0.5, color="gray", linestyle="--", alpha=0.5)
                ax1.set_xlabel("Pas de temps")
                ax1.set_ylabel("Log-rendement normalise")
                ax1.set_title("Log-rendements normalises", fontweight="bold")
                ax1.legend(fontsize=9)
                ax1.grid(alpha=0.3)

                ax2.plot(hist_dates, hist_prices, "k-", linewidth=2, marker="o", markersize=3,
                         label="Prix historique")
                ax2.plot(pred_dates, true_prices, "ko--", linewidth=2, markersize=6,
                         label="Prix reel")
                ax2.plot(pred_dates, pred_prices, "s-", linewidth=2, markersize=6,
                         color="#e74c3c", label=f"Prix predit ({model_choice})")
                if has_quantiles:
                    ax2.fill_between(pred_dates, bt_lo, bt_hi, alpha=0.2, color="#e74c3c",
                                     label="IC 80%")
                ax2.axvline(x=hist_dates[-1], color="gray", linestyle="--", alpha=0.5)
                ax2.set_xlabel("Date")
                ax2.set_ylabel(f"Prix ({serie_choice})")
                ax2.set_title("Prix reels ($)", fontweight="bold")
                ax2.legend(fontsize=9)
                ax2.grid(alpha=0.3)
                plt.xticks(rotation=30, ha="right")

                fig.suptitle(f"{model_choice} — {serie_choice} (fenetre test #{window_idx})",
                             fontsize=14, fontweight="bold", y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                col1, col2, col3, col4 = st.columns(4)
                mae_norm = np.mean(np.abs(vrai_norm - y_pred_serie))
                rmse_norm = np.sqrt(np.mean((vrai_norm - y_pred_serie) ** 2))
                mae_prix = np.mean(np.abs(true_prices - pred_prices))
                rmse_prix = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
                col1.metric("MAE (normalise)", f"{mae_norm:.6f}")
                col2.metric("RMSE (normalise)", f"{rmse_norm:.6f}")
                col3.metric("MAE (prix)", f"{mae_prix:.2f}")
                col4.metric("RMSE (prix)", f"{rmse_prix:.2f}")

                st.markdown("### Tableau comparatif")
                comp_data = {
                    "Date": pred_dates.strftime("%Y-%m-%d"),
                    "Prix reel": [f"{p:.2f}" for p in true_prices],
                    "Prix predit": [f"{p:.2f}" for p in pred_prices],
                    "Erreur": [f"{abs(t - p):.2f}" for t, p in zip(true_prices, pred_prices)],
                }
                if has_quantiles:
                    comp_data["IC bas"] = [f"{p:.2f}" for p in bt_lo]
                    comp_data["IC haut"] = [f"{p:.2f}" for p in bt_hi]
                    in_ic = sum(1 for t, lo, hi in zip(true_prices, bt_lo, bt_hi) if lo <= t <= hi)
                    st.success(f"**Couverture** : {in_ic}/{HORIZON} valeurs reelles tombent dans l'IC 80%")
                df_comp = pd.DataFrame(comp_data)
                df_comp.index = [f"J+{i+1}" for i in range(HORIZON)]
                st.dataframe(df_comp, use_container_width=True)

            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                t_hist = np.arange(LOOKBACK)
                t_pred = np.arange(LOOKBACK, LOOKBACK + HORIZON)
                ax.plot(t_hist, historique_norm, "k-", linewidth=2, label="Historique")
                ax.plot(t_pred, vrai_norm, "ko--", linewidth=2, markersize=6, label="Valeurs reelles")
                ax.plot(t_pred, y_pred_serie, "s-", linewidth=2, markersize=6,
                        color="#e74c3c", label=f"Prevision {model_choice}")
                ax.axvline(x=LOOKBACK - 0.5, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Pas de temps")
                ax.set_ylabel("Log-rendement normalise")
                ax.set_title(f"{model_choice} — {serie_choice} (fenetre test #{window_idx})", fontweight="bold")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()

                mae = np.mean(np.abs(vrai_norm - y_pred_serie))
                rmse = np.sqrt(np.mean((vrai_norm - y_pred_serie) ** 2))
                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{mae:.6f}")
                col2.metric("RMSE", f"{rmse:.6f}")

        # ── SHAP — Explication (pour les deux modes) ──
        if importance_info is not None:
            st.markdown("---")
            st.subheader("🔍 Explication SHAP — Pourquoi cette prevision ?")
            st.markdown(f"""
            SHAP decompose la prediction en **contributions de chaque feature**.
            Chaque feature = `serie_t-lag` (ex: `Exxon_t-1` = rendement d'Exxon hier).
            """)

            feature_names = [f"{s}_t-{LOOKBACK - lag}"
                             for lag in range(LOOKBACK) for s in series_names]
            sv = importance_info["shap_values"]

            top_k = 15
            abs_sv = np.abs(sv)
            top_indices = np.argsort(abs_sv)[::-1][:top_k]

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            top_names = [feature_names[i] for i in top_indices]
            top_vals = [sv[i] for i in top_indices]
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]

            ax2.barh(range(top_k), top_vals[::-1], color=colors[::-1])
            ax2.set_yticks(range(top_k))
            ax2.set_yticklabels(top_names[::-1])
            ax2.set_xlabel("Valeur SHAP (impact sur la prediction)")
            ax2.set_title(f"Top {top_k} features — {importance_info['model_name']}",
                         fontweight="bold")
            ax2.axvline(x=0, color="black", linewidth=0.5)
            ax2.grid(alpha=0.3, axis="x")
            st.pyplot(fig2)
            plt.close()

            st.markdown("""
            **Lecture** :
            - 🔴 **Rouge (positif)** : cette feature pousse la prediction vers le **haut** (hausse)
            - 🔵 **Bleu (negatif)** : cette feature pousse la prediction vers le **bas** (baisse)
            - Plus la barre est longue, plus l'influence est forte
            """)

            st.subheader("📊 Importance par serie financiere")
            serie_importance = {}
            for i, fname in enumerate(feature_names):
                s_name = fname.rsplit("_t-", 1)[0]
                serie_importance[s_name] = serie_importance.get(s_name, 0) + abs(sv[i])

            sorted_series = sorted(serie_importance.items(), key=lambda x: x[1], reverse=True)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            names_s = [x[0] for x in sorted_series]
            vals_s = [x[1] for x in sorted_series]
            ax3.bar(names_s, vals_s,
                    color=["#e74c3c" if n == serie_choice else "#3498db" for n in names_s])
            ax3.set_ylabel("Importance SHAP totale (|SHAP|)")
            ax3.set_title("Quelle serie influence le plus la prediction ?", fontweight="bold")
            ax3.grid(alpha=0.3, axis="y")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig3)
            plt.close()

            st.markdown(f"""
            **Interpretation** : Pour predire **{serie_choice}**, le modele {importance_info['model_name']}
            s'appuie principalement sur **{names_s[0]}** (importance = {vals_s[0]:.4f}),
            suivi de **{names_s[1]}** ({vals_s[1]:.4f}) et **{names_s[2]}** ({vals_s[2]:.4f}).

            Cela confirme les dependances inter-marches captees par le modele.
            """)

        elif model_choice == "Naive":
            st.info("Le modele Naive n'a pas de parametres — la prevision est simplement "
                     "la derniere valeur observee repetee H fois.")
        elif model_choice == "LSTM":
            st.info("SHAP n'est pas disponible pour LSTM (reseau de neurones). "
                     "Pour l'interpretabilite, utilisez XGBoost ou LightGBM.")


# ═══════════════════════════════════════════════════════════
# IMPUTATION INTERACTIVE
# ═══════════════════════════════════════════════════════════
elif page == "🧩 Imputation interactive":
    st.title("🧩 Imputation interactive — CSDI")

    st.markdown("""
    Le modele CSDI charge impute les valeurs manquantes en temps reel.
    Choisissez le dataset, le taux et le type de masquage, puis lancez l'imputation.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_imp = st.selectbox("Dataset", ["Exchange Rate", "Yahoo Finance"], key="imp_ds")
    with col2:
        mask_ratio = st.selectbox("Taux de masquage", [0.1, 0.3, 0.5], index=1,
                                  format_func=lambda x: f"{int(x*100)}%")
    with col3:
        mask_type = st.selectbox("Type de masquage", ["random", "block"],
                                 format_func=lambda x: "Aleatoire" if x == "random" else "Bloc")

    if dataset_imp == "Exchange Rate":
        er_model_dirs = sorted(glob_mod.glob("save/csdi_exchange_rate_*/model_best.pth"))
        model_path = er_model_dirs[-1] if er_model_dirs else ""
        data_folder = "data/csdi_exchange_rate"
        config_path = "config/exchange_rate.yaml"
        target_dim = 8
        series_names_csdi = ["Australie", "Angleterre", "Canada", "Suisse",
                             "Chine", "Japon", "Nouvelle-Zelande", "Singapour"]
    else:
        yahoo_dirs_list = sorted(glob_mod.glob("save/csdi_yahoo_*/model_best.pth"))
        model_path = yahoo_dirs_list[-1] if yahoo_dirs_list else ""
        data_folder = "data/csdi_yahoo"
        config_path = "config/yahoo.yaml"
        target_dim = 9
        series_names_csdi = ["Apple", "EUR/USD", "Or (GLD)", "JPMorgan", "Microsoft",
                             "Exxon", "CAC 40", "S&P 500", "VIX"]

    if not os.path.exists(model_path):
        st.error(f"Modele non trouve : {model_path}")
        st.stop()
    if not os.path.exists(data_folder):
        st.error(f"Donnees non trouvees : {data_folder}")
        st.stop()

    window_imp = st.slider("Fenetre dans le jeu de test", 0, 100, 0, key="imp_win")

    if st.button("Lancer l'imputation CSDI", type="primary"):
        import torch
        import torch.nn as nn
        import yaml
        from torch.utils.data import Dataset, DataLoader

        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CSDI"))
        from diff_models import diff_CSDI

        SEQ_LENGTH = 100
        N_SAMPLES = 10
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        st.info(f"Device : **{DEVICE}**" + (" (GPU)" if DEVICE == "cuda" else " (CPU — plus lent)"))

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Dataset class
        class CSDIDataset(Dataset):
            def __init__(self, data_folder, mode, mask_ratio, seq_length, mask_type, target_dim,
                         n_train=None, n_val=None, n_test=None):
                with open(os.path.join(data_folder, "data.pkl"), "rb") as f:
                    self.main_data, self.mask_data = pickle.load(f)
                with open(os.path.join(data_folder, "meanstd.pkl"), "rb") as f:
                    self.mean_data, self.std_data = pickle.load(f)
                self.main_data = (self.main_data - self.mean_data) / self.std_data
                self.seq_length = seq_length
                self.mask_ratio = mask_ratio
                self.mask_type = mask_type
                total = len(self.main_data)
                if n_train is not None:
                    nt, nv = n_train, n_val
                else:
                    nt = int(total * 0.8)
                    nv = int(total * 0.1)
                if mode == "test":
                    start, end = nt + nv, total - seq_length
                else:
                    start, end = 0, nt - seq_length
                self.use_index = np.arange(start, max(start + 1, end), 1)

            def __len__(self):
                return len(self.use_index)

            def __getitem__(self, idx):
                s = self.use_index[idx]
                observed = self.main_data[s:s + self.seq_length].copy()
                obs_mask = self.mask_data[s:s + self.seq_length].copy()
                gt_mask = obs_mask.copy()
                if self.mask_type == "random":
                    rand = np.random.rand(*gt_mask.shape)
                    gt_mask[rand < self.mask_ratio] = 0.0
                elif self.mask_type == "block":
                    block_len = max(1, int(self.seq_length * self.mask_ratio))
                    block_start = np.random.randint(0, self.seq_length - block_len + 1)
                    gt_mask[block_start:block_start + block_len, :] = 0.0
                return {
                    "observed_data": observed.astype(np.float32),
                    "observed_mask": obs_mask.astype(np.float32),
                    "gt_mask": gt_mask.astype(np.float32),
                    "timepoints": np.arange(self.seq_length).astype(np.float32),
                }

        # Model class
        class CSDI_Model(nn.Module):
            def __init__(self, config, device, target_dim):
                super().__init__()
                self.device = device
                self.target_dim = target_dim
                self.emb_time_dim = config["model"]["timeemb"]
                self.emb_feature_dim = config["model"]["featureemb"]
                self.is_unconditional = config["model"]["is_unconditional"]
                self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
                if not self.is_unconditional:
                    self.emb_total_dim += 1
                self.embed_layer = nn.Embedding(target_dim, self.emb_feature_dim)
                config_diff = config["diffusion"].copy()
                config_diff["side_dim"] = self.emb_total_dim
                input_dim = 1 if self.is_unconditional else 2
                self.diffmodel = diff_CSDI(config_diff, input_dim)
                self.num_steps = config_diff["num_steps"]
                if config_diff["schedule"] == "quad":
                    self.beta = np.linspace(config_diff["beta_start"]**0.5,
                                            config_diff["beta_end"]**0.5, self.num_steps)**2
                else:
                    self.beta = np.linspace(config_diff["beta_start"],
                                            config_diff["beta_end"], self.num_steps)
                self.alpha_hat = 1 - self.beta
                self.alpha = np.cumprod(self.alpha_hat)
                self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1)

            def time_embedding(self, pos, d_model=128):
                pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
                position = pos.unsqueeze(2)
                div_term = 1 / torch.pow(10000.0,
                                         torch.arange(0, d_model, 2).to(self.device) / d_model)
                pe[:, :, 0::2] = torch.sin(position * div_term)
                pe[:, :, 1::2] = torch.cos(position * div_term)
                return pe

            def get_side_info(self, observed_tp, cond_mask):
                B, K, L = cond_mask.shape
                time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
                time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
                feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
                feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
                side_info = torch.cat([time_embed, feature_embed], dim=-1)
                side_info = side_info.permute(0, 3, 2, 1)
                if not self.is_unconditional:
                    side_mask = cond_mask.unsqueeze(1)
                    side_info = torch.cat([side_info, side_mask], dim=1)
                return side_info

            def impute(self, observed_data, cond_mask, side_info, n_samples):
                B, K, L = observed_data.shape
                imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
                for i in range(n_samples):
                    current_sample = torch.randn_like(observed_data)
                    for t in range(self.num_steps - 1, -1, -1):
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                        predicted = self.diffmodel(diff_input, side_info,
                                                   torch.tensor([t]).to(self.device))
                        coeff1 = 1 / self.alpha_hat[t]**0.5
                        coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t])**0.5
                        current_sample = coeff1 * (current_sample - coeff2 * predicted)
                        if t > 0:
                            noise = torch.randn_like(current_sample)
                            sigma = ((1 - self.alpha[t-1]) / (1 - self.alpha[t]) * self.beta[t])**0.5
                            current_sample += sigma * noise
                    imputed_samples[:, i] = current_sample.detach()
                return imputed_samples

            def evaluate_batch(self, batch, n_samples):
                observed_data = batch["observed_data"].to(self.device).float().permute(0, 2, 1)
                observed_mask = batch["observed_mask"].to(self.device).float().permute(0, 2, 1)
                observed_tp = batch["timepoints"].to(self.device).float()
                gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
                with torch.no_grad():
                    cond_mask = gt_mask
                    target_mask = observed_mask - cond_mask
                    side_info = self.get_side_info(observed_tp, cond_mask)
                    samples = self.impute(observed_data, cond_mask, side_info, n_samples)
                    for i in range(n_samples):
                        samples[:, i] = cond_mask * observed_data + (1 - cond_mask) * samples[:, i]
                return samples, observed_data, target_mask, observed_mask

        with st.spinner("Chargement du modele CSDI..."):
            model = CSDI_Model(config, DEVICE, target_dim).to(DEVICE)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
            model.eval()

        with st.spinner(f"Imputation en cours ({N_SAMPLES} echantillons, {config['diffusion']['num_steps']} pas de diffusion)..."):
            if dataset_imp == "Yahoo Finance":
                test_ds = CSDIDataset(data_folder, "test", mask_ratio, SEQ_LENGTH, mask_type,
                                      target_dim, n_train=2734, n_val=501, n_test=945)
            else:
                test_ds = CSDIDataset(data_folder, "test", mask_ratio, SEQ_LENGTH, mask_type,
                                      target_dim)

            actual_idx = min(window_imp, len(test_ds) - 1)
            loader = DataLoader(test_ds, batch_size=1, shuffle=False)

            for i, batch in enumerate(loader):
                if i == actual_idx:
                    break

            samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)

            with open(os.path.join(data_folder, "meanstd.pkl"), "rb") as f:
                mean_data, std_data = pickle.load(f)

            std_cpu = torch.tensor(std_data).float().unsqueeze(-1)
            mean_cpu = torch.tensor(mean_data).float().unsqueeze(-1)

            samples_np = (samples[0].cpu() * std_cpu + mean_cpu).numpy()
            target_np = (target[0].cpu() * std_cpu + mean_cpu).numpy()
            tmask_np = target_mask[0].cpu().numpy()

        st.success("Imputation terminee !")

        # Metriques
        median_all = np.median(samples_np, axis=0)
        mask_flat = tmask_np.flatten() > 0
        target_flat = target_np.flatten()
        pred_flat = median_all.flatten()

        if mask_flat.sum() > 0:
            errors = target_flat[mask_flat] - pred_flat[mask_flat]
            imp_rmse = np.sqrt(np.mean(errors ** 2))
            imp_mae = np.mean(np.abs(errors))
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{imp_rmse:.6f}")
            col2.metric("MAE", f"{imp_mae:.6f}")
            col3.metric("Points imputes", f"{int(mask_flat.sum())}")

        # Graphique
        import matplotlib.pyplot as plt

        n_series = target_dim
        n_cols = 4 if n_series >= 4 else n_series
        n_rows = (n_series + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_series == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(n_series):
            ax = axes[i]
            t_vals = np.arange(SEQ_LENGTH)
            true_val = target_np[i]
            mask_i = tmask_np[i]

            ax.plot(t_vals, true_val, "k-", linewidth=1, label="Vrai", alpha=0.8)

            median_pred = np.median(samples_np[:, i, :], axis=0)
            q10 = np.quantile(samples_np[:, i, :], 0.1, axis=0)
            q90 = np.quantile(samples_np[:, i, :], 0.9, axis=0)

            masked_idx = np.where(mask_i > 0)[0]
            if len(masked_idx) > 0:
                for s_idx in range(samples_np.shape[0]):
                    sample_s = samples_np[s_idx, i, :]
                    ax.plot(t_vals[masked_idx], sample_s[masked_idx], color="#85c1e9", linewidth=0.4, alpha=0.3,
                            label="Echantillons" if s_idx == 0 else None)
                ax.fill_between(t_vals, q10, q90, alpha=0.3, color="#85c1e9", label="IC 80%", zorder=2)
                ax.plot(t_vals, median_pred, color="#2471a3", linewidth=1.2, alpha=0.9, label="Mediane CSDI", zorder=3)
                ax.scatter(masked_idx, true_val[masked_idx], color="red", s=15, zorder=6, label="Masque")

            ax.set_title(series_names_csdi[i], fontweight="bold")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)

        for j in range(n_series, len(axes)):
            axes[j].set_visible(False)

        mask_label = "aleatoire" if mask_type == "random" else "bloc"
        fig.suptitle(f"CSDI — Imputation {dataset_imp} (masquage {int(mask_ratio*100)}% {mask_label})",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Explication du processus
        st.subheader("🔍 Comment CSDI impute les donnees ?")
        st.markdown(f"""
        **Processus de diffusion inverse en {config['diffusion']['num_steps']} etapes :**

        1. **Masquage** : {int(mask_ratio*100)}% des valeurs sont masquees (type: {mask_label})
        2. **Bruit initial** : on part de bruit gaussien pur pour les positions masquees
        3. **Debruitage iteratif** : le Transformer predit le bruit a retirer a chaque etape
        4. **Conditionnement** : les valeurs **observees** guident l'imputation des valeurs masquees
        5. **Echantillonnage** : {N_SAMPLES} echantillons sont generes pour estimer l'incertitude

        **Resultats :**
        - La **ligne noire** = vraies valeurs
        - Les **points rouges** = positions masquees (ce que le modele doit deviner)
        - La **ligne bleue** = mediane des {N_SAMPLES} echantillons CSDI
        - La **zone bleue** = intervalle de confiance 80% (quantiles 10%-90%)

        Plus la zone bleue est etroite, plus le modele est **confiant** dans son imputation.
        Le masquage **bloc** est plus difficile car le modele n'a pas de valeurs proches pour s'aider.
        """)
