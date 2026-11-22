"""
Etape 7 : Baselines d'imputation

Baselines de reference pour comparer avec CSDI :
  1. Zero (imputer par 0, la moyenne des donnees normalisees)
  2. Interpolation lineaire (interpoler entre voisins non masques)

On evalue sur les memes configurations de masquage que CSDI
(10%, 30%, 50% — aleatoire et bloc) sur Exchange Rate et Yahoo Finance.
"""

import os
import json
import pickle
import numpy as np

np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LENGTH = 100


def create_test_windows(main_data, mask_data, start_idx, end_idx, seq_length):
    indices = np.arange(start_idx, max(start_idx + 1, end_idx - seq_length))
    windows = []
    for s in indices:
        observed = main_data[s:s + seq_length].copy()
        obs_mask = mask_data[s:s + seq_length].copy()
        windows.append((observed, obs_mask))
    return windows


def apply_mask(obs_mask, mask_ratio, mask_type, seq_length):
    gt_mask = obs_mask.copy()
    if mask_type == "random":
        rand = np.random.rand(*gt_mask.shape)
        gt_mask[rand < mask_ratio] = 0.0
    elif mask_type == "block":
        block_len = max(1, int(seq_length * mask_ratio))
        block_start = np.random.randint(0, seq_length - block_len + 1)
        gt_mask[block_start:block_start + block_len, :] = 0.0
    return gt_mask


def impute_zero(observed, obs_mask, gt_mask):
    target_mask = obs_mask - gt_mask
    imputed = observed.copy()
    imputed[target_mask > 0] = 0.0
    return imputed, target_mask


def impute_interpolation(observed, obs_mask, gt_mask):
    target_mask = obs_mask - gt_mask
    imputed = observed.copy()
    n_features = observed.shape[1]
    for f in range(n_features):
        mask_f = target_mask[:, f]
        masked_idx = np.where(mask_f > 0)[0]
        if len(masked_idx) == 0:
            continue
        known_idx = np.where(gt_mask[:, f] > 0)[0]
        if len(known_idx) < 2:
            imputed[masked_idx, f] = 0.0
            continue
        imputed[masked_idx, f] = np.interp(
            masked_idx, known_idx, observed[known_idx, f]
        )
    return imputed, target_mask


def impute_gaussian_conditional(observed, obs_mask, gt_mask, cov_matrix):
    """Gaussian conditional: E[x_missing | x_observed] using train covariance."""
    target_mask = obs_mask - gt_mask
    imputed = observed.copy()
    n_time, n_feat = observed.shape
    for t in range(n_time):
        obs_idx = np.where(gt_mask[t] > 0)[0]
        miss_idx = np.where(target_mask[t] > 0)[0]
        if len(miss_idx) == 0:
            continue
        if len(obs_idx) == 0:
            imputed[t, miss_idx] = 0.0
            continue
        sigma_mo = cov_matrix[np.ix_(miss_idx, obs_idx)]
        sigma_oo = cov_matrix[np.ix_(obs_idx, obs_idx)]
        try:
            imputed[t, miss_idx] = sigma_mo @ np.linalg.solve(sigma_oo, observed[t, obs_idx])
        except np.linalg.LinAlgError:
            imputed[t, miss_idx] = 0.0
    return imputed, target_mask


def calc_metrics_numpy(imputed, true_data, target_mask, mean_data, std_data):
    true_orig = true_data * std_data + mean_data
    imp_orig = imputed * std_data + mean_data
    diff = (imp_orig - true_orig) * target_mask
    n_eval = target_mask.sum()
    if n_eval == 0:
        return 0.0, 0.0
    mse = (diff ** 2).sum() / n_eval
    mae = np.abs(diff).sum() / n_eval
    rmse = np.sqrt(mse)
    return float(rmse), float(mae)


def evaluate_baselines(data_folder, split_info, dataset_name):
    with open(os.path.join(data_folder, "data.pkl"), "rb") as f:
        main_data, mask_data = pickle.load(f)
    with open(os.path.join(data_folder, "meanstd.pkl"), "rb") as f:
        mean_data, std_data = pickle.load(f)

    norm_data = (main_data - mean_data) / std_data

    # Train covariance for gaussian conditional baseline
    train_end = split_info.get("train_end", split_info["test_start"])
    train_norm = norm_data[:train_end]
    cov_matrix = np.cov(train_norm.T)

    start_test, end_test = split_info["test_start"], split_info["test_end"]
    windows = create_test_windows(norm_data, mask_data, start_test, end_test, SEQ_LENGTH)
    print(f"\n  {dataset_name} : {len(windows)} fenetres de test")

    results = {}
    for ratio in [0.1, 0.3, 0.5]:
        for mtype in ["random", "block"]:
            key = f"{int(ratio*100)}%_{mtype}"

            all_zero_rmse, all_zero_mae = [], []
            all_interp_rmse, all_interp_mae = [], []
            all_gauss_rmse, all_gauss_mae = [], []

            for observed, obs_mask in windows:
                gt_mask = apply_mask(obs_mask, ratio, mtype, SEQ_LENGTH)

                imp_zero, tmask = impute_zero(observed, obs_mask, gt_mask)
                rmse_z, mae_z = calc_metrics_numpy(imp_zero, observed, tmask, mean_data, std_data)
                all_zero_rmse.append(rmse_z)
                all_zero_mae.append(mae_z)

                imp_interp, tmask = impute_interpolation(observed, obs_mask, gt_mask)
                rmse_i, mae_i = calc_metrics_numpy(imp_interp, observed, tmask, mean_data, std_data)
                all_interp_rmse.append(rmse_i)
                all_interp_mae.append(mae_i)

                imp_gauss, tmask = impute_gaussian_conditional(observed, obs_mask, gt_mask, cov_matrix)
                rmse_g, mae_g = calc_metrics_numpy(imp_gauss, observed, tmask, mean_data, std_data)
                all_gauss_rmse.append(rmse_g)
                all_gauss_mae.append(mae_g)

            results[key] = {
                "Zero": {"RMSE": float(np.mean(all_zero_rmse)), "MAE": float(np.mean(all_zero_mae))},
                "Interpolation": {"RMSE": float(np.mean(all_interp_rmse)), "MAE": float(np.mean(all_interp_mae))},
                "Gaussian_conditional": {"RMSE": float(np.mean(all_gauss_rmse)), "MAE": float(np.mean(all_gauss_mae))},
            }
            print(f"    {key}")
            print(f"      Zero          RMSE={results[key]['Zero']['RMSE']:.6f}  MAE={results[key]['Zero']['MAE']:.6f}")
            print(f"      Interpolation RMSE={results[key]['Interpolation']['RMSE']:.6f}  MAE={results[key]['Interpolation']['MAE']:.6f}")
            print(f"      Gauss. cond.  RMSE={results[key]['Gaussian_conditional']['RMSE']:.6f}  MAE={results[key]['Gaussian_conditional']['MAE']:.6f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINES D'IMPUTATION")
    print("=" * 60)

    # Exchange Rate: 80/10/10 split
    with open("data/csdi_exchange_rate/data.pkl", "rb") as f:
        er_data, _ = pickle.load(f)
    total_er = len(er_data)
    n_train_er = int(total_er * 0.8)
    n_val_er = int(total_er * 0.1)

    er_results = evaluate_baselines(
        "data/csdi_exchange_rate",
        {"test_start": n_train_er + n_val_er, "test_end": total_er},
        "Exchange Rate",
    )

    # Yahoo Finance: fixed split
    yahoo_results = evaluate_baselines(
        "data/csdi_yahoo",
        {"test_start": 2734 + 501, "test_end": 2734 + 501 + 945},
        "Yahoo Finance",
    )

    all_results = {"exchange_rate": er_results, "yahoo": yahoo_results}
    out_path = os.path.join(RESULTS_DIR, "baselines_imputation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n>>> Resultats sauvegardes : {out_path}")

    print("\n" + "=" * 60)
    print("BASELINES D'IMPUTATION TERMINEES")
    print("=" * 60)
