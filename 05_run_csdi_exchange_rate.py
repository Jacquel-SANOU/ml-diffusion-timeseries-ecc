"""
Etape 5 : Entrainer et evaluer CSDI sur Exchange Rate

CSDI (Conditional Score-based Diffusion for Imputation) :
  - Modele de diffusion conditionnel pour l'imputation de series temporelles
  - Utilise un mecanisme de masquage pour gerer l'imputation et la prevision
  - Genere 100 trajectoires pour quantifier l'incertitude

Tache d'imputation : on masque artificiellement des valeurs (10%, 30%, 50%)
et on mesure la qualite de la reconstruction.

Produit :
  - Modele entraine dans save/
  - Resultats d'evaluation (RMSE, MAE, CRPS)
  - Figures pour le rapport
"""

import sys
import os
import pickle
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CSDI"))
from diff_models import diff_CSDI

# =====================================================================
# DATASET : Exchange Rate avec masques d'imputation
# =====================================================================
class ExchangeRateDataset(Dataset):
    def __init__(self, data_folder, mode="train", mask_ratio=0.1,
                 seq_length=100, mask_type="random"):
        with open(os.path.join(data_folder, "data.pkl"), "rb") as f:
            self.main_data, self.mask_data = pickle.load(f)
        with open(os.path.join(data_folder, "meanstd.pkl"), "rb") as f:
            self.mean_data, self.std_data = pickle.load(f)

        self.main_data = (self.main_data - self.mean_data) / self.std_data
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.num_features = self.main_data.shape[1]

        total = len(self.main_data)
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)

        if mode == "train":
            start, end = 0, n_train - seq_length
        elif mode == "valid":
            start, end = n_train, n_train + n_val - seq_length
        else:
            start, end = n_train + n_val, total - seq_length

        self.use_index = np.arange(start, max(start + 1, end), 1)

    def __len__(self):
        return len(self.use_index)

    def __getitem__(self, idx):
        start = self.use_index[idx]
        observed = self.main_data[start:start + self.seq_length].copy()
        obs_mask = self.mask_data[start:start + self.seq_length].copy()

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


# =====================================================================
# MODELE CSDI adapte (sans forecasting, juste imputation)
# =====================================================================
class CSDI_ExchangeRate(nn.Module):
    def __init__(self, config, device, target_dim=8):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

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
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5,
                config_diff["beta_end"] ** 0.5,
                self.num_steps,
            ) ** 2
        else:
            self.beta = np.linspace(
                config_diff["beta_start"],
                config_diff["beta_end"],
                self.num_steps,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)

        predicted = self.diffmodel(total_input, side_info, t)
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                predicted = self.diffmodel(
                    diff_input, side_info,
                    torch.tensor([t]).to(self.device),
                )
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float().permute(0, 2, 1)
        observed_mask = batch["observed_mask"].to(self.device).float().permute(0, 2, 1)
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)

        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy == "random":
            rand_for_mask = torch.rand_like(observed_mask) * observed_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(observed_mask)):
                sample_ratio = np.random.rand()
                num_observed = observed_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        else:
            cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)
        if is_train == 1:
            return self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train)
        else:
            loss_sum = 0
            for t_step in range(self.num_steps):
                loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, 0, set_t=t_step)
                loss_sum += loss.detach()
            return loss_sum / self.num_steps

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
        return samples, observed_data, target_mask, observed_mask


# =====================================================================
# METRIQUES
# =====================================================================
def calc_metrics(samples, target, target_mask, mean_data, std_data):
    """Calcule RMSE, MAE, CRPS, couverture IC 80% et largeur moyenne IC."""
    median_pred = samples.median(dim=1).values

    target_orig = target * std_data + mean_data
    median_orig = median_pred * std_data + mean_data
    samples_orig = samples * std_data.unsqueeze(1) + mean_data.unsqueeze(1)

    diff = (median_orig - target_orig) * target_mask
    n_eval = target_mask.sum().item()
    if n_eval == 0:
        return 0, 0, 0, 0, 0

    mse = (diff ** 2).sum().item() / n_eval
    mae = diff.abs().sum().item() / n_eval
    rmse = np.sqrt(mse)

    q10 = torch.quantile(samples_orig, 0.1, dim=1)
    q90 = torch.quantile(samples_orig, 0.9, dim=1)
    in_interval = ((target_orig >= q10) & (target_orig <= q90)).float() * target_mask
    coverage = in_interval.sum().item() / n_eval if n_eval > 0 else 0
    width = ((q90 - q10) * target_mask).sum().item() / n_eval if n_eval > 0 else 0

    quantiles = np.arange(0.05, 1.0, 0.05)
    crps = 0
    denom = (target_orig.abs() * target_mask).sum().item()
    if denom > 0:
        for q in quantiles:
            q_pred = torch.quantile(samples_orig, q, dim=1)
            q_loss = 2 * torch.sum(
                torch.abs((q_pred - target_orig) * target_mask * ((target_orig <= q_pred).float() - q))
            ).item()
            crps += q_loss / denom
        crps /= len(quantiles)

    return rmse, mae, crps, coverage, width


# =====================================================================
# ENTRAINEMENT
# =====================================================================
def train_csdi(model, config, train_loader, valid_loader, foldername):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    train_losses = []
    valid_losses = []

    for epoch in range(config["epochs"]):
        model.train()
        avg_loss = 0
        n_batch = 0

        for batch_no, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            n_batch = batch_no
            if batch_no >= config["itr_per_epoch"]:
                break

        lr_scheduler.step()
        epoch_loss = avg_loss / n_batch
        train_losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            model.eval()
            avg_valid = 0
            n_valid = 0
            with torch.no_grad():
                for batch in valid_loader:
                    vloss = model(batch, is_train=0)
                    avg_valid += vloss.item()
                    n_valid += 1
            val_loss = avg_valid / max(n_valid, 1)
            valid_losses.append(val_loss)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), os.path.join(foldername, "model_best.pth"))

            print(f"  Epoch {epoch+1:3d}/{config['epochs']}  |  train_loss={epoch_loss:.4f}  |  val_loss={val_loss:.4f}")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{config['epochs']}  |  train_loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(foldername, "model_final.pth"))
    return train_losses, valid_losses


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", type=str, default="",
                        help="Path to saved model folder (skip training)")
    args = parser.parse_args()

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_FOLDER = "data/csdi_exchange_rate"
    CONFIG_PATH = "config/exchange_rate.yaml"
    N_SAMPLES = int(os.environ.get("N_SAMPLES", 50))
    SEQ_LENGTH = 100

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"CSDI — Entrainement sur Exchange Rate ({DEVICE.upper()})")
    print("=" * 60)
    print(f"Config : {json.dumps(config, indent=2)}")
    print(f"Device : {DEVICE}")
    print(f"Samples pour evaluation : {N_SAMPLES}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"save/csdi_exchange_rate_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # --- Charger les donnees ---
    print("\n--- Chargement des donnees ---")
    mask_ratios = {"10%": 0.1, "30%": 0.3, "50%": 0.5}

    train_ds = ExchangeRateDataset(DATA_FOLDER, "train", mask_ratio=0.1, seq_length=SEQ_LENGTH)
    valid_ds = ExchangeRateDataset(DATA_FOLDER, "valid", mask_ratio=0.1, seq_length=SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    print(f"  Train : {len(train_ds)} fenetres")
    print(f"  Valid : {len(valid_ds)} fenetres")

    # --- Creer le modele ---
    model = CSDI_ExchangeRate(config, DEVICE, target_dim=8).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametres du modele : {n_params:,}")

    # --- Entrainer ou charger ---
    if args.eval_only:
        save_folder = args.eval_only
        model_path = os.path.join(save_folder, "model_best.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        print(f"\n  Modele charge : {model_path}")
    else:
        print("\n--- Entrainement ---")
        train_losses, valid_losses = train_csdi(
            model, config["train"], train_loader, valid_loader, save_folder
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train loss", color="#3498db")
        val_epochs = list(range(9, len(train_losses), 10))
        if valid_losses:
            ax.plot(val_epochs, valid_losses, "o-", label="Valid loss", color="#e74c3c")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("CSDI — Courbe d'entrainement (Exchange Rate)", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/figures/14_csdi_training_loss.png", bbox_inches="tight")
        plt.close()
        print("\n>>> FIGURE 14 sauvegardee : results/figures/14_csdi_training_loss.png")

        model_path = os.path.join(save_folder, "model_best.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
            print("  Modele best charge")

    # --- Evaluation sur le test avec differents taux de masquage ---
    print("\n--- Evaluation sur le test ---")
    model.eval()

    with open(os.path.join(DATA_FOLDER, "meanstd.pkl"), "rb") as f:
        mean_data, std_data = pickle.load(f)
    mean_t = torch.tensor(mean_data).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)
    std_t = torch.tensor(std_data).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

    results_all = {}

    for mask_name, mask_ratio in mask_ratios.items():
        for mask_type in ["random", "block"]:
            key = f"{mask_name}_{mask_type}"
            print(f"\n  Test : masquage {mask_name} ({mask_type})")

            test_ds = ExchangeRateDataset(
                DATA_FOLDER, "test", mask_ratio=mask_ratio,
                seq_length=SEQ_LENGTH, mask_type=mask_type,
            )
            test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

            all_rmse, all_mae, all_crps, all_cov, all_width = [], [], [], [], []

            for batch_idx, batch in enumerate(test_loader):
                samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)
                rmse, mae, crps, cov, width = calc_metrics(samples, target, target_mask, mean_t, std_t)
                all_rmse.append(rmse)
                all_mae.append(mae)
                all_crps.append(crps)
                all_cov.append(cov)
                all_width.append(width)

            avg_rmse = np.mean(all_rmse)
            avg_mae = np.mean(all_mae)
            avg_crps = np.mean(all_crps)
            avg_cov = np.mean(all_cov)
            avg_width = np.mean(all_width)

            results_all[key] = {"RMSE": avg_rmse, "MAE": avg_mae, "CRPS": avg_crps,
                                "Coverage_80": avg_cov, "IC_width": avg_width}
            print(f"    RMSE  = {avg_rmse:.6f}")
            print(f"    MAE   = {avg_mae:.6f}")
            print(f"    CRPS  = {avg_crps:.6f}")
            print(f"    Couv. IC 80% = {avg_cov:.4f}  ({avg_cov*100:.1f}%)")
            print(f"    Largeur IC   = {avg_width:.6f}")

    # --- Sauvegarder les resultats ---
    with open(os.path.join(save_folder, "results_imputation.json"), "w") as f:
        json.dump(results_all, f, indent=4)
    print(f"\n>>> Resultats sauvegardes : {save_folder}/results_imputation.json")

    # --- FIGURE 15 : Exemple d'imputation ---
    print("\n--- Generation d'une figure d'imputation ---")
    test_ds_vis = ExchangeRateDataset(
        DATA_FOLDER, "test", mask_ratio=0.3, seq_length=SEQ_LENGTH, mask_type="random",
    )
    vis_loader = DataLoader(test_ds_vis, batch_size=1, shuffle=False)

    batch = next(iter(vis_loader))
    samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)

    std_cpu = std_t[0].cpu()  # (8, 1)
    mean_cpu = mean_t[0].cpu()  # (8, 1)
    samples_np = (samples[0].cpu() * std_cpu + mean_cpu).numpy()  # (n_samples, 8, 100)
    target_np = (target[0].cpu() * std_cpu + mean_cpu).numpy()  # (8, 100)
    tmask_np = target_mask[0].cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    series_names = ["Australie", "Angleterre", "Canada", "Suisse",
                    "Chine", "Japon", "Nouvelle-Zelande", "Singapour"]

    for i in range(8):
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
            ax.scatter(masked_idx, true_val[masked_idx], color="red", s=10, zorder=5, label="Masque")
            ax.plot(t_vals, median_pred, color="#3498db", linewidth=0.8, alpha=0.7, label="Mediane CSDI")
            ax.fill_between(t_vals, q10, q90, alpha=0.2, color="#3498db", label="IC 80%")

        ax.set_title(series_names[i], fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("CSDI — Imputation Exchange Rate (masquage 30% aleatoire)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/15_csdi_imputation_exemple.png", bbox_inches="tight")
    plt.close()
    print(">>> FIGURE 15 sauvegardee : results/figures/15_csdi_imputation_exemple.png")

    # --- Tableau recapitulatif ---
    print("\n" + "=" * 60)
    print("RESULTATS CSDI — IMPUTATION EXCHANGE RATE")
    print("=" * 60)
    print(f"{'Masquage':<20} {'RMSE':>10} {'MAE':>10} {'CRPS':>10}")
    print("-" * 52)
    for key, vals in results_all.items():
        print(f"{key:<20} {vals['RMSE']:>10.6f} {vals['MAE']:>10.6f} {vals['CRPS']:>10.6f}")
    print("=" * 60)
