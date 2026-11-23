"""
08 — CSDI sur Yahoo Finance (9 series, log-rendements)
Split temporel : Train 2010-2020 (2734), Val 2021-2022 (501), Test 2023-2026 (945)
Produit figures 23-25.
"""

import sys, os, pickle, json, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CSDI"))
from diff_models import diff_CSDI

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 100
N_SAMPLES = int(os.environ.get("N_SAMPLES", 50))

N_TRAIN = 2734
N_VAL = 501
N_TEST = 945
TARGET_DIM = 9

SERIES_NAMES = ["Apple", "EUR/USD", "Or (GLD)", "JPMorgan", "Microsoft",
                "Exxon", "CAC 40", "S&P 500", "VIX"]

# ── Dataset ─────────────────────────────────────────────
class YahooDataset(Dataset):
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

        if mode == "train":
            start, end = 0, N_TRAIN - seq_length
        elif mode == "valid":
            start, end = N_TRAIN, N_TRAIN + N_VAL - seq_length
        else:
            start, end = N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST - seq_length
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

# ── Model ───────────────────────────────────────────────
class CSDI_Yahoo(nn.Module):
    def __init__(self, config, device, target_dim=TARGET_DIM):
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
            self.beta = np.linspace(
                config_diff["beta_start"]**0.5,
                config_diff["beta_end"]**0.5,
                self.num_steps
            )**2
        else:
            self.beta = np.linspace(
                config_diff["beta_start"],
                config_diff["beta_end"],
                self.num_steps
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(device)
            .unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0,
            torch.arange(0, d_model, 2).to(self.device) / d_model
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
        feature_embed = (
            feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        )
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):
        B, K, L = observed_data.shape
        t = np.random.randint(0, self.num_steps)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (1 - current_alpha)**0.5 * noise
        target_mask = observed_mask - cond_mask
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        diff_input = torch.cat([cond_obs, noisy_target], dim=1)
        predicted = self.diffmodel(
            diff_input, side_info, torch.tensor([t]).to(self.device)
        )
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
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
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )
                coeff1 = 1 / self.alpha_hat[t]**0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t])**0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1 - self.alpha[t-1]) / (1 - self.alpha[t]) * self.beta[t]
                    )**0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch):
        observed_data = batch["observed_data"].to(self.device).float().permute(0, 2, 1)
        observed_mask = batch["observed_mask"].to(self.device).float().permute(0, 2, 1)
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info)
        return loss

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


# ── Metrics ─────────────────────────────────────────────
def calc_metrics(samples, target, target_mask, mean_data, std_data):
    mean_t = torch.tensor(mean_data).float().to(samples.device).unsqueeze(-1)
    std_t = torch.tensor(std_data).float().to(samples.device).unsqueeze(-1)
    target_orig = target * std_t + mean_t
    samples_orig = samples * std_t.unsqueeze(0) + mean_t.unsqueeze(0)
    median_pred = samples_orig.median(dim=1).values
    mask = target_mask > 0
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    errors = (median_pred - target_orig)[mask]
    rmse = (errors**2).mean().sqrt().item()
    mae = errors.abs().mean().item()
    # CRPS
    n_samples = samples_orig.shape[1]
    abs_diff = (samples_orig - target_orig.unsqueeze(1)).abs()
    mask_exp = mask.unsqueeze(1).expand_as(abs_diff)
    crps_term1 = abs_diff[mask_exp].mean().item()
    spread = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            spread += (samples_orig[:, i] - samples_orig[:, j]).abs()[mask].mean().item()
    if n_samples > 1:
        spread /= (n_samples * (n_samples - 1) / 2)
    crps = crps_term1 - 0.5 * spread
    # Coverage and width of 80% CI
    q10 = torch.quantile(samples_orig, 0.1, dim=1)
    q90 = torch.quantile(samples_orig, 0.9, dim=1)
    in_interval = ((target_orig >= q10) & (target_orig <= q90)).float()
    n_eval = mask.sum().item()
    coverage = (in_interval * target_mask).sum().item() / n_eval if n_eval > 0 else 0
    width = ((q90 - q10) * target_mask).sum().item() / n_eval if n_eval > 0 else 0
    return rmse, mae, crps, coverage, width


# ── Training ────────────────────────────────────────────
def train_csdi(config):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"save/csdi_yahoo_{stamp}"
    os.makedirs(save_dir, exist_ok=True)

    train_ds = YahooDataset("data/csdi_yahoo", "train", mask_ratio=0.1,
                            seq_length=SEQ_LENGTH, mask_type="random")
    val_ds = YahooDataset("data/csdi_yahoo", "valid", mask_ratio=0.1,
                          seq_length=SEQ_LENGTH, mask_type="random")
    print(f"  Train : {len(train_ds)} fenetres")
    print(f"  Valid : {len(val_ds)} fenetres")

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["train"]["batch_size"],
                            shuffle=False, num_workers=0)

    model = CSDI_Yahoo(config, DEVICE, target_dim=TARGET_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametres du modele : {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            if i >= config["train"]["itr_per_epoch"]:
                break
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_train = np.mean(epoch_loss)
        train_losses.append(avg_train)

        if epoch % 5 == 0:
            msg = f"  Epoch {epoch:4d}/{config['train']['epochs']}  |  train_loss={avg_train:.4f}"
            if epoch % 10 == 0:
                model.eval()
                vl = []
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        if j >= 50:
                            break
                        loss = model(batch)
                        vl.append(loss.item())
                avg_val = np.mean(vl)
                val_losses.append(avg_val)
                msg += f"  |  val_loss={avg_val:.4f}"
                if avg_val < best_val:
                    best_val = avg_val
                    torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            print(msg)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Figure 23 — Training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train", alpha=0.7)
    val_epochs = list(range(10, len(train_losses) + 1, 10))[:len(val_losses)]
    ax.plot(val_epochs, val_losses, "ro-", label="Validation", markersize=4)
    ax.set_xlabel("Epoque")
    ax.set_ylabel("Loss (diffusion)")
    ax.set_title("CSDI Yahoo Finance — Courbe d'apprentissage")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/23_csdi_yahoo_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n>>> FIGURE 23 sauvegardee")

    return model, save_dir


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", type=str, default=None)
    args = parser.parse_args()

    with open("config/yahoo.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("CSDI — Entrainement sur Yahoo Finance (9 series)")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    print(f"Samples pour evaluation : {N_SAMPLES}")

    with open("data/csdi_yahoo/meanstd.pkl", "rb") as f:
        mean_data, std_data = pickle.load(f)

    if args.eval_only:
        save_dir = args.eval_only
        model = CSDI_Yahoo(config, DEVICE, target_dim=TARGET_DIM).to(DEVICE)
        model.load_state_dict(torch.load(
            os.path.join(save_dir, "model_best.pth"),
            weights_only=True, map_location=DEVICE
        ))
        model.eval()
        print(f"  Modele charge : {save_dir}")
    else:
        print("\n--- Chargement des donnees ---")
        model, save_dir = train_csdi(config)

    # ── Evaluation imputation ──
    print("\n--- Evaluation sur le test ---")
    model.eval()
    results = {}

    for ratio in [0.1, 0.3, 0.5]:
        for mtype in ["random", "block"]:
            key = f"{int(ratio*100)}%_{mtype}"
            test_ds = YahooDataset("data/csdi_yahoo", "test",
                                   mask_ratio=ratio, seq_length=SEQ_LENGTH,
                                   mask_type=mtype)
            test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

            all_rmse, all_mae, all_crps, all_cov, all_width = [], [], [], [], []
            for bi, batch in enumerate(test_loader):
                samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)
                rmse, mae, crps, cov, width = calc_metrics(
                    samples, target, target_mask, mean_data, std_data
                )
                all_rmse.append(rmse)
                all_mae.append(mae)
                all_crps.append(crps)
                all_cov.append(cov)
                all_width.append(width)

            results[key] = {
                "RMSE": float(np.mean(all_rmse)),
                "MAE": float(np.mean(all_mae)),
                "CRPS": float(np.mean(all_crps)),
                "Coverage_80": float(np.mean(all_cov)),
                "IC_width": float(np.mean(all_width)),
            }
            print(f"\n  Test : masquage {key}")
            print(f"    RMSE  = {results[key]['RMSE']:.6f}")
            print(f"    MAE   = {results[key]['MAE']:.6f}")
            print(f"    CRPS  = {results[key]['CRPS']:.6f}")
            print(f"    Couv. IC 80% = {results[key]['Coverage_80']:.4f}  ({results[key]['Coverage_80']*100:.1f}%)")
            print(f"    Largeur IC   = {results[key]['IC_width']:.6f}")

    with open(os.path.join(save_dir, "results_imputation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n>>> Resultats sauvegardes : {save_dir}/results_imputation.json")

    # ── Figure 24 — Imputation example ──
    print("\n--- Generation figure imputation ---")
    test_ds_vis = YahooDataset("data/csdi_yahoo", "test",
                               mask_ratio=0.3, seq_length=SEQ_LENGTH,
                               mask_type="random")
    vis_loader = DataLoader(test_ds_vis, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(vis_loader))
    samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)

    std_cpu = torch.tensor(std_data).float().unsqueeze(-1)
    mean_cpu = torch.tensor(mean_data).float().unsqueeze(-1)
    samples_np = (samples[0].cpu() * std_cpu + mean_cpu).numpy()
    target_np = (target[0].cpu() * std_cpu + mean_cpu).numpy()
    tmask_np = target_mask[0].cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i in range(TARGET_DIM):
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
            ax.scatter(masked_idx, true_val[masked_idx], color="red", s=10,
                       zorder=5, label="Masque")
            ax.plot(t_vals, median_pred, color="#3498db", linewidth=0.8,
                    alpha=0.7, label="Mediane CSDI")
            ax.fill_between(t_vals, q10, q90, alpha=0.2, color="#3498db",
                            label="IC 80%")
        ax.set_title(SERIES_NAMES[i], fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("CSDI — Imputation Yahoo Finance (masquage 30% aleatoire)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/24_csdi_yahoo_imputation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(">>> FIGURE 24 sauvegardee")

    # ── Figure 25 — Bar chart comparing masking configs ──
    configs = list(results.keys())
    crps_vals = [results[k]["CRPS"] for k in configs]
    mae_vals = [results[k]["MAE"] for k in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(configs))
    colors = ["#2ecc71" if "random" in c else "#e74c3c" for c in configs]

    ax1.bar(x, crps_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.set_ylabel("CRPS")
    ax1.set_title("CRPS par configuration de masquage")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, mae_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE par configuration de masquage")
    ax2.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Aleatoire"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Bloc"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("CSDI Yahoo — Comparaison des configurations de masquage",
                 fontsize=13, fontweight="bold", y=1.10)
    plt.tight_layout()
    plt.savefig("results/figures/25_csdi_yahoo_masking_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(">>> FIGURE 25 sauvegardee")

    print("\n" + "=" * 60)
    print("TERMINE — Resultats Yahoo Finance CSDI")
    print("=" * 60)
