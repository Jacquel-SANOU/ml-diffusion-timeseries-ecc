"""Generate CSDI imputation figure only (model already trained)."""

import sys, os, pickle, json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CSDI"))

# Inline the needed classes to avoid re-running everything
from torch.utils.data import Dataset
from diff_models import diff_CSDI
import torch.nn as nn

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
        total = len(self.main_data)
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)
        if mode == "test":
            start, end = n_train + n_val, total - seq_length
        else:
            start, end = 0, n_train - seq_length
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

class CSDI_ExchangeRate(nn.Module):
    def __init__(self, config, device, target_dim=8):
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
            self.beta = np.linspace(config_diff["beta_start"]**0.5, config_diff["beta_end"]**0.5, self.num_steps)**2
        else:
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
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
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t-1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
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
        return samples, observed_data, target_mask, observed_mask


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 100
N_SAMPLES = 10

with open("config/exchange_rate.yaml", "r") as f:
    config = yaml.safe_load(f)

model = CSDI_ExchangeRate(config, DEVICE, target_dim=8).to(DEVICE)
model.load_state_dict(torch.load(
    "save/csdi_exchange_rate_20261106_111634/model_best.pth",
    weights_only=True, map_location=DEVICE
))
model.eval()
print("Modele charge")

with open("data/csdi_exchange_rate/meanstd.pkl", "rb") as f:
    mean_data, std_data = pickle.load(f)

std_cpu = torch.tensor(std_data).float().unsqueeze(-1)   # (8, 1)
mean_cpu = torch.tensor(mean_data).float().unsqueeze(-1)  # (8, 1)

# Generate figure
test_ds = ExchangeRateDataset("data/csdi_exchange_rate", "test",
                               mask_ratio=0.3, seq_length=SEQ_LENGTH, mask_type="random")
vis_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

batch = next(iter(vis_loader))
samples, target, target_mask, obs_mask = model.evaluate_batch(batch, N_SAMPLES)

samples_np = (samples[0].cpu() * std_cpu + mean_cpu).numpy()  # (n_samples, 8, 100)
target_np = (target[0].cpu() * std_cpu + mean_cpu).numpy()     # (8, 100)
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
print(">>> FIGURE 15 sauvegardee")

# Also print results table
with open("save/csdi_exchange_rate_20261106_111634/results_imputation.json") as f:
    results = json.load(f)
print("\nRESULTATS CSDI IMPUTATION:")
print(f"{'Masquage':<20} {'RMSE':>10} {'MAE':>10} {'CRPS':>10}")
print("-" * 52)
for key, vals in results.items():
    print(f"{key:<20} {vals['RMSE']:>10.6f} {vals['MAE']:>10.6f} {vals['CRPS']:>10.6f}")
