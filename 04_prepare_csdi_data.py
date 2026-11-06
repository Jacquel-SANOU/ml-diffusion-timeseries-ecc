"""
Etape 4 : Convertir les donnees au format pickle pour CSDI

CSDI attend :
  - data.pkl : (main_data, mask_data)  avec shapes (T, N)
  - meanstd.pkl : (mean, std)

On prepare Exchange Rate (8 series) et Yahoo Finance (9 series).
"""

import pandas as pd
import numpy as np
import pickle
import os

# =====================================================================
# 1. EXCHANGE RATE -> format CSDI
# =====================================================================
print("=" * 60)
print("PREPARATION DES DONNEES POUR CSDI")
print("=" * 60)

out_dir = "data/csdi_exchange_rate"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv("exchange_rate.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
data = df.values.astype(np.float32)

mean_data = data.mean(axis=0)
std_data = data.std(axis=0)
mask_data = np.ones_like(data, dtype=np.float32)

print(f"\nExchange Rate :")
print(f"  Shape : {data.shape}  (T={data.shape[0]}, N={data.shape[1]})")
print(f"  Mean  : {mean_data.round(4)}")
print(f"  Std   : {std_data.round(4)}")

with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
    pickle.dump([data, mask_data], f)
with open(os.path.join(out_dir, "meanstd.pkl"), "wb") as f:
    pickle.dump([mean_data, std_data], f)

print(f"  >>> {out_dir}/data.pkl")
print(f"  >>> {out_dir}/meanstd.pkl")

# =====================================================================
# 2. YAHOO FINANCE -> format CSDI
# =====================================================================
out_dir_yahoo = "data/csdi_yahoo"
os.makedirs(out_dir_yahoo, exist_ok=True)

log_ret = pd.read_csv("data/processed/yahoo_log_returns.csv", index_col="Date", parse_dates=True)
data_yahoo = log_ret.values.astype(np.float32)

mean_yahoo = data_yahoo.mean(axis=0)
std_yahoo = data_yahoo.std(axis=0)
mask_yahoo = np.ones_like(data_yahoo, dtype=np.float32)

print(f"\nYahoo Finance (log-rendements) :")
print(f"  Shape : {data_yahoo.shape}  (T={data_yahoo.shape[0]}, N={data_yahoo.shape[1]})")

with open(os.path.join(out_dir_yahoo, "data.pkl"), "wb") as f:
    pickle.dump([data_yahoo, mask_yahoo], f)
with open(os.path.join(out_dir_yahoo, "meanstd.pkl"), "wb") as f:
    pickle.dump([mean_yahoo, std_yahoo], f)

print(f"  >>> {out_dir_yahoo}/data.pkl")
print(f"  >>> {out_dir_yahoo}/meanstd.pkl")

print("\n" + "=" * 60)
print("DONNEES CSDI PRETES")
print("=" * 60)
