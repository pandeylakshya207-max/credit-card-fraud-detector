"""
fix_scalers.py — Fixes a bug in the original train_model.py: a single
StandardScaler object was fit_transform'd twice (once for Time, once for
Amount), so the second call silently overwrote the first's fitted
statistics. The saved scaler.pkl ended up only holding Amount's mean/std,
which would incorrectly scale any new Time value at inference.

The trained model itself is UNAFFECTED — it was trained on correctly
computed Time_scaled/Amount_scaled column values at the time each
fit_transform() call ran (Python evaluates each line correctly in sequence;
the bug only affected what got *persisted* for reuse afterward). This
script regenerates two separate, correctly-fit scalers using the exact
same full-dataset fitting procedure as the original, producing scalers
that reproduce the identical training-time values — no retraining needed.

Run:
    python3 fix_scalers.py
"""
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

time_scaler = StandardScaler()
time_scaler.fit(df[["Time"]])

amount_scaler = StandardScaler()
amount_scaler.fit(df[["Amount"]])

print(f"Time scaler   — mean: {time_scaler.mean_[0]:.2f}, scale: {time_scaler.scale_[0]:.2f}")
print(f"Amount scaler — mean: {amount_scaler.mean_[0]:.2f}, scale: {amount_scaler.scale_[0]:.2f}")

with open("time_scaler.pkl", "wb") as f:
    pickle.dump(time_scaler, f)
with open("amount_scaler.pkl", "wb") as f:
    pickle.dump(amount_scaler, f)

print("Saved: time_scaler.pkl, amount_scaler.pkl (replaces the buggy shared scaler.pkl)")
