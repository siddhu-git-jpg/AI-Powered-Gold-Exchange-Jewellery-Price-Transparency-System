"""Train an exchange-value regressor from jeweller-verified records.

CSV columns: gross_weight_g, stone_weight_g, net_weight_g, purity_fraction,
rate_per_gram, refining_percent, wastage_percent, verified_exchange_value_inr.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA = Path("training/exchange_records.csv")
TARGET = "verified_exchange_value_inr"
FEATURES = ["gross_weight_g", "stone_weight_g", "net_weight_g", "purity_fraction",
            "rate_per_gram", "refining_percent", "wastage_percent"]

frame = pd.read_csv(DATA).dropna(subset=FEATURES + [TARGET])
if len(frame) < 100:
    raise SystemExit("Need at least 100 verified records before training; collect a representative dataset first.")
x_train, x_test, y_train, y_test = train_test_split(
    frame[FEATURES], frame[TARGET], test_size=0.2, random_state=42
)
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                     subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                     n_jobs=4, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(f"Holdout MAE: ₹{mean_absolute_error(y_test, predictions):,.2f}")
out = ROOT / "models/exchange_xgb.joblib"
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out)
print(f"Saved {out}")
