"""Train a score model using audited store offers and human-reviewed labels.

CSV columns: the seven exchange features, has_bill, has_hallmark,
model_difference_percent, transparency_score (0–100).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = Path("training/transparency_labels.csv")
TARGET = "transparency_score"
FEATURES = ["gross_weight_g", "stone_weight_g", "net_weight_g", "purity_fraction",
            "rate_per_gram", "refining_percent", "wastage_percent", "has_bill",
            "has_hallmark", "model_difference_percent"]
frame = pd.read_csv(DATA).dropna(subset=FEATURES + [TARGET])
if len(frame) < 200:
    raise SystemExit("Need at least 200 independently reviewed offers before training a score model.")
x_train, x_test, y_train, y_test = train_test_split(
    frame[FEATURES], frame[TARGET].clip(0, 100), test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                              random_state=42, n_jobs=4)
model.fit(x_train, y_train)
print(f"Holdout MAE: {mean_absolute_error(y_test, model.predict(x_test)):.2f} points")
out = ROOT / "models/transparency_score.joblib"
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out)
print(f"Saved {out}")
