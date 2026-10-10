"""Optional ML service for the Gold Exchange calculator.

The service deliberately reports when trained assets are missing. A generic
image model cannot establish jewellery purity or weight from appearance.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
CNN_PATH = Path(os.getenv("JEWELLERY_CNN_MODEL", APP_DIR / "models/jewellery_cnn.keras"))
XGB_PATH = Path(os.getenv("EXCHANGE_XGB_MODEL", APP_DIR / "models/exchange_xgb.joblib"))
SCORE_PATH = Path(os.getenv("TRANSPARENCY_SCORE_MODEL", APP_DIR / "models/transparency_score.joblib"))
RATE_HISTORY = Path(os.getenv("RATE_HISTORY_CSV", APP_DIR / "data/rate_history.csv"))
CLASS_NAMES = json.loads(os.getenv("JEWELLERY_CLASSES", '["bangle", "chain", "earrings", "necklace", "pendant", "ring"]'))
REFINING_PERCENT = float(os.getenv("REFINING_DEDUCTION_PERCENT", "1.2"))

app = FastAPI(title="Gold Exchange ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:4173,http://127.0.0.1:4173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)

_cnn: Any = None
_xgb: Any = None
_score_model: Any = None


class AnalysisRequest(BaseModel):
    photo_data_url: str | None = None
    metal_type: str = Field(default="gold", pattern="^(gold|silver)$")
    gross_weight_g: float = Field(ge=0, le=10000)
    stone_weight_g: float = Field(default=0, ge=0, le=10000)
    # Gold uses karat (18–24); silver can use fineness (925/999).
    purity_karat: float = Field(default=22, gt=0, le=999)
    rate_per_gram: float = Field(gt=0, le=10_000_000)
    refining_percent: float = Field(default=REFINING_PERCENT, ge=0, le=100)
    wastage_percent: float = Field(default=0, ge=0, le=100)
    has_bill: bool = False
    has_hallmark: bool = False


def _load_image(data_url: str) -> np.ndarray:
    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
    except (IndexError, ValueError) as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc
    if len(raw) > 8_000_000:
        raise ValueError("The uploaded image is too large for the ML service.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Choose a valid JPG, PNG, or WebP jewellery photo.")
    return image


def _classify(image: np.ndarray) -> dict[str, Any]:
    global _cnn
    if not CNN_PATH.is_file():
        return {"status": "model_required", "label": None, "confidence": None,
                "message": "A jewellery CNN trained on labelled examples is not installed."}
    if _cnn is None:
        try:
            from tensorflow.keras.models import load_model
            _cnn = load_model(CNN_PATH, compile=False)
        except Exception as exc:
            return {"status": "model_error", "label": None, "confidence": None,
                    "message": f"CNN model could not be loaded: {type(exc).__name__}."}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    # Model training must use the same scaling and image size.
    tensor = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
    probabilities = np.asarray(_cnn.predict(tensor, verbose=0)).reshape(-1)
    if probabilities.size != len(CLASS_NAMES):
        return {"status": "model_error", "label": None, "confidence": None,
                "message": "CNN output size does not match JEWELLERY_CLASSES."}
    index = int(np.argmax(probabilities))
    return {"status": "ready", "label": CLASS_NAMES[index],
            "confidence": round(float(probabilities[index]), 4)}


def _image_quality(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    issues = []
    if min(width, height) < 300:
        issues.append("low_resolution")
    if blur_score < 45:
        issues.append("possibly_blurry")
    if brightness < 45:
        issues.append("possibly_too_dark")
    elif brightness > 220:
        issues.append("possibly_overexposed")
    return {"width": width, "height": height,
            "sharpness_indicator": round(blur_score, 1),
            "brightness_indicator": round(brightness, 1), "issues": issues}


def _features(req: AnalysisRequest) -> tuple[list[float], float, float]:
    net = max(0.0, req.gross_weight_g - req.stone_weight_g)
    purity_fraction = (req.purity_karat / 999.0 if req.metal_type == "silver"
                       else req.purity_karat / 24.0)
    pure_metal_value = net * req.rate_per_gram * purity_fraction
    rule_value = pure_metal_value * (1 - req.refining_percent / 100) * (1 - req.wastage_percent / 100)
    values = [req.gross_weight_g, req.stone_weight_g, net, purity_fraction,
              req.rate_per_gram, req.refining_percent, req.wastage_percent]
    return values, pure_metal_value, rule_value


def _value(req: AnalysisRequest, features: list[float], rule_value: float) -> dict[str, Any]:
    global _xgb
    if not XGB_PATH.is_file():
        return {"status": "rules_fallback", "value_inr": round(rule_value, 2),
                "message": "No trained XGBoost model is installed; showing the auditable rate formula."}
    if _xgb is None:
        try:
            _xgb = joblib.load(XGB_PATH)
        except Exception as exc:
            return {"status": "model_error", "value_inr": round(rule_value, 2),
                    "message": f"XGBoost model could not be loaded: {type(exc).__name__}."}
    try:
        prediction = float(_xgb.predict(np.asarray([features], dtype=float))[0])
    except Exception as exc:
        return {"status": "model_error", "value_inr": round(rule_value, 2),
                "message": f"XGBoost inference failed: {type(exc).__name__}."}
    if not np.isfinite(prediction) or prediction < 0:
        return {"status": "model_error", "value_inr": round(rule_value, 2),
                "message": "The model returned an invalid value; showing the formula instead."}
    return {"status": "ready", "value_inr": round(prediction, 2),
            "model_difference_percent": round((prediction - rule_value) / max(rule_value, 1) * 100, 2)}


def _evidence_score(req: AnalysisRequest, photo_present: bool, features: list[float],
                    value: dict[str, Any], formula_value: float) -> dict[str, Any]:
    global _score_model
    # An AI score is only returned when a trained and validated score model exists.
    valid_purity = (0 < req.purity_karat <= 24 if req.metal_type == "gold"
                    else 0 < req.purity_karat <= 999)
    checks = [photo_present, req.has_bill, req.has_hallmark,
              req.gross_weight_g > req.stone_weight_g >= 0,
              valid_purity, req.rate_per_gram > 0]
    score = round(sum(checks) / len(checks) * 100)
    if SCORE_PATH.is_file():
        try:
            if _score_model is None:
                _score_model = joblib.load(SCORE_PATH)
            model_features = features + [int(req.has_bill), int(req.has_hallmark),
                                         float(value.get("model_difference_percent", 0))]
            predicted = float(_score_model.predict(np.asarray([model_features], dtype=float))[0])
            if np.isfinite(predicted):
                return {"status": "trained_model", "score": round(min(100, max(0, predicted))),
                        "label": "Price transparency score",
                        "note": "Model score; inspect the disclosed rate and each deduction."}
        except Exception:
            pass
    return {"status": "untrained_fallback", "score": score,
            "label": "Evidence completeness (temporary proxy)",
            "checks_passed": sum(checks), "checks_total": len(checks),
            "note": "The AI scoring model is not trained. This proxy measures evidence completeness, not offer fairness."}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"service": "ready", "cnn_model": CNN_PATH.is_file(),
            "xgboost_model": XGB_PATH.is_file(), "rate_history": RATE_HISTORY.is_file(),
            "opencv": cv2.__version__}


@app.post("/analyze")
def analyze(req: AnalysisRequest) -> dict[str, Any]:
    image = None
    if req.photo_data_url:
        try:
            image = _load_image(req.photo_data_url)
        except ValueError as exc:
            return {"status": "invalid_image", "error": str(exc)}
    features, metal_value, rule_value = _features(req)
    item = _classify(image) if image is not None else {
        "status": "photo_missing", "label": None, "confidence": None,
        "message": "Add an ornament photo to run the CNN classifier."}
    valuation = _value(req, features, rule_value)
    return {
        "status": "complete",
        "image_processing": {"status": "ready" if image is not None else "photo_missing",
                             "method": "OpenCV image validation and quality checks; CNN input resized to 224×224",
                             "quality": _image_quality(image) if image is not None else None,
                             "limitations": "OpenCV quality checks do not certify purity or weight. Jewellery, stone and hallmark localization still needs trained region labels and a localization model."},
        "classification": item,
        "valuation": {**valuation,
                      "formula_value_inr": round(rule_value, 2),
                      "metal_value_before_deductions_inr": round(metal_value, 2),
                      "model": "XGBoost Regressor" if XGB_PATH.is_file() else "Transparent formula",
                      "refining_percent": req.refining_percent},
        "transparency_score": _evidence_score(req, image is not None, features,
                                                valuation, rule_value),
        "input_features": {"gross_weight_g": req.gross_weight_g,
                           "stone_weight_g": req.stone_weight_g,
                           "net_weight_g": features[2],
                           "purity_fraction": features[3],
                           "rate_per_gram": req.rate_per_gram},
    }


@app.get("/trend")
def trend(days: int = 7) -> dict[str, Any]:
    """Fit a simple regression to the supplied historical rate CSV."""
    if not RATE_HISTORY.is_file():
        return {"status": "history_required", "message": "Add dated, verified rates to data/rate_history.csv."}
    try:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        frame = pd.read_csv(RATE_HISTORY)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["rate_inr_g"] = pd.to_numeric(frame["rate_inr_g"], errors="coerce")
        frame = frame.dropna(subset=["date", "rate_inr_g"]).sort_values("date")
        if len(frame) < 5:
            return {"status": "insufficient_history", "message": "At least five verified daily rates are required."}
        x = np.arange(len(frame), dtype=float).reshape(-1, 1)
        model = LinearRegression().fit(x, frame["rate_inr_g"].to_numpy())
        horizon = min(max(int(days), 1), 30)
        forecast = model.predict(np.arange(len(frame), len(frame) + horizon).reshape(-1, 1))
        return {"status": "forecast", "method": "Linear Regression",
                "history_points": len(frame), "forecast_days": horizon,
                "predicted_rate_inr_g": [round(float(v), 2) for v in forecast],
                "warning": "Trend estimate only; not a live rate or a guaranteed market forecast."}
    except Exception as exc:
        return {"status": "history_error", "message": f"Trend calculation failed: {type(exc).__name__}."}
