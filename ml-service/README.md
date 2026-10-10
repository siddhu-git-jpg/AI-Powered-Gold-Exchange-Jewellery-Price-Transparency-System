# Optional ML service

This FastAPI service adds OpenCV image preparation, a CNN jewellery classifier,
an XGBoost exchange-value regressor, a trained transparency score model, and a
Linear Regression metal-rate trend endpoint. The Node website proxies requests
to it at `http://127.0.0.1:8000`; configure `ML_SERVICE_URL` when deployed.
Netlify uses the `/api/ml/analyze` and `/api/ml/trend` proxy functions, which
also require `ML_SERVICE_URL` to point at a reachable deployment.

## Run

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Start the website in another terminal with `npm start`. The calculator keeps
its current transparent formula when the optional service or trained models are
unavailable. CNN and XGBoost inference only become active after you place
compatible trained artifacts in `models/` or configure the model path variables.

## Model/data contract

- `models/jewellery_cnn.keras`: CNN with outputs in the order in
  `JEWELLERY_CLASSES`; train with RGB 224×224 images scaled to 0–1. Run
  `python train_cnn.py` after adding labelled folders to `training/jewellery_images/`.
- `models/exchange_xgb.joblib`: XGBRegressor trained using the seven ordered
  features documented in `train_exchange_xgb.py`.
- `models/transparency_score.joblib`: reviewed-label model trained using the
  ten ordered features documented in `train_transparency_score.py`.
- Run `python train_exchange_xgb.py` and `python train_transparency_score.py`
  after adding the verified CSVs described in each script.
- `data/rate_history.csv`: verified daily rates with `date,rate_inr_g` columns;
  `/trend?days=7` fits Linear Regression and labels its output as a trend, not a
  live quote or guaranteed forecast.

Collect customer consent, minimize personal data, remove duplicate records,
validate dates and rates, and split model evaluation by time and jewellery
type. Do not use a photo to certify weight or purity. The transparency score
needs independently reviewed labels and published scoring criteria; until then
the API explicitly returns an evidence-completeness proxy, not a fairness score.
