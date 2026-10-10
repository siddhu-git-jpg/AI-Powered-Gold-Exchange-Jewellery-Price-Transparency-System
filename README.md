# Gold Exchange & Jewellery Price Transparency Calculator

The refreshed homepage is `outputs/index.html` (also available as `outputs/Gold-Exchange-Calculator.html`). It combines an auditable exchange formula with photo/bill uploads. Its optional Python ML service uses OpenCV quality checks, a CNN jewellery classifier, an XGBoost exchange-value regressor, an AI transparency-score model, and Linear Regression over verified historical rates. The service has training scripts and API contracts, but trained weights and labelled records are not bundled. Until those are supplied, the website uses the transparent formula and labels its score as an evidence-completeness proxy.

The existing Netlify function analyzes an uploaded jewellery photo and old purchase bill with the OpenAI Responses API. Keep `OPENAI_API_KEY` server-side; never place it in `outputs/index.html`.

## Run locally

Use Node.js 18 or later. Set `OPENAI_API_KEY` in the terminal environment, then run:

```sh
npm start
```

Open `http://localhost:4173`. Opening the HTML directly as a `file://` page will show the calculator, but server analysis needs the local server endpoint.

To enable the separate ML service, create a Python environment in `ml-service/`, install `ml-service/requirements.txt`, then run `uvicorn app:app --reload --port 8000` from that folder. In Netlify, set `ML_SERVICE_URL` to the deployed FastAPI service URL. Without it, the website's formula remains usable and the ML panel says the service is offline.

Add jeweller-verified training data before training or using learned predictions. See `ml-service/README.md` and its `train_*.py` files for the data columns, minimum record counts, and model paths. `ml-service/data/rate_history.csv` should contain verified 22K INR/gram rates with `date,rate_inr_g` columns; the regression is a trend estimate, not a live rate. The rate inputs on the website are manual and editable.

## Deploy on Netlify

Deploy this project with its `netlify.toml`. Set `OPENAI_API_KEY` as a server-side environment variable in the Netlify site settings, then redeploy. You may set `OPENAI_MODEL` there if your API project uses a different vision-capable model; the default is `gpt-6-astra`.

The function sends the selected image and/or PDF to the OpenAI API with response storage disabled. Image and PDF analysis uses API billing. The AI can describe the jewellery photo and extract clearly printed weight, purity, and invoice amount from a bill. It must not infer purity from appearance. Bill values are suggestions: customers should verify them before applying them to the calculator, and a jeweller must confirm final purity, weight, and deductions.
