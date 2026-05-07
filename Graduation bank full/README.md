# Bank Marketing Predictor

This project turns the bank marketing notebook workflow into a small application with:

- `train.py` for offline model training and artifact generation
- `app/main.py` for the FastAPI prediction service
- `ui/streamlit_app.py` for the Streamlit web UI

The production model uses a sklearn preprocessing pipeline plus `GradientBoostingClassifier`, and predictions use a tuned probability threshold selected on a validation split to improve F1 and recall on the positive class.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train The Model

```bash
python train.py
```

This creates:

- `artifacts/bank_marketing_model.joblib`
- `artifacts/bank_marketing_metadata.json`

## Run FastAPI

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the API docs at `http://127.0.0.1:8000/docs`.

## Run Streamlit

```bash
streamlit run ui/streamlit_app.py
```

If the API is not running on the default address, set:

```bash
set BANK_API_URL=http://127.0.0.1:8000
```

or update the URL from the Streamlit sidebar.

## Run The API Tests

```bash
python -m unittest tests.test_api
```
