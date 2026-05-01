# bank-retention-analytics

Streamlit dashboard for customer engagement and retention analysis using European Bank customer data.

## Setup

1. Create a Python environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

If the dataset is stored in a `data/` folder, the app will automatically load `data/European_Bank.csv`. Otherwise it falls back to `European_Bank.csv` in the repository root.

## Features

- Filter customers by geography, gender, age, balance, and number of products
- Visualize churn behavior by engagement profile, product utilization, and high-value customers
- Score retention strength using a relationship score model
