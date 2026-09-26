# mle-assignment1-dpp

CS611 Assignment 1 - data processing pipelines (Medallion Architecture) for a loan default model.

## How to run

```
docker-compose build
docker-compose up          # JupyterLab at http://localhost:8888
python main.py             # in a terminal inside JupyterLab, creates datamart/
python sanity_check_model.py   # optional, trains a simple model on the feature store + label store
```

## Files

- `main.py` - runs bronze -> silver -> gold for every month from 2023-01 to 2025-12 (months without data are skipped, no empty files)
- `utils/data_processing_bronze_table.py` - copy raw csv into monthly bronze partitions
- `utils/data_processing_silver_table.py` - data types, cleaning, extra columns
- `utils/data_processing_gold_table.py` - label store (from Lab 2) and feature store
- `data/` - raw csv files
- `datamart/` - created by main.py (bronze / silver / gold), not in git

## Datamart

| layer | tables |
|---|---|
| bronze | lms, clickstream, attributes, financials (csv per snapshot_date) |
| silver | loan_daily, clickstream, attributes, financials (parquet per snapshot_date) |
| gold | label_store (one row per loan, label = 30dpd at mob 6), feature_store (one row per customer per application month) |

Rows: 137,500 lms (35 months), 215,376 clickstream (24 months), 12,500 attributes and financials (25 months).
Gold: 12,500 labels and 12,500 feature rows (110 features), every label matches a feature row.

Feature store and label store join on `Customer_ID`, with `feature_store.snapshot_date = label_store.snapshot_date - 6 months`.
Clickstream features only use months up to the application month (no future data).
Customers who applied from 2024-07 have no clickstream at all, so their clickstream features are null.
