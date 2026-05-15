# 🛒 Walmart Inventory Forecast App

A Streamlit app to forecast Walmart store sales and plan inventory.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## App Pages

| Page | Description |
|---|---|
| 📂 Upload Data | Upload CSV, session memory, clear cache button |
| 📊 EDA Dashboard | Sales trends, store analysis, correlation heatmap |
| 🔮 Forecast | Configurable weeks (1–26), LightGBM / XGBoost |
| 📦 Inventory | Safety stock, reorder points, inventory plan |

## Required CSV Columns

| Column | Type |
|---|---|
| Store | int |
| Date | date (DD-MM-YYYY) |
| Weekly_Sales | float |
| Holiday_Flag | int (0 or 1) |
| Temperature | float |
| Fuel_Price | float |
| CPI | float |
| Unemployment | float |

## Folder Structure

```
Inventory_Forecast/
├── app.py
├── config.py
├── requirements.txt
├── backend/
│   ├── data_loader.py
│   ├── eda.py
│   ├── features.py
│   ├── model.py
│   └── inventory.py
└── ui/
    ├── upload_page.py
    ├── eda_page.py
    ├── forecast_page.py
    └── inventory_page.py
```
