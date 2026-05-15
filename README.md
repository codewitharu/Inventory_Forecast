# 🛒 Inventory Forecast App

A Streamlit-based machine learning application for Walmart sales forecasting and inventory optimization.  
The app helps analyze sales trends, predict future demand, and generate inventory planning insights using ML models like LightGBM and XGBoost.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📂 Upload Data | Upload Walmart sales CSV files with session-based caching |
| 📊 EDA Dashboard | Interactive sales analysis, trends, and correlation heatmaps |
| 🔮 Forecasting | Predict sales for 1–12 weeks using ML models |
| 📦 Inventory Planning | Calculate safety stock, reorder points, and inventory requirements |

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas & NumPy
- Plotly / Matplotlib / Seaborn
- Scikit-learn
- LightGBM
- XGBoost

---

## 📥 Clone Repository

```bash
git clone https://github.com/codewitharu/Inventory_Forecast
cd Inventory_Forecast
```

---

## ⚙️ Installation & Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📄 Required CSV Columns

| Column | Type |
|---|---|
| Store | Integer |
| Date | DD-MM-YYYY |
| Weekly_Sales | Float |
| Holiday_Flag | 0 or 1 |
| Temperature | Float |
| Fuel_Price | Float |
| CPI | Float |
| Unemployment | Float |

---

## 📁 Project Structure

```text
walmart_forecast_app/
├── app.py
├── config.py
├── requirements.txt
│
├── backend/
│   ├── data_loader.py
│   ├── eda.py
│   ├── features.py
│   ├── model.py
│   └── inventory.py
│
└── ui/
    ├── upload_page.py
    ├── eda_page.py
    ├── forecast_page.py
    └── inventory_page.py
```

---

## 🎯 Project Goals

- Forecast Walmart store sales accurately
- Improve inventory planning efficiency
- Reduce stock shortages and overstocking
- Provide interactive retail analytics dashboards

---

## 📌 Future Improvements

- Real-time inventory alerts
- Deep learning-based forecasting
- Cloud deployment support
- Automated model retraining

---
