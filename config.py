TARGET        = 'Weekly_Sales'
EXOG_COLS     = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
REQUIRED_COLS = ['Store', 'Date', 'Weekly_Sales',
                 'Holiday_Flag', 'Temperature',
                 'Fuel_Price', 'CPI', 'Unemployment']

DEFAULT_FORECAST_WEEKS = 12
MAX_FORECAST_WEEKS     = 26
MIN_FORECAST_WEEKS     = 1

MODELS = ['LightGBM', 'XGBoost', 'Both']

CACHE_KEY_DATA     = 'uploaded_data'
CACHE_KEY_FORECAST = 'forecast_results'
CACHE_KEY_MODEL    = 'trained_models'
