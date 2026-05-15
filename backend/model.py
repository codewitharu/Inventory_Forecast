import pandas as pd
import numpy as np
import streamlit as st
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from backend.features import engineer_features, forecast_exog
from config import TARGET, EXOG_COLS


def smape(a, p):
    return 100 * np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-8))


def get_metrics(actual, pred):
    return {
        'MAE'  : round(mean_absolute_error(actual, pred), 2),
        'RMSE' : round(np.sqrt(mean_squared_error(actual, pred)), 2),
        'SMAPE': round(smape(np.array(actual), np.array(pred)), 2),
    }


def get_models(selected='Both'):
    models = {}
    if selected in ['LightGBM', 'Both']:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=500, learning_rate=0.04, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, random_state=42, verbose=-1)
    if selected in ['XGBoost', 'Both']:
        models['XGBoost'] = XGBRegressor(
            n_estimators=500, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, random_state=42, verbosity=0)
    return models


@st.cache_data
def train_and_forecast(df_hash, df, n_weeks, model_choice):
    """Train global model and forecast all stores."""

    feat_df      = engineer_features(df)
    feature_cols = [c for c in feat_df.columns if c not in [TARGET, 'Date']]
    clean_df     = feat_df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    X = clean_df[feature_cols]
    y = clean_df[TARGET]

    # Train/test split — last n_weeks per store
    test_rows, train_rows = [], []
    for store_id in df['Store'].unique():
        sd = clean_df[clean_df['Store'] == store_id].sort_values('Date')
        if len(sd) > n_weeks:
            test_rows.append(sd.tail(n_weeks))
            train_rows.append(sd.iloc[:-n_weeks])

    train_df = pd.concat(train_rows).sort_values('Date').reset_index(drop=True)
    test_df  = pd.concat(test_rows).sort_values('Date').reset_index(drop=True)

    X_tr = train_df[feature_cols]
    y_tr = train_df[TARGET]
    X_te = test_df[feature_cols]
    y_te = test_df[TARGET]

    # CV
    tscv   = TimeSeriesSplit(n_splits=5)
    models = get_models(model_choice)
    cv_res = {name: [] for name in models}

    for tr_idx, val_idx in tscv.split(X_tr):
        for name, model in models.items():
            model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = model.predict(X_tr.iloc[val_idx])
            cv_res[name].append(get_metrics(y_tr.iloc[val_idx].values, pred))

    cv_summary = {
        name: {k: round(np.mean([f[k] for f in folds]), 2)
               for k in ['MAE', 'RMSE', 'SMAPE']}
        for name, folds in cv_res.items()
    }

    # Test metrics
    test_metrics = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        test_metrics[name] = get_metrics(y_te.values, pred)

    # Retrain on full data
    for name, model in models.items():
        model.fit(X, y)

    best_model_name = min(cv_summary, key=lambda x: cv_summary[x]['SMAPE'])

    # Forecast all stores
    all_forecasts = []
    stores        = df['Store'].unique()

    for store_id in stores:
        store_df    = df[df['Store'] == store_id].copy().reset_index(drop=True)
        future_exog = forecast_exog(df, store_df, n_weeks)
        history     = df.copy()

        store_fc = {name: [] for name in models}

        for i in range(n_weeks):
            exog_row  = future_exog.iloc[i]
            next_date = exog_row['Date']
            new_row   = pd.DataFrame([{
                'Store': store_id, 'Date': next_date, TARGET: np.nan,
                **{col: exog_row[col] for col in EXOG_COLS}
            }])
            history   = pd.concat([history, new_row], ignore_index=True)
            feat_temp = engineer_features(history)
            last_feat = feat_temp[
                (feat_temp['Store'] == store_id) &
                (feat_temp['Date']  == next_date)
            ][feature_cols]

            preds = {}
            for name, model in models.items():
                preds[name] = model.predict(last_feat)[0]
                store_fc[name].append(preds[name])

            history.loc[
                (history['Store'] == store_id) &
                (history['Date']  == next_date), TARGET
            ] = preds[best_model_name]

        for name in models:
            vals = np.array(store_fc[name])
            for j, (date, val) in enumerate(zip(future_exog['Date'], vals)):
                all_forecasts.append({
                    'Store'   : store_id,
                    'Model'   : name,
                    'Date'    : date,
                    'Forecast': round(val, 2),
                    'Lower'   : round(val * 0.92, 2),
                    'Upper'   : round(val * 1.08, 2),
                    'Week'    : j + 1,
                })

    forecast_df = pd.DataFrame(all_forecasts)
    return forecast_df, cv_summary, test_metrics, best_model_name, models, feature_cols
