import pandas as pd
import numpy as np
from config import TARGET, EXOG_COLS


def engineer_features(df):
    df = df.copy().sort_values(['Store', 'Date']).reset_index(drop=True)

    # Calendar
    df['week']      = df['Date'].dt.isocalendar().week.astype(int)
    df['month']     = df['Date'].dt.month
    df['year']      = df['Date'].dt.year
    df['quarter']   = df['Date'].dt.quarter
    df['week_sin']  = np.sin(2 * np.pi * df['week']  / 52)
    df['week_cos']  = np.cos(2 * np.pi * df['week']  / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lags
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f'lag_{lag}'] = df.groupby('Store')[TARGET].shift(lag)

    # Rolling stats
    for w in [4, 8, 12, 26]:
        grp = df.groupby('Store')[TARGET]
        df[f'roll_mean_{w}'] = grp.shift(1).transform(lambda x: x.rolling(w).mean())
        df[f'roll_std_{w}']  = grp.shift(1).transform(lambda x: x.rolling(w).std())
        df[f'roll_max_{w}']  = grp.shift(1).transform(lambda x: x.rolling(w).max())
        df[f'roll_min_{w}']  = grp.shift(1).transform(lambda x: x.rolling(w).min())

    # Interactions
    df['temp_x_holiday'] = df['Temperature'] * df['Holiday_Flag']
    df['fuel_x_cpi']     = df['Fuel_Price']  * df['CPI']
    df['unemp_x_cpi']    = df['Unemployment'] * df['CPI']

    # Rolling exog
    for col in EXOG_COLS:
        df[f'{col}_roll4'] = df.groupby('Store')[col]\
                               .transform(lambda x: x.rolling(4).mean())
    return df


def forecast_exog(df, store_df, n_weeks):
    last_date    = store_df['Date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks, freq='W')
    future_df = pd.DataFrame({'Date': future_dates})

    # Holiday — extract from actual data
    holiday_weeks = df[df['Holiday_Flag'] == 1]['Date']\
                      .dt.isocalendar().week.astype(int).unique().tolist()
    future_df['Holiday_Flag'] = future_df['Date']\
        .dt.isocalendar().week.astype(int).isin(holiday_weeks).astype(int)

    # Temperature — same week historical avg
    week_temp = store_df.groupby(
        store_df['Date'].dt.isocalendar().week.astype(int))['Temperature'].mean()
    future_df['week_num']    = future_df['Date'].dt.isocalendar().week.astype(int)
    future_df['Temperature'] = future_df['week_num'].map(week_temp)\
                                    .fillna(store_df['Temperature'].mean())
    future_df.drop(columns='week_num', inplace=True)

    # Fuel & CPI — linear trend
    for col in ['Fuel_Price', 'CPI']:
        recent           = store_df[col].iloc[-52:].values
        x                = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        future_df[col]   = slope * np.arange(
            len(recent), len(recent) + n_weeks) + intercept

    # Unemployment — forward fill
    future_df['Unemployment'] = store_df['Unemployment'].iloc[-1]
    return future_df
