import pandas as pd
import numpy as np
from config import TARGET


def get_summary_stats(df):
    total_revenue   = df[TARGET].sum()
    total_stores    = df['Store'].nunique()
    total_weeks     = df['Date'].nunique()
    avg_weekly      = df[TARGET].mean()
    holiday_revenue = df[df['Holiday_Flag'] == 1][TARGET].mean()
    normal_revenue  = df[df['Holiday_Flag'] == 0][TARGET].mean()
    return {
        'Total Revenue'    : f"${total_revenue:,.0f}",
        'Total Stores'     : total_stores,
        'Total Weeks'      : total_weeks,
        'Avg Weekly Sales' : f"${avg_weekly:,.0f}",
        'Holiday Sales Avg': f"${holiday_revenue:,.0f}",
        'Normal Sales Avg' : f"${normal_revenue:,.0f}",
    }


def get_monthly_revenue(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    return df.groupby('Month')[TARGET].sum().reset_index()


def get_store_revenue(df):
    return df.groupby('Store')[TARGET].sum()\
             .sort_values(ascending=False).reset_index()


def get_holiday_impact(df):
    return df.groupby('Holiday_Flag')[TARGET].mean().reset_index()


def get_top_stores(df, n=10):
    return df.groupby('Store')[TARGET].sum()\
             .sort_values(ascending=False).head(n).reset_index()


def get_weekly_trend(df, store_id):
    return df[df['Store'] == store_id][['Date', TARGET]].copy()


def get_correlation(df):
    cols = ['Weekly_Sales', 'Holiday_Flag',
            'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    return df[cols].corr()


def get_exog_trends(df):
    df = df.copy()
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    return df.groupby('Month')[['Temperature', 'Fuel_Price',
                                 'CPI', 'Unemployment']].mean().reset_index()


def get_weekly_sales_trend(df):
    """Get aggregated weekly sales trend across all stores"""
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Sales', 'Avg_Sales', 'Store_Count'])
    weekly = df.groupby('Date')[TARGET].agg(['sum', 'mean', 'count']).reset_index()
    weekly.columns = ['Date', 'Total_Sales', 'Avg_Sales', 'Store_Count']
    return weekly.sort_values('Date').reset_index(drop=True)


def identify_peak_weeks(df, n=3):
    """Identify top n peak weeks by sales"""
    if df.empty:
        return pd.DataFrame(columns=['Date', TARGET])
    weekly = df.groupby('Date')[TARGET].sum().reset_index()
    peaks = weekly.nlargest(n, TARGET).reset_index(drop=True)
    return peaks


def get_top_bottom_stores_by_period(df, holiday_flag=1, n=10):
    """Get top and bottom n stores during holiday or normal periods"""
    period_df = df[df['Holiday_Flag'] == holiday_flag]
    top_stores = period_df.groupby('Store')[TARGET].sum().sort_values(ascending=False).head(n).reset_index()
    bottom_stores = period_df.groupby('Store')[TARGET].sum().sort_values(ascending=True).head(n).reset_index()
    return top_stores, bottom_stores


def get_store_sales_distribution(df):
    """Get sales distribution by store"""
    if df.empty:
        return pd.DataFrame(columns=['Store', 'sum', 'mean', 'std'])
    return df.groupby('Store')[TARGET].agg(['sum', 'mean', 'std']).reset_index().sort_values('sum', ascending=False)
