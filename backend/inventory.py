import pandas as pd
import numpy as np


def get_inventory_recommendations(forecast_df, best_model, safety_stock_weeks=2):
    """
    Inventory recommendations based on forecast.
    safety_stock_weeks: buffer weeks of stock to keep.
    """
    fc = forecast_df[forecast_df['Model'] == best_model].copy()

    recommendations = fc.groupby('Store').agg(
        Total_Forecasted_Sales=('Forecast', 'sum'),
        Avg_Weekly_Sales      =('Forecast', 'mean'),
        Max_Weekly_Sales      =('Forecast', 'max'),
        Min_Weekly_Sales      =('Forecast', 'min'),
        Forecast_Weeks        =('Week',     'count'),
    ).reset_index()

    # Safety stock = avg weekly × buffer weeks
    recommendations['Safety_Stock'] = (
        recommendations['Avg_Weekly_Sales'] * safety_stock_weeks).round(0)

    # Reorder point = max weekly (1 week lead time assumed)
    recommendations['Reorder_Point'] = (
        recommendations['Max_Weekly_Sales'] * 1).round(0)

    # Total inventory needed
    recommendations['Total_Inventory_Needed'] = (
        recommendations['Total_Forecasted_Sales'] +
        recommendations['Safety_Stock']).round(0)

    return recommendations
