import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from backend.model import train_and_forecast
from config import (DEFAULT_FORECAST_WEEKS, MAX_FORECAST_WEEKS,
                    MIN_FORECAST_WEEKS, MODELS, CACHE_KEY_FORECAST)


def show_forecast_page(df):
    st.title("🔮 Sales Forecast")

    # Settings
    st.subheader("⚙️ Forecast Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        n_weeks = st.slider(
            "Forecast Weeks",
            min_value=MIN_FORECAST_WEEKS,
            max_value=MAX_FORECAST_WEEKS,
            value=DEFAULT_FORECAST_WEEKS,
            step=1,
            help="Number of weeks to forecast ahead"
        )
    with col2:
        model_choice = st.selectbox("Select Model", MODELS)
    with col3:
        selected_store = st.selectbox(
            "Select Store", sorted(df['Store'].unique()))

    run_forecast = st.button("🚀 Run Forecast", type='primary')

    if run_forecast or CACHE_KEY_FORECAST in st.session_state:

        if run_forecast:
            with st.spinner("Training models and generating forecast... ⏳"):
                df_hash = pd.util.hash_pandas_object(df).sum()
                forecast_df, cv_summary, test_metrics, \
                    best_model_name, models, feature_cols = \
                    train_and_forecast(df_hash, df, n_weeks, model_choice)

            st.session_state[CACHE_KEY_FORECAST] = {
                'forecast_df'    : forecast_df,
                'cv_summary'     : cv_summary,
                'test_metrics'   : test_metrics,
                'best_model_name': best_model_name,
                'models'         : models,
                'feature_cols'   : feature_cols,
            }

        # Load from session
        cached          = st.session_state[CACHE_KEY_FORECAST]
        forecast_df     = cached['forecast_df']
        cv_summary      = cached['cv_summary']
        test_metrics    = cached['test_metrics']
        best_model_name = cached['best_model_name']
        models          = cached['models']
        feature_cols    = cached['feature_cols']

        st.success(f"✅ Best Model: **{best_model_name}**")

        # Metrics
        st.subheader("📊 Model Performance")
        tab1, tab2 = st.tabs(["CV Metrics", "Test Metrics"])

        with tab1:
            for name, m in cv_summary.items():
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{name} MAE",   f"${m['MAE']:,.0f}")
                col2.metric(f"{name} RMSE",  f"${m['RMSE']:,.0f}")
                col3.metric(f"{name} SMAPE", f"{m['SMAPE']}%")

        with tab2:
            for name, m in test_metrics.items():
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{name} MAE",   f"${m['MAE']:,.0f}")
                col2.metric(f"{name} RMSE",  f"${m['RMSE']:,.0f}")
                col3.metric(f"{name} SMAPE", f"{m['SMAPE']}%")

        st.markdown("---")

        # Forecast Chart
        st.subheader(f"📈 Store {selected_store} — {n_weeks} Week Forecast")

        store_hist = df[df['Store'] == selected_store]
        colors     = {'LightGBM': '#e74c3c', 'XGBoost': '#3498db'}

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(store_hist['Date'], store_hist['Weekly_Sales'],
                color='#2c3e50', linewidth=1.5, label='Actual', zorder=5)

        for model_name in forecast_df['Model'].unique():
            fc  = forecast_df[
                (forecast_df['Store'] == selected_store) &
                (forecast_df['Model'] == model_name)]
            col = colors.get(model_name, 'green')
            ax.plot(fc['Date'], fc['Forecast'],
                    color=col, linewidth=2, linestyle='--',
                    label=f'{model_name} Forecast', zorder=4)
            ax.fill_between(fc['Date'], fc['Lower'], fc['Upper'],
                            color=col, alpha=0.12)

        ax.axvline(store_hist['Date'].max(), color='gray',
                   linestyle=':', linewidth=1.2, label='Forecast Start')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weekly Sales ($)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # KPI Cards
        st.subheader("📌 Forecast Summary")
        fc_best  = forecast_df[
            (forecast_df['Store'] == selected_store) &
            (forecast_df['Model'] == best_model_name)]
        hist_avg = df[df['Store'] == selected_store]['Weekly_Sales'].mean()
        fc_avg   = fc_best['Forecast'].mean()
        fc_total = fc_best['Forecast'].sum()
        fc_chg   = ((fc_avg - hist_avg) / hist_avg) * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Store",             f"#{selected_store}")
        k2.metric("Total Forecast",    f"${fc_total:,.0f}")
        k3.metric("Avg Weekly",        f"${fc_avg:,.0f}")
        k4.metric("vs Historical Avg", f"{fc_chg:+.1f}%")

        st.markdown("---")

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        fig, axes = plt.subplots(1, len(models), figsize=(16, 6))
        if len(models) == 1:
            axes = [axes]
        for ax, (name, model) in zip(axes, models.items()):
            fi = pd.Series(model.feature_importances_,
                           index=feature_cols).sort_values().tail(15)
            fi.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f'{name} — Top 15 Features', fontweight='bold')
            ax.set_xlabel('Importance')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Download
        st.subheader("⬇️ Download Forecast")
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Forecast CSV",
            data=csv,
            file_name="walmart_forecast.csv",
            mime="text/csv"
        )
