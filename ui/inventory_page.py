import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from backend.inventory import get_inventory_recommendations
from config import CACHE_KEY_FORECAST


def show_inventory_page(df):
    st.title("📦 Inventory Recommendations")

    if CACHE_KEY_FORECAST not in st.session_state:
        st.warning("⚠️ Please run the forecast first on the Forecast page.")
        return

    cached          = st.session_state[CACHE_KEY_FORECAST]
    forecast_df     = cached['forecast_df']
    best_model_name = cached['best_model_name']

    # Settings
    st.subheader("⚙️ Inventory Settings")
    safety_weeks = st.slider(
        "Safety Stock Buffer (weeks)",
        min_value=1, max_value=4, value=2,
        help="Extra weeks of stock to keep as buffer"
    )

    recs = get_inventory_recommendations(
        forecast_df, best_model_name, safety_weeks)

    # Summary KPIs
    st.subheader("📌 Overall Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stores",           recs['Store'].nunique())
    col2.metric("Total Inventory Needed", f"${recs['Total_Inventory_Needed'].sum():,.0f}")
    col3.metric("Avg Safety Stock/Store", f"${recs['Safety_Stock'].mean():,.0f}")

    st.markdown("---")

    # Store wise table
    st.subheader("🏪 Store-wise Inventory Plan")
    st.dataframe(
        recs.style.format({
            'Total_Forecasted_Sales' : '${:,.0f}',
            'Avg_Weekly_Sales'       : '${:,.0f}',
            'Max_Weekly_Sales'       : '${:,.0f}',
            'Min_Weekly_Sales'       : '${:,.0f}',
            'Safety_Stock'           : '${:,.0f}',
            'Reorder_Point'          : '${:,.0f}',
            'Total_Inventory_Needed' : '${:,.0f}',
        }),
        use_container_width=True
    )

    st.markdown("---")

    # Top stores chart
    st.subheader("📊 Top 10 Stores by Inventory Needed")
    top10   = recs.nlargest(10, 'Total_Inventory_Needed')
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(top10['Store'].astype(str),
                   top10['Total_Inventory_Needed'],
                   color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='${:,.0f}', padding=3, fontsize=8)
    ax.set_xlabel('Total Inventory Needed ($)')
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Top 10 Stores — Total Inventory Needed',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Download
    csv = recs.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Inventory Plan CSV",
        data=csv,
        file_name="inventory_plan.csv",
        mime="text/csv"
    )
