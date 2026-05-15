import streamlit as st
from ui.upload_page    import show_upload_page
from ui.eda_page       import show_eda_page
from ui.forecast_page  import show_forecast_page
from ui.inventory_page import show_inventory_page
from backend.data_loader import load_from_session

st.set_page_config(
    page_title="Walmart Inventory Forecast",
    page_icon="🛒",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🛒 Walmart Forecast")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📂 Upload Data",
        "📊 EDA Dashboard",
        "🔮 Forecast",
        "📦 Inventory",
    ])
    st.markdown("---")
    st.markdown("**App Info**")
    st.markdown("- Models: LightGBM / XGBoost")
    st.markdown("- Global model across all stores")
    st.markdown("- Configurable forecast weeks (1–26)")
    st.markdown("- Session memory with clear option")

# Page Router
if page == "📂 Upload Data":
    show_upload_page()
else:
    df = load_from_session()
    if df is None:
        st.warning("⚠️ Please upload your data first on the Upload page.")
    else:
        if page == "📊 EDA Dashboard":
            show_eda_page(df)
        elif page == "🔮 Forecast":
            show_forecast_page(df)
        elif page == "📦 Inventory":
            show_inventory_page(df)
