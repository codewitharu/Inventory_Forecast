import streamlit as st
from backend.data_loader import (load_and_validate, save_to_session,
                                  load_from_session, clear_session)


def show_upload_page():
    st.title("🛒 Walmart Inventory Forecast App")
    st.markdown("Upload your Walmart sales CSV to get started.")

    # Clear cache button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Data", type='secondary'):
            clear_session()
            st.success("Cache cleared!")
            st.rerun()

    # Check existing session
    existing_data = load_from_session()
    if existing_data is not None:
        st.success(
            f"✅ Data already loaded — "
            f"{existing_data['Store'].nunique()} stores, "
            f"{len(existing_data):,} rows"
        )
        st.dataframe(existing_data.head(10), use_container_width=True)
        return existing_data

    # Upload
    uploaded_file = st.file_uploader(
        "📂 Upload Walmart Sales CSV",
        type=['csv'],
        help="Required columns: Store, Date, Weekly_Sales, Holiday_Flag, "
             "Temperature, Fuel_Price, CPI, Unemployment"
    )

    if uploaded_file is not None:
        with st.spinner("Validating and loading data..."):
            df, error = load_and_validate(uploaded_file)

        if error:
            st.error(f"❌ {error}")
            return None

        save_to_session(df)
        st.success("✅ Data loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stores",     df['Store'].nunique())
        col2.metric("Total Rows", f"{len(df):,}")
        col3.metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        col4.metric("Weeks",      df['Date'].nunique())

        st.dataframe(df.head(10), use_container_width=True)
        return df

    else:
        st.info("👆 Please upload your CSV file to get started.")

        # Show required format
        st.markdown("### 📋 Required CSV Format")
        st.markdown("""
        | Column | Type | Description |
        |---|---|---|
        | Store | int | Store ID |
        | Date | date | Week date (DD-MM-YYYY) |
        | Weekly_Sales | float | Weekly sales amount |
        | Holiday_Flag | int | 1 = holiday week, 0 = normal |
        | Temperature | float | Average temperature |
        | Fuel_Price | float | Fuel price |
        | CPI | float | Consumer Price Index |
        | Unemployment | float | Unemployment rate |
        """)
        return None
