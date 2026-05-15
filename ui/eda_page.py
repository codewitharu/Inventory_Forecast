import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from config import TARGET
from backend.eda import (get_summary_stats, get_monthly_revenue,
                          get_holiday_impact, get_top_stores,
                          get_weekly_trend, get_correlation,
                          get_exog_trends, get_weekly_sales_trend,
                          identify_peak_weeks, get_top_bottom_stores_by_period,
                          get_store_sales_distribution)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 9


def show_eda_page(df):
    st.title("📊 EDA Dashboard")

    # Interactive Filters Section
    st.subheader("🔍 Interactive Filters")
    with st.expander("📅 Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # Date Range Filter - Convert Timestamps to dates for slider
        with col1:
            min_date = df['Date'].min().date() if pd.api.types.is_datetime64_any_dtype(df['Date']) else df['Date'].min()
            max_date = df['Date'].max().date() if pd.api.types.is_datetime64_any_dtype(df['Date']) else df['Date'].max()
            
            date_range = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="date_range"
            )
            
            # Convert slider result back to datetime for filtering
            start_date = pd.Timestamp(date_range[0])
            end_date = pd.Timestamp(date_range[1])
            filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        
        # Store Filter
        # with col2:
        #     selected_stores = st.multiselect(
        #         "Select Stores (leave blank for all)",
        #         options=sorted(df['Store'].unique()),
        #         key="store_filter"
        #     )
        #     if selected_stores:
        #         filtered_df = filtered_df[filtered_df['Store'].isin(selected_stores)]
        
        # Holiday Filter Toggle
        with col3:
            filter_holiday = st.checkbox("Include Holiday Weeks Only", value=False)
            if filter_holiday:
                filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 1]
    
    st.markdown("---")

    # Summary KPIs (Dynamic based on filters)
    stats = get_summary_stats(filtered_df)
    cols  = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, v)

    st.markdown("---")

    # Monthly Revenue Trend with enhanced styling
    st.subheader("📈 Monthly Revenue Trend")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        monthly = get_monthly_revenue(filtered_df)
        
        if monthly.empty:
            st.warning("⚠️ No monthly data available for the selected period.")
        else:
            with st.expander("Show Monthly Revenue Chart", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    chart_type = st.radio(
                        "Chart Type",
                        options=["Line", "Area", "Bar"],
                        horizontal=True
                    )
                
                with col1:
                    fig, ax = plt.subplots(figsize=(14, 5))
                    
                    if chart_type == "Line":
                        ax.plot(monthly['Month'], monthly['Weekly_Sales'],
                                color='#2E86AB', linewidth=3, marker='o', markersize=6, label='Total Revenue')
                        ax.fill_between(range(len(monthly)), monthly['Weekly_Sales'],
                                        alpha=0.2, color='#2E86AB')
                    elif chart_type == "Area":
                        ax.fill_between(range(len(monthly)), monthly['Weekly_Sales'],
                                        alpha=0.4, color='#2E86AB')
                        ax.plot(monthly['Month'], monthly['Weekly_Sales'],
                                color='#2E86AB', linewidth=3, marker='o', markersize=6)
                    else:  # Bar
                        colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(monthly)))
                        ax.bar(range(len(monthly)), monthly['Weekly_Sales'],
                               color=colors_bar, edgecolor='white', linewidth=1.5)
                        ax.set_xticks(range(len(monthly)))
                    
                    ax.set_xlabel('Month', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Total Revenue ($)', fontsize=10, fontweight='bold')
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
                    ax.grid(alpha=0.4, linestyle='--')
                    if chart_type != "Bar":
                        plt.xticks(range(len(monthly)), monthly['Month'], rotation=45, fontsize=8)
                    else:
                        ax.set_xticklabels(monthly['Month'], rotation=45, fontsize=8)
                    ax.legend(loc='best')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    st.markdown("---")

    # Weekly Sales Trend with Top 3 Peaks & Holiday Periods
    st.subheader("📊 Weekly Sales Trend with Top Peaks & Holiday Periods")
    
    with st.expander("Configure Peak Detection", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_peaks = st.slider("Number of Peaks to Highlight", 1, 10, 3, key="peaks_slider")
        
        with col2:
            show_holidays = st.checkbox("Show Holiday Periods", value=True, key="show_holidays")
        
        with col3:
            smooth_data = st.checkbox("Smooth Data (7-day Moving Avg)", value=False, key="smooth_data")
    
    weekly_data = get_weekly_sales_trend(filtered_df)
    
    if filtered_df.empty or weekly_data.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
    else:
        # Apply smoothing if requested
        if smooth_data and len(weekly_data) >= 7:
            weekly_data_plot = weekly_data.copy()
            weekly_data_plot['Total_Sales'] = weekly_data_plot['Total_Sales'].rolling(window=7, center=True).mean()
        else:
            weekly_data_plot = weekly_data.copy()
        
        peak_weeks = identify_peak_weeks(filtered_df, n=num_peaks)
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot all weeks
        if not weekly_data_plot.empty:
            ax.plot(weekly_data_plot['Date'], weekly_data_plot['Total_Sales'],
                    color='#1B4965', linewidth=2, label='Weekly Sales', alpha=0.7)
            ax.fill_between(weekly_data_plot['Date'], weekly_data_plot['Total_Sales'],
                            alpha=0.15, color='#1B4965')
        
        # Highlight peak weeks
        if not peak_weeks.empty:
            for idx, row in peak_weeks.iterrows():
                ax.scatter(row['Date'], row[TARGET], color='#E63946', s=200, zorder=5, marker='*')
                ax.annotate(f"Peak {idx+1}\n${row[TARGET]:,.0f}", 
                           xy=(row['Date'], row[TARGET]),
                           xytext=(0, 15), textcoords='offset points',
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#E63946', alpha=0.7, edgecolor='none'),
                           color='white', fontweight='bold')
        
        # Highlight holiday weeks
        if show_holidays:
            holiday_weeks = filtered_df[filtered_df['Holiday_Flag'] == 1]['Date'].unique()
            for holiday in holiday_weeks:
                ax.axvline(x=holiday, color='#F77F00', alpha=0.3, linestyle='--', linewidth=1)
        
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Weekly Sales ($)', fontsize=10, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
        ax.grid(alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, fontsize=8)
        
        # Add custom legend
        legend_elements = [plt.Line2D([0], [0], color='#1B4965', linewidth=2, label='Weekly Sales')]
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#E63946', 
                                markersize=15, label=f'Top {num_peaks} Peak Weeks'))
        if show_holidays:
            legend_elements.append(plt.Line2D([0], [0], color='#F77F00', linewidth=2, linestyle='--', 
                                label='Holiday Periods'))
        
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        plt.title(f"Weekly Sales Trend with Top {num_peaks} Peaks{'& Holiday Periods' if show_holidays else ''}", 
                 fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Top vs Bottom 10 Store Sales During Peak Weeks
    st.subheader("🏆 Top vs Bottom Store Sales During Peak Weeks")
    
    with st.expander("Store Ranking Options", expanded=True):
        num_stores_display = st.slider("Number of Stores to Display", 5, 20, 10, step=1, key="stores_display")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        peak_weeks_data = identify_peak_weeks(filtered_df, n=num_peaks)
        
        if peak_weeks_data.empty:
            st.warning("⚠️ No peak weeks available for the selected data.")
        else:
            peak_weeks_dates = peak_weeks_data['Date'].values
            peak_df = filtered_df[filtered_df['Date'].isin(peak_weeks_dates)]
            
            if peak_df.empty:
                st.warning("⚠️ No data found for peak weeks.")
            else:
                top_peak = peak_df.groupby('Store')[TARGET].sum().nlargest(num_stores_display).reset_index()
                bottom_peak = peak_df.groupby('Store')[TARGET].sum().nsmallest(num_stores_display).reset_index()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Top stores
                if not top_peak.empty:
                    colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_peak)))
                    ax1.barh(top_peak['Store'].astype(str), top_peak[TARGET],
                            color=colors_top, edgecolor='white', linewidth=1.5)
                    ax1.set_xlabel('Total Sales During Peak Weeks ($)', fontsize=10, fontweight='bold')
                    ax1.set_title(f'Top {len(top_peak)} Stores', fontsize=11, fontweight='bold', color='#06A77D')
                    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
                    ax1.grid(axis='x', alpha=0.3)
                    ax1.invert_yaxis()
                
                # Bottom stores
                if not bottom_peak.empty:
                    colors_bottom = plt.cm.Reds(np.linspace(0.4, 0.9, len(bottom_peak)))
                    ax2.barh(bottom_peak['Store'].astype(str), bottom_peak[TARGET],
                            color=colors_bottom, edgecolor='white', linewidth=1.5)
                    ax2.set_xlabel('Total Sales During Peak Weeks ($)', fontsize=10, fontweight='bold')
                    ax2.set_title(f'Bottom {len(bottom_peak)} Stores', fontsize=11, fontweight='bold', color='#E63946')
                    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
                    ax2.grid(axis='x', alpha=0.3)
                    ax2.invert_yaxis()
                
                plt.suptitle('Store Performance During Peak Sales Weeks', fontsize=12, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    # Store Revenue + Holiday Impact (improved)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏪 Top Stores by Revenue")
        
        with st.expander("Store Ranking Options", expanded=True):
            num_top_stores = st.slider("Number of Top Stores to Show", 5, 20, 10, step=1, key="top_stores")
        
        if filtered_df.empty:
            st.warning("⚠️ No data available for the selected filters.")
        else:
            top_stores_data = filtered_df.groupby('Store')[TARGET].sum().sort_values(ascending=False).head(num_top_stores).reset_index()
            
            if not top_stores_data.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_stores_data)))
                ax.barh(top_stores_data['Store'].astype(str),
                        top_stores_data[TARGET],
                        color=colors, edgecolor='white', linewidth=1.5)
                ax.set_xlabel('Total Revenue ($)', fontsize=9, fontweight='bold')
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
                ax.grid(axis='x', alpha=0.3)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    with col2:
        st.subheader("🎉 Holiday vs Normal Sales Impact")
        
        with st.expander("Chart Options", expanded=True):
            show_percentage = st.checkbox("Show Percentage Difference", value=True, key="pct_diff")
        
        if filtered_df.empty:
            st.warning("⚠️ No data available for the selected filters.")
        else:
            holiday = get_holiday_impact(filtered_df)
            
            if holiday.empty:
                st.warning("⚠️ No holiday data available.")
            else:
                labels = ['Normal Weeks', 'Holiday Weeks'] if len(holiday) > 1 else [holiday['Holiday_Flag'].iloc[0].astype(str)]
                fig, ax = plt.subplots(figsize=(8, 6))
                colors_holiday = ['#2E86AB', '#F77F00'][:len(holiday)]
                bars = ax.bar(labels, holiday[TARGET],
                              color=colors_holiday, edgecolor='white', linewidth=2, width=0.6)
                ax.bar_label(bars, fmt='$%,.0f', padding=3, fontsize=10, fontweight='bold')
                ax.set_ylabel('Avg Weekly Sales ($)', fontsize=9, fontweight='bold')
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
                ax.grid(axis='y', alpha=0.3)
                
                # Add percentage difference
                if len(holiday) > 1 and show_percentage:
                    pct_diff = ((holiday[TARGET].iloc[1] - holiday[TARGET].iloc[0]) / holiday[TARGET].iloc[0] * 100)
                    ax.text(0.5, max(holiday[TARGET]) * 0.5, 
                           f'{pct_diff:+.1f}%\nIncrease',
                           ha='center', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    # Store Weekly Trend (improved)
    st.subheader("📉 Store Weekly Sales Trend")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        with st.expander("Store Selection", expanded=True):
            col1, col2 = st.columns(2)
            
            unique_stores = sorted(filtered_df['Store'].unique())
            
            with col1:
                store_id = st.selectbox("Select Store", unique_stores, key="store_select")
            
            with col2:
                apply_ma = st.checkbox("Apply 4-Week Moving Average", value=False, key="apply_ma")
        
        trend = get_weekly_trend(filtered_df, store_id)
        
        if trend.empty:
            st.warning(f"⚠️ No data available for Store {store_id}.")
        else:
            # Apply moving average if requested
            if apply_ma and len(trend) >= 4:
                trend_plot = trend.copy()
                trend_plot[TARGET] = trend_plot[TARGET].rolling(window=4, center=True).mean()
            else:
                trend_plot = trend.copy()
            
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(trend_plot['Date'], trend_plot[TARGET],
                    color='#06A77D', linewidth=2.5, label=f'Store {store_id} Sales')
            ax.fill_between(trend_plot['Date'], trend_plot[TARGET], alpha=0.2, color='#06A77D')
            
            # Add original trend as faded line if smoothing applied
            if apply_ma and len(trend) >= 4:
                ax.plot(trend['Date'], trend[TARGET], color='#06A77D', linewidth=1, 
                       alpha=0.3, linestyle='--', label='Original (Unsmoothed)')
            
            ax.set_xlabel('Date', fontsize=10, fontweight='bold')
            ax.set_ylabel('Weekly Sales ($)', fontsize=10, fontweight='bold')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
            ax.grid(alpha=0.3, linestyle='--')
            plt.xticks(rotation=45, fontsize=8)
            ax.legend(loc='best')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Sales Distribution by Store
    st.subheader("📦 Sales Distribution Across Stores")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        with st.expander("Distribution Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                dist_metric = st.radio("Metric", options=["Total Sales", "Avg Sales", "Std Dev"], key="dist_metric")
            
            with col2:
                max_sales = int(filtered_df.groupby('Store')[TARGET].sum().max()) if not filtered_df.empty else 0
                min_sales = st.slider("Minimum Sales Filter ($)", 
                                     min_value=0, 
                                     max_value=max_sales,
                                     step=max(1, max_sales//10) if max_sales > 0 else 100000,
                                     key="min_sales")
        
        store_dist = get_store_sales_distribution(filtered_df)
        
        if store_dist.empty:
            st.warning("⚠️ No sales data available.")
        else:
            # Filter based on minimum sales
            store_dist_filtered = store_dist[store_dist['sum'] >= min_sales]
            
            if store_dist_filtered.empty:
                st.warning(f"⚠️ No stores with sales ≥ ${min_sales:,.0f}.")
            else:
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Select metric to display
                if dist_metric == "Total Sales":
                    values = store_dist_filtered['sum']
                    ylabel = "Total Sales ($)"
                    formatter = lambda x, _: f'${x/1e6:.1f}M'
                elif dist_metric == "Avg Sales":
                    values = store_dist_filtered['mean']
                    ylabel = "Avg Weekly Sales ($)"
                    formatter = lambda x, _: f'${x/1e3:.0f}K'
                else:  # Std Dev
                    values = store_dist_filtered['std']
                    ylabel = "Sales Std Dev ($)"
                    formatter = lambda x, _: f'${x/1e3:.0f}K'
                
                ax.bar(range(len(store_dist_filtered)), values,
                      color=plt.cm.viridis(np.linspace(0.2, 0.9, len(store_dist_filtered))),
                      edgecolor='white', linewidth=0.5)
                ax.set_xlabel('Store ID', fontsize=10, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
                ax.set_xticks(range(0, len(store_dist_filtered), max(1, len(store_dist_filtered)//20)))
                ax.set_xticklabels(store_dist_filtered['Store'].iloc[::max(1, len(store_dist_filtered)//20)].astype(int), rotation=45)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
                ax.grid(axis='y', alpha=0.3)
                
                # Add info box
                st.info(f"📊 Showing {len(store_dist_filtered)} out of {len(store_dist)} stores with sales ≥ ${min_sales:,.0f}")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    # Correlation Heatmap (improved)
    st.subheader("🔥 Correlation Heatmap")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        with st.expander("Heatmap Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                heatmap_style = st.selectbox("Color Scheme", options=["RdYlGn", "coolwarm", "viridis", "Blues"], key="heatmap_style")
            
            with col2:
                show_values = st.checkbox("Show Correlation Values", value=True, key="show_corr_values")
        
        corr = get_correlation(filtered_df)
        
        if corr.empty:
            st.warning("⚠️ Unable to compute correlations.")
        else:
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr, annot=show_values, fmt='.2f', cmap=heatmap_style,
                        linewidths=0.5, linecolor='white', ax=ax,
                        cbar_kws={'label': 'Correlation'}, center=0)
            ax.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Exogenous Trends (improved)
    st.subheader("🌡️ Exogenous Feature Trends")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        with st.expander("Feature Selection", expanded=True):
            exog_features = st.multiselect(
                "Select Features to Display",
                options=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],
                default=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],
                key="exog_features"
            )
        
        if exog_features:
            exog = get_exog_trends(filtered_df)
            
            if exog.empty:
                st.warning("⚠️ No exogenous data available.")
            else:
                # Dynamically create subplots based on selected features
                num_features = len(exog_features)
                cols_layout = 2
                rows_layout = (num_features + 1) // cols_layout
                
                fig, axes = plt.subplots(rows_layout, cols_layout, figsize=(14, 4 * rows_layout))
                if rows_layout == 1:
                    axes = axes if cols_layout == 1 else axes.flatten()
                else:
                    axes = axes.flatten()
                
                colors_exog = ['#E63946', '#457B9D', '#1D3557', '#F77F00']
                
                for idx, (ax, col, color) in enumerate(zip(axes, exog_features, colors_exog)):
                    if col in exog.columns and len(exog) > 0:
                        ax.plot(range(len(exog)), exog[col], color=color,
                                linewidth=2.5, marker='o', markersize=4, label=col)
                        ax.fill_between(range(len(exog)), exog[col], alpha=0.15, color=color)
                        ax.set_title(col, fontweight='bold', fontsize=10)
                        ax.set_xlabel('Month', fontsize=9)
                        ax.set_ylabel('Value', fontsize=9)
                        ax.grid(alpha=0.3, linestyle='--')
                        ax.set_xticks(range(0, len(exog), max(1, len(exog)//6)))
                        ax.set_xticklabels(exog['Month'].iloc[::max(1, len(exog)//6)], rotation=45, fontsize=7)
                
                # Hide unused subplots
                for idx in range(len(exog_features), len(axes)):
                    axes[idx].axis('off')
                
                plt.suptitle('Exogenous Feature Trends Over Time',
                             fontsize=13, fontweight='bold', y=0.995)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("⚠️ Please select at least one feature to display")

