import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- Configuration and Styling ---
def set_page_config_and_styles():
    """Sets Streamlit page configuration and injects custom CSS for styling."""
    st.set_page_config(page_title="📊 Auto Visualization Dashboard", layout="wide")
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white; /* Ensure headers are visible */
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            background-color: white;
            color: black;
        }
        .stSelectbox, .stMultiSelect {
            color: black;
        }
        div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            color: white;
        }
        div[data-testid="stAlert"] p {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- File Loading ---
@st.cache_data(show_spinner=False)
def load_uploaded_file(uploaded_file):
    """Loads a DataFrame from an uploaded file with caching."""
    try:
        if uploaded_file.type == "text/csv":
            return pd.read_csv(uploaded_file)
        elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            return pd.read_excel(uploaded_file)
        elif uploaded_file.type == "application/json":
            return pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type! Please upload CSV, Excel, or JSON.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- Data Preprocessing & Cleaning ---
def detect_column_types(df):
    """Detects and categorizes column types (numeric, categorical, datetime, geo)."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    geo_cols = []

    for col in df.columns:
        col_lower = col.lower()
        # Datetime detection
        if any(x in col_lower for x in ['date', 'time', 'timestamp']):
            try:
                temp_series = pd.to_datetime(df[col], errors='coerce')
                if temp_series.count() / len(df) > 0.5:
                    df[col] = temp_series
                    if col not in datetime_cols:
                        datetime_cols.append(col)
            except Exception:
                pass

        # Geo-column detection
        if any(x in col_lower for x in ['lat', 'lon', 'latitude', 'longitude', 'region', 'country', 'state', 'city', 'location']):
            if col not in geo_cols:
                geo_cols.append(col)

    # Remove datetime overlap from others.
    categorical_cols = [c for c in categorical_cols if c not in datetime_cols]
    numeric_cols = [c for c in numeric_cols if c not in datetime_cols]

    return numeric_cols, categorical_cols, datetime_cols, geo_cols

@st.cache_data(show_spinner=False)
def clean_data(df_input):
    """
    Performs a series of data cleaning steps on a DataFrame.
    Uses a copy so as not to mutate the original.
    """
    df = df_input.copy()
    cleaning_log = []
    initial_shape = df.shape

    # 1. Drop columns with >95% missing
    null_thresh = int(0.95 * len(df))
    cols_to_drop_null = df.columns[df.isnull().sum() > null_thresh].tolist()
    if cols_to_drop_null:
        df = df.drop(columns=cols_to_drop_null)
        cleaning_log.append(f"Dropped columns with >95% missing values: {', '.join(cols_to_drop_null)}")

    # 2. Drop constant columns
    constant_cols = df.columns[df.nunique() <= 1].tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        cleaning_log.append(f"Dropped constant columns: {', '.join(constant_cols)}")

    # 3. Drop duplicate rows and rows entirely null
    rows_before_dedupe = df.shape[0]
    df = df.drop_duplicates()
    if rows_before_dedupe > df.shape[0]:
        cleaning_log.append(f"Removed {rows_before_dedupe - df.shape[0]} duplicate rows.")
    rows_before_dropna_all = df.shape[0]
    df = df.dropna(how='all')
    if rows_before_dropna_all > df.shape[0]:
        cleaning_log.append(f"Removed {rows_before_dropna_all - df.shape[0]} rows that were entirely null.")

    # 4. Remove special characters from object columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if df[col].notna().any() and df[col].dtype == 'object':
            original_col_str = df[col].astype(str).copy()
            df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\w\s.,;:?!@#%&()/\-\+]', '', x))
            if not original_col_str.equals(df[col]):
                cleaning_log.append(f"Removed special characters from column '{col}'.")

    # 5. Convert numeric-like strings to numbers
    for col in df.select_dtypes(include='object').columns:
        if df[col].notna().any():
            try:
                converted_series = pd.to_numeric(df[col], errors='coerce')
                if converted_series.notna().sum() / df[col].notna().sum() > 0.8 and pd.api.types.is_numeric_dtype(converted_series):
                    df[col] = converted_series
                    cleaning_log.append(f"Converted column '{col}' to numeric.")
            except Exception:
                pass

    # 6. Fill missing numeric values with median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            cleaning_log.append(f"Filled missing numeric values in '{col}' with median ({median_val:.2f}).")

    # 7. Fill missing categorical values with mode or 'unknown'
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else 'unknown'
            df[col] = df[col].fillna(fill_value)
            cleaning_log.append(f"Filled missing categorical values in '{col}' with mode ('{fill_value}').")

    # 8. Outlier removal using IQR (for numeric columns)
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty and len(outliers) < len(df) * 0.5:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            cleaning_log.append(f"Removed {len(outliers)} outliers from '{col}' using IQR method.")

    final_shape = df.shape
    if initial_shape != final_shape or cleaning_log:
        st.success(f"Data cleaning completed! Original shape: {initial_shape}, Cleaned shape: {final_shape}")
        for log_entry in cleaning_log:
            st.info(f"- {log_entry}")
    else:
        st.info("No significant changes after cleaning (or data was already clean).")

    return df

# --- Dashboard Components ---
def generate_kpis(df, numeric_cols):
    st.subheader("📊 Key Performance Indicators (KPIs)")
    if not numeric_cols:
        st.info("No numeric columns available to generate KPIs.")
        return
    cols = st.columns(min(len(numeric_cols), 4))
    for i, col in enumerate(numeric_cols):
        with cols[i % 4]:
            st.metric(label=col, value=round(df[col].mean(), 2), delta=round(df[col].std(), 2))

def generate_filters(df, categorical_cols):
    filters = {}
    st.sidebar.subheader("⚙️ Filters")
    if not categorical_cols:
        st.sidebar.info("No categorical columns available for filtering.")
        return filters
    for col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        options = [str(x) if isinstance(x, (list, dict)) else x for x in options]
        selected = st.sidebar.multiselect(f"Filter by **{col}**", options, default=options, key=f"filter_{col}")
        filters[col] = selected
    return filters

def apply_filters(df, filters):
    df_filtered = df.copy()
    for col, values in filters.items():
        if values:
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                try:
                    numeric_values = [pd.to_numeric(v, errors='coerce') for v in values]
                    numeric_values = [v for v in numeric_values if pd.notna(v)]
                    if numeric_values:
                        df_filtered = df_filtered[df_filtered[col].isin(numeric_values)]
                except Exception:
                    pass
            else:
                df_filtered = df_filtered[df_filtered[col].astype(str).isin([str(v) for v in values])]
    return df_filtered

# --- Univariate Analysis ---
def get_chart_options_univariate(dtype, nunique=None):
    if dtype in ['object', 'category']:
        if nunique is None or nunique <= 30:
            return ['Bar Chart', 'Pie Chart', 'Donut Chart', 'Count Plot']
        else:
            return ['Bar Chart', 'Count Plot']
    elif np.issubdtype(dtype, np.number):
        return ['Histogram', 'Box Plot', 'Violin Plot', 'Line Chart (by index)']
    elif np.issubdtype(dtype, np.datetime64):
        return ['Line Chart', 'Area Chart']
    else:
        return []

def plot_univariate_custom(df, numeric_cols, categorical_cols, datetime_cols):
    st.subheader("📌 Univariate Analysis")

    # Categorical Columns
    if categorical_cols:
        st.markdown("##### Categorical Columns")
        for col in categorical_cols:
            st.markdown(f"###### Column: **{col}**")
            nunique = df[col].nunique()
            options = get_chart_options_univariate(df[col].dtype, nunique)
            if not options:
                st.info(f"No suitable univariate chart options for '{col}'.")
                continue
            chart_type = st.selectbox(f"Select chart type for '{col}'", options, key=f"{col}_cat_uni")
            count_df = df[col].value_counts().reset_index()
            count_df.columns = [col, 'count']

            if chart_type == 'Bar Chart':
                fig = px.bar(count_df, x=col, y='count', title=f"Bar Chart: {col} Value Counts", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Pie Chart':
                fig = px.pie(count_df, names=col, values='count', title=f"Pie Chart: {col} Proportions", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Donut Chart':
                fig = px.pie(count_df, names=col, values='count', title=f"Donut Chart: {col} Proportions", hole=0.4, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Count Plot':
                fig = px.histogram(df, x=col, title=f"Count Plot: {col} Frequencies", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns detected for univariate analysis.")

    st.markdown("---")
    # Numeric Columns
    if numeric_cols:
        st.markdown("##### Numeric Columns")
        for col in numeric_cols:
            st.markdown(f"###### Column: **{col}**")
            options = get_chart_options_univariate(df[col].dtype)
            if not options:
                st.info(f"No suitable univariate chart options for '{col}'.")
                continue
            chart_type = st.selectbox(f"Select chart type for '{col}'", options, key=f"{col}_num_uni")
            if chart_type == 'Histogram':
                fig = px.histogram(df, x=col, title=f"Histogram: {col} Distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Box Plot':
                fig = px.box(df, y=col, title=f"Box Plot: {col} Distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Violin Plot':
                fig = px.violin(df, y=col, title=f"Violin Plot: {col} Distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Line Chart (by index)':
                fig = px.line(df, y=col, title=f"Line Chart: {col} (by index)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for univariate analysis.")

    st.markdown("---")
    # Datetime Columns
    if datetime_cols:
        st.markdown("##### Datetime Columns")
        for col in datetime_cols:
            st.markdown(f"###### Column: **{col}**")
            options = get_chart_options_univariate(df[col].dtype)
            if not options:
                st.info(f"No suitable univariate chart options for '{col}'.")
                continue
            chart_type = st.selectbox(f"Select chart type for '{col}'", options, key=f"{col}_date_uni")
            if numeric_cols:
                selected_num_col = st.selectbox(f"Select a numeric column to plot against '{col}'", numeric_cols, key=f"{col}_num_select_uni")
                if chart_type == 'Line Chart':
                    fig = px.line(df.sort_values(col), x=col, y=selected_num_col, title=f"Time Series: {selected_num_col} over {col}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type == 'Area Chart':
                    fig = px.area(df.sort_values(col), x=col, y=selected_num_col, title=f"Area Chart: {selected_num_col} over {col}", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No numeric columns available to plot against '{col}' for time series analysis.")
    else:
        st.info("No datetime columns detected for univariate analysis.")

# --- Bivariate Analysis ---
def plot_charts_bivariate(df, numeric_cols, categorical_cols, datetime_cols, geo_cols):
    charts_bivariate = []

    st.subheader("🔗 Bivariate Analysis")

    # Numeric vs Numeric
    st.markdown("##### Numeric vs. Numeric")
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            num_pair_x = st.selectbox("Select X-axis for Numeric vs Numeric", numeric_cols, key="num_x_bivar")
        with col2:
            num_pair_y = st.selectbox("Select Y-axis for Numeric vs Numeric", [col for col in numeric_cols if col != num_pair_x], key="num_y_bivar")
        if num_pair_x and num_pair_y:
            bivar_num_options = ['Scatter Plot', 'Hexbin Plot', '2D Density Heatmap', 'Bubble Chart']
            selected_bivar_num_chart = st.selectbox(f"Choose chart for {num_pair_x} vs {num_pair_y}", bivar_num_options, key="num_num_chart_type")
            if selected_bivar_num_chart == 'Scatter Plot':
                fig = px.scatter(df, x=num_pair_x, y=num_pair_y, title=f"Scatter Plot: {num_pair_x} vs {num_pair_y}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_num_chart == 'Hexbin Plot':
                fig = px.density_heatmap(df, x=num_pair_x, y=num_pair_y, marginal_x="histogram", marginal_y="histogram", title=f"Hexbin Plot: {num_pair_x} vs {num_pair_y}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_num_chart == '2D Density Heatmap':
                fig = go.Figure(go.Histogram2dContour(x=df[num_pair_x], y=df[num_pair_y], colorscale='Viridis'))
                fig.update_layout(title_text=f"2D Density Heatmap: {num_pair_x} vs {num_pair_y}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_num_chart == 'Bubble Chart':
                if len(numeric_cols) >= 3:
                    bubble_cols = [col for col in numeric_cols if col not in [num_pair_x, num_pair_y]]
                    if bubble_cols:
                        size_col = st.selectbox("Select Numeric Column for Bubble Size", bubble_cols, key="bubble_size_col")
                        if size_col:
                            fig = px.scatter(df, x=num_pair_x, y=num_pair_y, size=size_col,
                                             color=size_col, title=f"Bubble Chart: {num_pair_x} vs {num_pair_y} (Size by {size_col})",
                                             size_max=60, template="plotly_white")
                            charts_bivariate.append(fig)
                    else:
                        st.info("Need at least three numeric columns for a Bubble Chart (X, Y, and Size).")
                else:
                    st.info("Need at least three numeric columns for a Bubble Chart (X, Y, and Size).")
    else:
        st.info("Need at least two numeric columns for Numeric vs. Numeric analysis.")

    # Categorical vs Numeric
    st.markdown("---")
    st.markdown("##### Categorical vs. Numeric")
    if categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox("Select Categorical Column", categorical_cols, key="cat_bivar")
        with col2:
            num_col = st.selectbox("Select Numeric Column", numeric_cols, key="num_bivar")
        if cat_col and num_col:
            bivar_cat_num_options = ['Box Plot', 'Violin Plot', 'Grouped Bar Chart (Mean)']
            selected_bivar_cat_num_chart = st.selectbox(f"Choose chart for {cat_col} vs {num_col}", bivar_cat_num_options, key="cat_num_chart_type")
            if selected_bivar_cat_num_chart == 'Box Plot':
                fig = px.box(df, x=cat_col, y=num_col, title=f"Box Plot: {num_col} by {cat_col}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_cat_num_chart == 'Violin Plot':
                fig = px.violin(df, x=cat_col, y=num_col, title=f"Violin Plot: {num_col} by {cat_col}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_cat_num_chart == 'Grouped Bar Chart (Mean)':
                agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
                fig = px.bar(agg_df, x=cat_col, y=num_col, title=f"Mean of {num_col} by {cat_col}", template="plotly_white")
                charts_bivariate.append(fig)
    else:
        st.info("Need at least one categorical and one numeric column for Categorical vs. Numeric analysis.")

    # Categorical vs Categorical & Hierarchical
    st.markdown("---")
    st.markdown("##### Categorical vs. Categorical & Hierarchical")
    if len(categorical_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            cat_pair_x = st.selectbox("Select first Categorical Column", categorical_cols, key="cat_x_bivar")
        with col2:
            cat_pair_y = st.selectbox("Select second Categorical Column", [col for col in categorical_cols if col != cat_pair_x], key="cat_y_bivar")
        if cat_pair_x and cat_pair_y:
            bivar_cat_options = ['Grouped Bar Chart', 'Heatmap (Crosstab)', 'Crosstab Table', 'Tree Map']
            selected_bivar_cat_chart = st.selectbox(f"Choose chart for {cat_pair_x} vs {cat_pair_y}", bivar_cat_options, key="cat_cat_chart_type")
            if selected_bivar_cat_chart == 'Grouped Bar Chart':
                counts = df.groupby([cat_pair_x, cat_pair_y]).size().reset_index(name='count')
                fig = px.bar(counts, x=cat_pair_x, y='count', color=cat_pair_y, barmode='group', title=f"Grouped Bar Chart: {cat_pair_x} vs {cat_pair_y}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_cat_chart == 'Heatmap (Crosstab)':
                ct = pd.crosstab(df[cat_pair_x], df[cat_pair_y])
                fig = px.imshow(ct, text_auto=True, color_continuous_scale='Viridis', title=f"Heatmap: Counts of {cat_pair_x} vs {cat_pair_y}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_bivar_cat_chart == 'Crosstab Table':
                ct = pd.crosstab(df[cat_pair_x], df[cat_pair_y])
                charts_bivariate.append(("table", ct))
            elif selected_bivar_cat_chart == 'Tree Map':
                if numeric_cols:
                    treemap_value_col = st.selectbox("Select Numeric Column for Tree Map Size", numeric_cols, key="treemap_value_col")
                    if treemap_value_col:
                        path_cols_options = [col for col in categorical_cols if col != treemap_value_col]
                        path_cols = st.multiselect("Select Categorical Columns for Tree Map Hierarchy (order matters!)",
                                                   path_cols_options,
                                                   default=[cat_pair_x, cat_pair_y] if cat_pair_x in path_cols_options and cat_pair_y in path_cols_options else [],
                                                   key="treemap_path_cols")
                        if path_cols:
                            fig = px.treemap(df, path=path_cols, values=treemap_value_col, title=f"Tree Map: {treemap_value_col} by {', '.join(path_cols)}", template="plotly_white")
                            charts_bivariate.append(fig)
                        else:
                            st.info("Please select at least one categorical column for the Tree Map hierarchy.")
                    else:
                        st.info("Please select a numeric column for the Tree Map size/value.")
                else:
                    st.info("No numeric columns available to create a Tree Map (requires a value column).")
    else:
        st.info("Need at least two categorical columns for Categorical vs. Categorical analysis.")

    # Time Series Analysis (Datetime vs. Numeric)
    st.markdown("---")
    st.markdown("##### Time Series Analysis (Datetime vs. Numeric)")
    if datetime_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select Datetime Column for Time Series", datetime_cols, key="date_bivar")
        with col2:
            num_col_ts = st.selectbox("Select Numeric Column for Time Series", numeric_cols, key="num_ts_bivar")
        if date_col and num_col_ts:
            ts_options = ['Line Chart', 'Area Chart']
            selected_ts_chart = st.selectbox(f"Choose chart for {date_col} vs {num_col_ts}", ts_options, key="ts_chart_type")
            if selected_ts_chart == 'Line Chart':
                fig = px.line(df.sort_values(date_col), x=date_col, y=num_col_ts, title=f"Time Series: {num_col_ts} over {date_col}", template="plotly_white")
                charts_bivariate.append(fig)
            elif selected_ts_chart == 'Area Chart':
                fig = px.area(df.sort_values(date_col), x=date_col, y=num_col_ts, title=f"Area Chart: {num_col_ts} over {date_col}", template="plotly_white")
                charts_bivariate.append(fig)
    else:
        st.info("No datetime columns detected for time series analysis.")

    # --- Geospatial Analysis ---
    st.markdown("---")
    st.markdown("##### Geospatial Analysis")
    
    # New: let the user choose the map type
    map_type = st.selectbox("Select Map Type", ["Scatter Map", "Choropleth Map"], key="map_type_selector")
    lat_found = next((c for c in geo_cols if 'lat' in c.lower()), None)
    lon_found = next((c for c in geo_cols if 'lon' in c.lower()), None)

    if map_type == "Scatter Map":
        if lat_found and lon_found:
            st.write(f"Scatter Map using Latitude: *{lat_found}* and Longitude: *{lon_found}*")
            fig = px.scatter_mapbox(df, lat=lat_found, lon=lon_found, zoom=3, height=400, title="Map: Locations")
            fig.update_layout(mapbox_style="open-street-map")
            charts_bivariate.append(fig)
            # Option to color the map points by another column
            if numeric_cols or categorical_cols:
                all_colorable_cols = numeric_cols + categorical_cols
                color_by_col_map_check = st.checkbox("Color map points by another column?", key="color_map_check")
                if color_by_col_map_check and all_colorable_cols:
                    color_col_map = st.selectbox("Select column to color map by", all_colorable_cols, key="color_map_col")
                    if color_col_map:
                        if pd.api.types.is_numeric_dtype(df[color_col_map]):
                            st.info(f"Coloring map points by numeric column: '{color_col_map}'.")
                            fig_colored = px.scatter_mapbox(df, lat=lat_found, lon=lon_found, color=color_col_map,
                                                            zoom=3, height=400, title=f"Map: Locations colored by {color_col_map}",
                                                            color_continuous_scale=px.colors.sequential.Viridis)
                            fig_colored.update_layout(mapbox_style="open-street-map")
                            charts_bivariate.append(fig_colored)
                        elif pd.api.types.is_categorical_dtype(df[color_col_map]) or df[color_col_map].dtype == 'object':
                            st.info(f"Coloring map points by categorical column: '{color_col_map}'.")
                            if df[color_col_map].nunique() > 50:
                                st.warning(f"'{color_col_map}' has too many unique values ({df[color_col_map].nunique()}) for effective coloring. Consider a different column.")
                            fig_colored = px.scatter_mapbox(df, lat=lat_found, lon=lon_found, color=color_col_map,
                                                            zoom=3, height=400, title=f"Map: Locations colored by {color_col_map}",
                                                            color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig_colored.update_layout(mapbox_style="open-street-map")
                            charts_bivariate.append(fig_colored)
                        else:
                            st.warning(f"Selected column '{color_col_map}' is not suitable for coloring (must be numeric or categorical).")
                elif color_by_col_map_check and not all_colorable_cols:
                    st.info("No other suitable columns (numeric or categorical) to color the map points by.")
        else:
            st.info("Latitude and Longitude columns not found. Please ensure your dataset has valid lat and long fields for a Scatter Map.")
    
    elif map_type == "Choropleth Map":
        if geo_cols and numeric_cols:
            region_col_candidates = [c for c in geo_cols if c not in [lat_found, lon_found]]
            if region_col_candidates:
                st.write("Choropleth Map: Aggregating data by region.")
                region_col = st.selectbox("Select a region/country column for Choropleth", region_col_candidates, key="region_choropleth")
                if region_col:
                    num_col_choropleth = st.selectbox("Select numeric column for Choropleth color", numeric_cols, key="num_choropleth_col")
                    if num_col_choropleth:
                        st.write(f"Choropleth Map: *{num_col_choropleth}* by *{region_col}*")
                        choropleth_df = df.groupby(region_col)[num_col_choropleth].mean().reset_index()
                        location_mode = st.selectbox("Select Location Mode for Choropleth",
                                                     ["country names", "USA-states", "ISO-3"],
                                                     key="loc_mode_choropleth")
                        if pd.api.types.is_numeric_dtype(choropleth_df[region_col]):
                            st.info("Warning: Region column is numeric. Ensure locationmode matches your numeric region codes (e.g., ISO-3).")
                        fig = px.choropleth(choropleth_df,
                                            locations=region_col,
                                            color=num_col_choropleth,
                                            locationmode=location_mode,
                                            color_continuous_scale="Viridis",
                                            title=f"{num_col_choropleth} by {region_col}")
                        charts_bivariate.append(fig)
                    else:
                        st.info("No numeric column selected for choropleth coloring.")
            else:
                st.info("No suitable region/country column found for Choropleth Map.")
        else:
            st.info("Geospatial columns are insufficient to create a Choropleth Map.")

    return charts_bivariate

# --- Revised Charts Grid Display ---
def display_charts_grid(charts):
    """
    Displays a list of Plotly charts or Pandas DataFrames (for crosstabs) in a two-column grid.
    Each chart is given a unique key.
    """
    if not charts:
        return
    num_cols = 2
    columns = st.columns(num_cols)
    for idx, chart in enumerate(charts):
        col = columns[idx % num_cols]
        with col:
            if isinstance(chart, tuple) and chart[0] == "table":
                st.dataframe(chart[1])
            else:
                st.plotly_chart(chart, use_container_width=True, key=f"chart_{idx}")

# --- Dashboard Logic Wrapper ---
def generate_dashboard(df):
    """Orchestrates the display of KPIs, filters, and charts."""
    numeric_cols, categorical_cols, datetime_cols, geo_cols = detect_column_types(df)
    generate_kpis(df, numeric_cols)
    filters = generate_filters(df, categorical_cols)
    df_filtered = apply_filters(df, filters)

    if df_filtered.empty:
        st.warning("No data remains after applying filters. Please adjust your filter selections.")
        return

    plot_univariate_custom(df_filtered, numeric_cols, categorical_cols, datetime_cols)
    st.markdown("---")
    charts_bivariate = plot_charts_bivariate(df_filtered, numeric_cols, categorical_cols, datetime_cols, geo_cols)
    display_charts_grid(charts_bivariate)

# --- Streamlit App Entry Point ---
def main():
    set_page_config_and_styles()
    st.title("🚀 AutoViz Dashboard: Visualize Data Effortlessly 🚀")

    uploaded_file = st.file_uploader("Upload a file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'json'])
    df = None
    if uploaded_file:
        # Identify file uniquely by file name
        if 'last_uploaded_file_id' not in st.session_state or uploaded_file.name != st.session_state['last_uploaded_file_id']:
            st.session_state['cleaned_df'] = None
            st.session_state['last_uploaded_file_id'] = uploaded_file.name

        df = load_uploaded_file(uploaded_file)
        if df is not None:
            st.subheader("🔍 Dataset Preview")
            st.dataframe(df.head())
            
            st.markdown("## Remove Unwanted Features")
            # Remove unwanted features in main page (not sidebar)
            columns_to_drop = st.multiselect("Select columns to REMOVE from analysis:", options=list(df.columns), key="initial_drop_cols")
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop, errors='ignore')
                st.info("Dropped columns: " + ", ".join(columns_to_drop))
                st.dataframe(df.head())

            # Clean Data button
            if st.button("✨ Clean Data", key="clean_data_button"):
                with st.spinner("Cleaning data... This may take a moment..."):
                    df = clean_data(df.copy())
                st.session_state['cleaned_df'] = df.copy()
            
            # Use cleaned data if available
            if 'cleaned_df' in st.session_state and st.session_state['cleaned_df'] is not None:
                df = st.session_state['cleaned_df']
            else:
                st.info("Click '✨ Clean Data' to prepare your dataset for analysis.")
                return

            if df is None or df.empty:
                st.warning("Please upload a file and/or ensure data cleaning did not result in an empty dataset.")
                return

            
            numeric_cols, categorical_cols, datetime_cols, geo_cols = detect_column_types(df)
            if not df.empty and (numeric_cols or categorical_cols or datetime_cols or geo_cols):
                generate_dashboard(df)
            else:
                st.warning("No usable data or columns remaining after selection/cleaning. Please adjust your selections.")
        else:
            st.error("Failed to load the uploaded file.")

    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #555555;
        text-align: center;
        padding: 8px 0;
        font-size: 14px;
        border-top: 1px solid #e6e6e6;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        Copyright © 2025 Peraisoodan Viswanath S. All rights reserved.
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

