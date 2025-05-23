import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input {
            background-color: white;
            color: black;
        }
        div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px;
            color: white;
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

# --- Data Cleaning ---
def clean_data(df_input):
    """Performs data cleaning steps on a DataFrame."""
    df = df_input.copy()
    
    # Drop empty or constant columns
    constant_cols = df.columns[df.nunique() <= 1].tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing numeric values with median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values with mode or 'unknown'
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
    
    st.success(f"Data cleaning completed! Cleaned shape: {df.shape}")
    return df

# --- Dashboard Execution ---
def generate_dashboard(df):
    """Orchestrates the display of dataset preview and feature removal before cleaning."""
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Feature Removal Section BELOW dataset preview
    st.subheader("🚀 Unwanted Feature Removal")
    columns_to_drop = st.multiselect("Select columns to remove from analysis:", options=list(df.columns), key="feature_removal")

    # Condition: Disable the Clean Data button UNTIL user interacts with feature selection
    clean_disabled = not columns_to_drop  # Only enable button when user selects at least one feature

    if not clean_disabled:
        if st.button("✨ Clean Data", key="clean_data_button"):
            with st.spinner("Cleaning data..."):
                df = df.drop(columns=columns_to_drop)
                df = clean_data(df)
                st.session_state['cleaned_df'] = df.copy()
    
    if 'cleaned_df' in st.session_state and st.session_state['cleaned_df'] is not None:
        df = st.session_state['cleaned_df']

    return df

# --- Streamlit App Entry Point ---
def main():
    set_page_config_and_styles()
    st.title("🚀 AutoViz Dashboard: Visualize Data Effortlessly 🚀")

    uploaded_file = st.file_uploader("Upload a file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'json'])
    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        if df is not None:
            df = generate_dashboard(df)
        else:
            st.error("Failed to load the uploaded file.")

    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #f0f2f6; text-align: center; padding: 8px;">
        Copyright © 2025 Peraisoodan Viswanath S. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

