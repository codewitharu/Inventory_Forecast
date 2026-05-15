import pandas as pd
import streamlit as st
from config import REQUIRED_COLS, CACHE_KEY_DATA


def load_and_validate(uploaded_file):
    """Load CSV, validate columns, parse dates."""
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception as e:
        return None, f"Error reading file: {e}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"

    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    except Exception as e:
        return None, f"Date parsing failed: {e}"

    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df, None


def save_to_session(df):
    st.session_state[CACHE_KEY_DATA] = df


def load_from_session():
    return st.session_state.get(CACHE_KEY_DATA, None)


def clear_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
