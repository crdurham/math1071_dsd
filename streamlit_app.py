import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
from db import load_comments, clear_comments
import matplotlib.pyplot as plt

st.title("Math 1071: Data Science Demos")

st.text("Navigate to the appropriate demo in the sidebar. Read through the content and utilize the interactive elements. " \
        "Lastly, remember to complete the associated problem sheet in HuskyCT!")
ROOT = Path(__file__).resolve().parent
NYSE_DATA_FILE = ROOT / "data" / "NYSE.csv"

nyse = pd.read_csv(NYSE_DATA_FILE)
nyse.columns = nyse.columns.str.strip() 
nyse['Date'] = pd.to_datetime(nyse['Date'])
nyse = nyse.set_index("Date")


COMMENTS_FILE = Path("comments.csv")

# --- INITIALIZE STORAGE ---
if "comments_df" not in st.session_state:
    if "comments_df" not in st.session_state:
        if COMMENTS_FILE.exists() and os.path.getsize(COMMENTS_FILE) > 0:
                df = pd.read_csv(COMMENTS_FILE)
                
                df = df.reindex(columns=["name", "timestamp", "comment"])
                st.session_state.comments_df = df

    else:
        st.session_state.comments_df = pd.DataFrame(columns=["name", "timestamp", "comment"])

st.sidebar.markdown("### Instructor Access")

password = st.sidebar.text_input("Enter password:", type="password")
INSTRUCTOR_PASSWORD = "3.1415926535//10_71?$teaching*&isn't@fun"

if password == INSTRUCTOR_PASSWORD:
    st.sidebar.success("Welcome, Cole.")
    df = load_comments()
    if not df.empty:
        st.sidebar.dataframe(df)
        st.sidebar.download_button(
            label="Download Comments",
            data=df.to_csv(index=False),
            file_name="comments.csv",
            mime="text/csv"
        )
        if st.sidebar.button("Clear Comments DB"):
            clear_comments()
        st.sidebar.success("Comments database cleared!")

    else:
        st.sidebar.info("No comments submitted yet.")
elif password:
    st.sidebar.error("Incorrect password")


st.markdown("#### Other Visuals and Interactives")
st.write("More will be added over time!")


st.markdown("""
            1. ##### NYSE Closing Price Over Time
            Enter two day/month/year combinations to see the market movement over that time frame.
            """)

with st.form(key="nyse_range"):
    start_date = st.date_input("Start date", value=nyse.index.min().date(), min_value=nyse.index.min().date(),
    max_value=nyse.index.max().date())
    end_date = st.date_input("End date", value=nyse.index.max().date(),min_value=nyse.index.min().date(),
    max_value=nyse.index.max().date())
    submit = st.form_submit_button("Plot")

    if submit:
        if start_date > end_date:
            st.error("Start date must be before end date")
        else:
            filtered = nyse.loc[start_date:end_date]
            st.line_chart(filtered["Close"])
    

st.write("To see the numeric values from the above plot:")
with st.expander("NYSE Data", expanded=False):
    st.write(nyse[(nyse.index.date >= start_date) & (nyse.index.date<=end_date)])    