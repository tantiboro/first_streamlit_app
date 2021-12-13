import streamlit as st
from model_page import show_model_page
from explore_page import show_explore_page


page = st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_model_page()
else:
    show_explore_page()