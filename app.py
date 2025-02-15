import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# Configure App
st.set_page_config(
	page_title = "Fraud Detection App",
	page_icon="🛡️",
    layout="wide"
)

# Load Model
def load_model():
    with open("best_model.pkl", 'rb') as file:
        return pkl.load(file)
        
def generate_random_values():
    random_values = {}
    random_values['Time'] = np.random.uniform(0, 1000)
    for i in range(1, 29):
        random_values[f'V{i}'] = np.random.uniform(-10, 10)
    
    random_values['Amount'] = np.random.uniform(0, 1000000)
    
    return random_values
    
        
def make_predictions():
    st.title("Generate Prediction")
    
    if st.button("Generate Forecast"):
        st.session_state.random_values = generate_random_values()
        st.rerun()
