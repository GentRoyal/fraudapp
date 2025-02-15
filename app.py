import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        
    with st.form("Prediction Form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        features = {}
        
        # Time and Amount
        with col1:
            features['Time'] = st.number_input("Transaction Time (seconds)",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get('Time', 0.0)
                                        )
            features['Amount'] = st.number_input("Transaction Amount ($)",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get('Amount', 0.0)
                                        )
        # V1 - V7
        with col2:
            for i in range(1, 8):
                features[f'V{i}'] = st.number_input(f"V{i}",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get(f"V{i}", 0.0)
                                        )
                                        
        # V8 - V14
        with col3:
            for i in range(8, 15):
                features[f'V{i}'] = st.number_input(f"V{i}",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get(f"V{i}", 0.0)
                                        )
        
        # V15 - V21
        with col4:
            for i in range(15, 22):
                features[f'V{i}'] = st.number_input(f"V{i}",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get(f"V{i}", 0.0)
                                        )
    
        with col5:
            for i in range(23, 28):
                features[f'V{i}'] = st.number_input(f"V{i}",
                                                min_value = 0.0,
                                                value = st.session_state.random_values.get(f"V{i}", 0.0)
                                        )
    
    
def main():
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio("Select", ["Prediction Page", "Some Other Page"])
    
    if page == "Prediction Page":
        make_predictions()
    elif page == "Some Other Page":
        other_page()

if __name__ == "__main__":
    main()
