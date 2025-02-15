import pandas as pd
import numpy as np
import pickle
import streamlit as st
from typing import Optional

# Configure App
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'random_values' not in st.session_state:
    st.session_state.random_values = {}
if 'model' not in st.session_state:
    st.session_state.model = None

# Load Model with better error handling
def load_model() -> Optional[object]:
    """Load the pickle model if not already loaded in session state"""
    if st.session_state.model is not None:
        return st.session_state.model
        
    try:
        with open('best_model2.pkl', 'rb') as file:
            model = pickle.load(file)
            st.session_state.model = model
            return model
    except FileNotFoundError:
        st.error("Model file 'best_model2.pkl' not found. Please ensure it's in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_random_values() -> dict:
    """Generate random values for all features"""
    random_values = {}
    random_values['Time'] = np.random.uniform(0, 1000)
    for i in range(1, 29):
        random_values[f'V{i}'] = np.random.uniform(-10, 10)
    random_values['Amount'] = np.random.uniform(0, 1000000)
    return random_values

def make_prediction(features_df: pd.DataFrame) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Make prediction with error handling"""
    model = load_model()
    if model is None:
        return None, None
        
    try:
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[:, 1].round(3)
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def make_predictions():
    st.title("Generate Prediction")
    
    if st.button("Generate Random Values"):
        st.session_state.random_values = generate_random_values()
        st.rerun()
    
    with st.form("Prediction Form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        features = {}
        
        # Time and Amount
        with col1:
            features['Time'] = st.number_input("Transaction Time (seconds)",
                                             min_value=0.0,
                                             value=st.session_state.random_values.get('Time', 0.0))
            features['Amount'] = st.number_input("Transaction Amount ($)",
                                               min_value=0.0,
                                               value=st.session_state.random_values.get('Amount', 0.0))
        
        # V1 - V28
        for col, range_start, range_end in [
            (col2, 1, 8),
            (col3, 8, 15),
            (col4, 15, 22),
            (col5, 22, 29)
        ]:
            with col:
                for i in range(range_start, range_end):
                    features[f'V{i}'] = st.number_input(
                        f"V{i}",
                        min_value=-10.0,
                        max_value=10.0,
                        value=st.session_state.random_values.get(f"V{i}", 0.0)
                    )
        
        submitted = st.form_submit_button("Make Prediction", type="primary")
        
        if submitted:
            # Create DataFrame with proper column order
            features_df = pd.DataFrame([features])
            expected_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            features_df = features_df[expected_order]
            
            # Make prediction
            prediction, probability = make_prediction(features_df)
            
            if prediction is not None and probability is not None:
                st.success("Prediction made successfully!")
                st.write("Prediction:", "Fraudulent" if prediction[0] == 1 else "Legitimate")
                st.write("Fraud Probability:", f"{probability[0]:.1%}")
                
                # Display feature importance or additional insights
                st.write("Input Features:")
                st.dataframe(features_df)

def other_page():
    st.title("Other Page")
    st.write("This is another page of the application.")

def main():
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio("Select", ["Prediction Page", "Some Other Page"])
    
    if page == "Prediction Page":
        make_predictions()
    elif page == "Some Other Page":
        other_page()

if __name__ == "__main__":
    main()
