import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# --- 1. Load the Saved Model ---
model_filename = 'decision_tree_iris.pkl'

try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: The file '{model_filename}' was not found. Please upload it to your GitHub repository.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- 2. App Interface ---
st.title("🌸 Iris Flower Classifier")
st.markdown("Enter the flower measurements below to predict the species.")

# Create a form for the inputs
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_len = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_wid = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        
    with col2:
        petal_len = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_wid = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        
    # Submit button
    submit_button = st.form_submit_button("Predict Species")

# --- 3. Prediction Logic ---
if submit_button:
    # Prepare the input array (reshaped for a single sample)
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    
    # Make prediction
    prediction = model.predict(input_data)
    species = prediction[0]
    
    # Display Result
    st.success(f"The predicted species is: **{species}**")
