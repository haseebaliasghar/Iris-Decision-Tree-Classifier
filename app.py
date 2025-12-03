%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('decision_tree_iris.pkl')

st.title("Iris Flower Classifier (Task 2)")
st.write("Enter the flower details below to predict its species.")

# Input sliders matching the Iris dataset features
sepal_len = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0)
sepal_wid = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0)
petal_len = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5)
petal_wid = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

if st.button("Predict"):
    # Reshape input to 2D array
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Species: {prediction[0]}")
