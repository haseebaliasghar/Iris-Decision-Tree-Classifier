import streamlit as st
import joblib
import numpy as np

# 1. Load the model you saved
# Make sure this filename matches exactly what is in your GitHub repo
model = joblib.load('decision_tree_iris.pkl')

# 2. App Title
st.title("Iris Flower Classifier (Task 2)")
st.write("Enter the flower details below to predict its species.")

# 3. Input Fields (Matching the 4 features of the Iris dataset)
sepal_len = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0)
sepal_wid = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0)
petal_len = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5)
petal_wid = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# 4. Prediction Logic
if st.button("Predict"):
    # Reshape input to a 2D array [1 row, 4 columns] as required by scikit-learn
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    
    # Get prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"Predicted Species: {prediction[0]}")
