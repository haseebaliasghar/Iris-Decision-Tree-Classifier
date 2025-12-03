import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# The file 'decision_tree_iris.pkl' must be in the same folder as this script
try:
    model = joblib.load('decision_tree_iris.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'decision_tree_iris.pkl' not found. Please upload it to your GitHub repo.")
    st.stop()

# 2. App Title & Description
st.title("Iris Species Predictor")
st.markdown("Enter the measurements of the iris flower to predict its species.")

# 3. Sidebar for Inputs
st.sidebar.header("Input Features")

def user_input_features():
    # We set min/max values based on typical Iris dataset ranges
    sepal_len = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_wid = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_len = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
    petal_wid = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    
    # Store in a numpy array matching the training format
    data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    return data

input_df = user_input_features()

# 4. Display User Input
st.subheader('User Input parameters')
st.write(f"**Sepal Length:** {input_df[0][0]} cm")
st.write(f"**Sepal Width:** {input_df[0][1]} cm")
st.write(f"**Petal Length:** {input_df[0][2]} cm")
st.write(f"**Petal Width:** {input_df[0][3]} cm")

# 5. Prediction Logic
if st.button("Classify"):
    prediction = model.predict(input_df)
    
    # The model was trained on string labels (e.g., 'Iris-setosa'), 
    # so we can print the result directly.
    st.subheader('Prediction')
    st.success(f"The flower is likely: **{prediction[0]}**")
