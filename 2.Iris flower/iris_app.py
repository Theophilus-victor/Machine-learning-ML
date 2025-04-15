import streamlit as st
import numpy as np
from iris_classifier import predict_species

st.title("🌸 Iris Flower Species Classifier")
st.write("Enter the flower's features to predict its species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    features = np.array([sepal_length, sepal_width, petal_length, petal_width])
    species = predict_species(features)
    st.success(f"The predicted Iris species is: **{species}**")
