import streamlit as st
from model import predict_genre, load_model

# Load the pre-trained model
model = load_model()

# Streamlit UI
st.title('Music Genre Classification')
st.subheader('Classify a music track based on its features.')

# Input form
tempo = st.number_input("Tempo (beats per minute)", min_value=0, max_value=300, value=120)
loudness = st.number_input("Loudness (in dB)", min_value=-100, max_value=100, value=-5)
danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
energy = st.slider("Energy", min_value=0.0, max_value=1.0, step=0.1, value=0.8)
key = st.number_input("Key (0-11 scale, with 0=Chromatic and 11=B)", min_value=0, max_value=11, value=5)

# Prediction button
if st.button("Predict Genre"):
    # Prepare input features as a list
    features = [tempo, loudness, danceability, energy, key]

    # Make prediction
    genre = predict_genre(model, features)
    st.write(f"The predicted genre is: **{genre}**")
   