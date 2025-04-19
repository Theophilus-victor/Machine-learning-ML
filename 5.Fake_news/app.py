import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("🧠 Fake News Detection App")
st.subheader("Enter any news content below 👇")

# Input
news_input = st.text_area("News Content", height=200)

# Predict Button
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        transformed_input = vectorizer.transform([news_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.error("⚠️ This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")
