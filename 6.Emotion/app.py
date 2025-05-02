import streamlit as st
import pickle
import nltk
import string
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="Emotion Detection App", layout="centered", initial_sidebar_state="collapsed")

with st.container():
    st.markdown(
        """
        <style>
            body {
                background-color: #0e1117;
                color: white;
            }
            .stButton>button {
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("😎 Emotion Detection from Text")
    st.subheader("Type something and let me guess how you're feeling 🧠")

    user_input = st.text_area("💬 Enter your message here:")

    if st.button("🔍 Detect Emotion"):
        if user_input.strip() == "":
            st.warning("Please enter some text da.")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            st.success(f"🔮 Predicted Emotion: {prediction}")
