import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Function to make predictions
def predict_spam_or_not_spam(text, model, vectorizer):
    # Vectorize the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Make prediction using the loaded model
    prediction = model.predict(text_vectorized)
    
    # Check the prediction
    if prediction == 1:
        return "Spam"
    else:
        return "Not Spam"



def main():
    st.title("Spam Detection App")
    st.subheader("Enter a message to check if it's spam or not")

    # Get the user input
    user_input = st.text_area("Message", "")

    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Predict on button click
    if st.button("Check"):
        if user_input:
            prediction = predict_spam_or_not_spam(user_input, model, vectorizer)
            if prediction == 1:
                st.write("This is a **Spam** message.")
            else:
                st.write("This is a **Not Spam** message.")
        else:
            st.write("Please enter a message.")

if __name__ == "__main__":
    main()
