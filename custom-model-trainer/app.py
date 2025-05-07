import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Trainer", layout="centered")

st.title("🧠 Custom ML Model Trainer")
st.write("Upload your dataset and train a machine learning model easily!")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of dataset", df.head())

    # Step 2: Select target column
    target_column = st.selectbox("Select the target column", df.columns)

    # Step 3: Select features (auto-exclude target)
    features = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])
    
    if features and target_column:
        X = df[features]
        y = df[target_column]

        # Step 4: Train-test split
        test_size = st.slider("Test size (percentage)", min_value=10, max_value=50, value=20, step=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Step 5: Model selection
        model_option = st.selectbox("Choose a model", ["Logistic Regression", "Naive Bayes", "SVM"])
        
        if st.button("Train Model"):
            if model_option == "Logistic Regression":
                model = LogisticRegression()
            elif model_option == "Naive Bayes":
                model = MultinomialNB()
            elif model_option == "SVM":
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"✅ Model trained successfully!")
            st.write(f"📊 **Accuracy**: {accuracy * 100:.2f}%")

            # Optional: Accuracy bar chart
            fig, ax = plt.subplots()
            ax.bar(["Accuracy"], [accuracy], color="skyblue")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

