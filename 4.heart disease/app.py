import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set dark mode
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #03DAC5;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stSidebar {
            background-color: #1F1F1F;
        }
        .css-1d391kg, .css-1v0mbdj p {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#BB86FC;text-align:center;'>💓 Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app uses <b style='color:#03DAC5;'>Random Forest Classifier</b> to predict the likelihood of heart disease based on patient data.", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

st.sidebar.header("🧾 Enter Patient Details")

def user_input():
    age = st.sidebar.slider("Age", 29, 77, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.slider("Slope of Peak Exercise ST", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.sidebar.slider("Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)", 0, 2, 1)

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([data])

input_df = user_input()
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)

st.markdown(f"""
    <div style='background-color:#1E1E1E;padding:20px;border-radius:10px'>
        <h3 style='color:#03DAC5;'>🎯 Prediction:</h3>
        <p style='font-size:20px;'>
        <b style='color:#BB86FC;'>{'💔 Has Heart Disease' if prediction == 1 else '❤️ No Heart Disease'}</b>
        </p>
        <h4 style='color:#03DAC5;'>🔎 Confidence: {np.max(prediction_proba)*100:.2f}%</h4>
    </div>
""", unsafe_allow_html=True)

if st.button("📊 Show Model Accuracy & Metrics"):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**✅ Accuracy:** <span style='color:#B2FF59'>{acc:.2f}</span>", unsafe_allow_html=True)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

if st.button("📈 Show Feature Importance"):
    st.subheader("Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importance, y=importance.index, palette="viridis")
    st.pyplot(fig)
