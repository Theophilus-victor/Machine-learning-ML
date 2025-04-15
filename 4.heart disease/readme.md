# ❤️ Heart Disease Prediction App

A machine learning-based web application that predicts the likelihood of heart disease using patient data. Built using **Random Forest Classifier** and **Streamlit** with a dark-themed UI for a modern and clean experience.

---

## 📌 Features

The model uses the following input features for prediction:

- Age  
- Sex  
- Chest Pain Type (`cp`)  
- Resting Blood Pressure (`trestbps`)  
- Serum Cholesterol (`chol`)  
- Fasting Blood Sugar (`fbs`)  
- Resting ECG Results (`restecg`)  
- Maximum Heart Rate (`thalach`)  
- Exercise-Induced Angina (`exang`)  
- ST Depression (`oldpeak`)  
- Slope of ST Segment (`slope`)  
- Number of Major Vessels (`ca`)  
- Thalassemia (`thal`)

**Target Output:**
- `0`: Not likely to have heart disease  
- `1`: Likely to have heart disease

---

## 🤖 Model Used

- **Algorithm:** Random Forest Classifier  
- **Accuracy:** Approximately 67% on test data  
- **Evaluation Metric:** Accuracy Score

---

## 🔄 Workflow

1. **Data Cleaning:**  
   - Handled missing/null values (if any)  
   - Standardized numerical values using `StandardScaler`

2. **Model Training:**  
   - Data split into training and test sets  
   - Trained using `RandomForestClassifier`

3. **Model Evaluation:**  
   - Evaluated using accuracy score  
   - Shows confidence level after prediction

4. **Prediction:**  
   - Users input health metrics via Streamlit UI  
   - The model returns a prediction along with confidence

---

## 🛠️ Tools & Libraries

- Python
- pandas
- scikit-learn
- streamlit

---

## 🚀 How to Run

```bash
streamlit run app.py

***what does it means by the confidence %***

🔢 1. 95%
💡 "Model is very confident that the person has heart disease."
👉 Almost a strong positive prediction. You can trust this result quite a bit — the patterns in the input data strongly match people with heart disease in the training data. A warning sign.


🔢 2. 82%
💡 "Model is fairly confident about the person having heart disease."
👉 Still a good level of confidence. Not as strong as 95%, but the evidence is leaning towards the presence of disease. Worth a medical check-up just in case.

🔢 3. 67%
💡 "Model is moderately confident – not too strong, not too weak."
👉 Kinda in the 'might have it' zone. A few symptoms align, but not enough for a super solid prediction. Can’t ignore, but don’t panic. More info/tests needed.

🔢 4. 53%
💡 "Model is almost unsure – leaning slightly toward heart disease."
👉 Basically a coin toss. Not reliable. Could be noise or mild indicators in the data. More accurate input or deeper tests are needed.

🔢 5. 35%
💡 "Model thinks it's more likely that the person does not have heart disease."
👉 Confidence is leaning toward the healthy side. Still, not very high, so it’s just an estimate. Keep an eye on health, but it’s not a red flag yet.


