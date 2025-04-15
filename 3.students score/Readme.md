**🚀 Student Performance Prediction**
This project leverages machine learning to predict student performance based on various factors such as gender, ethnic group, parental education, lunch type, test preparation, and more. The model uses a RandomForestRegressor to predict scores in Math, Reading, and Writing.

🎯 Features
***Input Features:***
Gender
Ethnic Group
Parent's Education Level
Lunch Type
Test Preparation
Parental Marital Status
Practice Sport
Is First Child
Transport Means
Weekly Study Hours
Number of Siblings
Target Variables:
Math Score
Reading Score
Writing Score

🧠 Model
The model uses a Random Forest Regressor to predict student scores based on the features.
Data Preprocessing: Handles missing values and performs one-hot encoding for categorical variables.
Standardization: Features are standardized using StandardScaler to ensure consistent performance across the model.

🔄 Workflow
Data Preprocessing:
Missing values are dropped from the dataset.
Categorical variables are encoded using one-hot encoding.

Model Training:
The dataset is split into training and testing sets.
The Random Forest Regressor is trained on the scaled features.

Model Evaluation:
The model is evaluated using Mean Squared Error (MSE) for each of the three target variables (Math, Reading, Writing).
The model’s performance is further validated using the R² score.

Prediction:
After training, the model can predict scores for new students based on their input features.

🛠️ How to Use
Input Data: Users can enter values for the features (e.g., gender, parental education, weekly study hours, etc.).
Prediction: After entering the data, users can click the "Predict" button to get the predicted Math, Reading, and Writing scores for the student.

📚 L# **ibraries and Tools Used**
pandas: For data handling.
scikit-learn: For machine learning model and data preprocessing.
streamlit: For building the interactive web interface.




**TO RUN** 
streamlit run app.py

**reference**
https://www.kaggle.com/datasets/desalegngeb/students-exam-scores

