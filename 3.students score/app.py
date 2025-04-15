import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('students_score.csv')

df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep', 
                                 'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans', 
                                 'WklyStudyHours'])

X = df.drop(columns=['MathScore', 'ReadingScore', 'WritingScore'])
y = df[['MathScore', 'ReadingScore', 'WritingScore']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

st.title("Student Performance Prediction")

st.sidebar.header('Input Student Data')

input_dict = {
    'Gender': ['Male', 'Female'],
    'EthnicGroup': ['Group A', 'Group B', 'Group C', 'Group D', 'Group E'],
    'ParentEduc': ['some_highschool', 'highschool', 'associate_degree', 'bachelor_degree', 'master_degree'],
    'LunchType': ['standard', 'free/reduced'],
    'TestPrep': ['completed', 'none'],
    'ParentMaritalStatus': ['married', 'single', 'widowed', 'divorced'],
    'PracticeSport': ['never', 'sometimes', 'regularly'],
    'IsFirstChild': ['yes', 'no'],
    'TransportMeans': ['schoolbus', 'private'],
    'WklyStudyHours': ['less than 5hrs', 'between 5 and 10hrs', 'more than 10hrs']
}

for key, options in input_dict.items():
    if key not in st.session_state:
        st.session_state[key] = options[0]

gender = st.sidebar.selectbox('Gender', input_dict['Gender'], key='Gender')
ethnic_group = st.sidebar.selectbox('Ethnic Group', input_dict['EthnicGroup'], key='EthnicGroup')
parent_education = st.sidebar.selectbox('Parent Education Level', input_dict['ParentEduc'], key='ParentEduc')
lunch_type = st.sidebar.selectbox('Lunch Type', input_dict['LunchType'], key='LunchType')
test_prep = st.sidebar.selectbox('Test Preparation', input_dict['TestPrep'], key='TestPrep')
parent_marital_status = st.sidebar.selectbox('Parent Marital Status', input_dict['ParentMaritalStatus'], key='ParentMaritalStatus')
practice_sport = st.sidebar.selectbox('Practice Sport', input_dict['PracticeSport'], key='PracticeSport')
is_first_child = st.sidebar.selectbox('Is First Child', input_dict['IsFirstChild'], key='IsFirstChild')
nr_siblings = st.sidebar.slider('Number of Siblings', 0, 7, key='NrSiblings')
transport_means = st.sidebar.selectbox('Transport Means', input_dict['TransportMeans'], key='TransportMeans')
wkly_study_hours = st.sidebar.selectbox('Weekly Study Hours', input_dict['WklyStudyHours'], key='WklyStudyHours')

if st.sidebar.button('Predict'):
    input_data = pd.DataFrame({
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Gender_Female': [1 if gender == 'Female' else 0],
        'EthnicGroup_group A': [1 if ethnic_group == 'Group A' else 0],
        'EthnicGroup_group B': [1 if ethnic_group == 'Group B' else 0],
        'EthnicGroup_group C': [1 if ethnic_group == 'Group C' else 0],
        'EthnicGroup_group D': [1 if ethnic_group == 'Group D' else 0],
        'EthnicGroup_group E': [1 if ethnic_group == 'Group E' else 0],
        'ParentEduc_some_highschool': [1 if parent_education == 'some_highschool' else 0],
        'ParentEduc_highschool': [1 if parent_education == 'highschool' else 0],
        'ParentEduc_associate_degree': [1 if parent_education == 'associate_degree' else 0],
        'ParentEduc_bachelor_degree': [1 if parent_education == 'bachelor_degree' else 0],
        'ParentEduc_master_degree': [1 if parent_education == 'master_degree' else 0],
        'LunchType_standard': [1 if lunch_type == 'standard' else 0],
        'LunchType_free/reduced': [1 if lunch_type == 'free/reduced' else 0],
        'TestPrep_completed': [1 if test_prep == 'completed' else 0],
        'TestPrep_none': [1 if test_prep == 'none' else 0],
        'ParentMaritalStatus_married': [1 if parent_marital_status == 'married' else 0],
        'ParentMaritalStatus_single': [1 if parent_marital_status == 'single' else 0],
        'ParentMaritalStatus_widowed': [1 if parent_marital_status == 'widowed' else 0],
        'ParentMaritalStatus_divorced': [1 if parent_marital_status == 'divorced' else 0],
        'PracticeSport_never': [1 if practice_sport == 'never' else 0],
        'PracticeSport_sometimes': [1 if practice_sport == 'sometimes' else 0],
        'PracticeSport_regularly': [1 if practice_sport == 'regularly' else 0],
        'IsFirstChild_yes': [1 if is_first_child == 'yes' else 0],
        'IsFirstChild_no': [1 if is_first_child == 'no' else 0],
        'NrSiblings': [nr_siblings],
        'TransportMeans_schoolbus': [1 if transport_means == 'schoolbus' else 0],
        'TransportMeans_private': [1 if transport_means == 'private' else 0],
        'WklyStudyHours_less than 5hrs': [1 if wkly_study_hours == 'less than 5hrs' else 0],
        'WklyStudyHours_between 5 and 10hrs': [1 if wkly_study_hours == 'between 5 and 10hrs' else 0],
        'WklyStudyHours_more than 10hrs': [1 if wkly_study_hours == 'more than 10hrs' else 0]
    })

    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_data)

    predictions = model.predict(input_scaled)

    predicted_math_score = predictions[0][0]
    predicted_reading_score = predictions[0][1]
    predicted_writing_score = predictions[0][2]

    st.subheader('Predicted Student Scores')
    st.write(f'Math Score: {predicted_math_score:.2f}')
    st.write(f'Reading Score: {predicted_reading_score:.2f}')
    st.write(f'Writing Score: {predicted_writing_score:.2f}')
