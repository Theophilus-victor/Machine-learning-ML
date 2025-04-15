import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv('students_score.csv')

print(df.head())

df = df.dropna()

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

y_pred = model.predict(X_test_scaled)

mse_math = mean_squared_error(y_test['MathScore'], y_pred[:, 0])
mse_reading = mean_squared_error(y_test['ReadingScore'], y_pred[:, 1])
mse_writing = mean_squared_error(y_test['WritingScore'], y_pred[:, 2])

print(f'Mean Squared Error for MathScore: {mse_math}')
print(f'Mean Squared Error for ReadingScore: {mse_reading}')
print(f'Mean Squared Error for WritingScore: {mse_writing}')

score = model.score(X_test_scaled, y_test)
print(f'R^2 score: {score}')
new_student_data = [[...]]  
new_student_scaled = scaler.transform(new_student_data)
predictions = model.predict(new_student_scaled)
print(f'Predicted scores for Math, Reading, and Writing: {predictions}')
