import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('iris.csv')
df = df.drop(columns=['Id'])

X = df.drop(columns=['Species'])
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)

def predict_species(features):
    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)[0]
    return prediction
