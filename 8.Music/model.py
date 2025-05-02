import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess dataset
def load_data():
    df = pd.read_csv('data/music_genres.csv')
    
    # Assuming dataset has columns: 'tempo', 'loudness', 'danceability', 'energy', 'key', 'genre'
    features = df[['tempo', 'loudness', 'danceability', 'energy', 'key']]
    labels = df['genre']
    
    return features, labels

# Train model and save it
def train_model():
    X, y = load_data()
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully.")

# Load the trained model and scaler
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Predict genre for new data
def predict_genre(model, scaler, features):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Predict using the model
    prediction = model.predict(features_scaled)
    return prediction[0]

# Uncomment the next line to train and save the model
# train_model()
