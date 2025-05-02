import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spam_dataset.csv')  # Make sure to load your CSV dataset here

# Preprocessing and splitting data
X = df['message']
y = df['label']  # Make sure your dataset has the 'label' column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Save the model and vectorizer to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Evaluate the model (optional, for checking accuracy)
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
