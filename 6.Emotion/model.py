import pandas as pd
import string
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered)

def train_and_save():
    df = pd.read_csv("emotion_dataset.csv")
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["emotion"]

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_and_save()
